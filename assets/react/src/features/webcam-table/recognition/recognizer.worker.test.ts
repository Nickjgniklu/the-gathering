import { afterEach, expect, it, vi } from "vite-plus/test"
import type { BundleInfo, WorkerRequest, WorkerResponse } from "./messages"
import type { Quad } from "./pipeline"

const runtime = vi.hoisted(() => ({ create: vi.fn() }))
vi.mock("onnxruntime-web/wasm", () => ({
  env: { wasm: {} },
  Tensor: class {
    constructor(
      public type: string,
      public data: unknown,
      public dims: number[],
    ) {}
  },
  InferenceSession: { create: runtime.create },
}))

afterEach(() => {
  vi.unstubAllGlobals()
  vi.resetModules()
  vi.clearAllMocks()
})

it.each([
  ["modern", "old", "extended", "tall", "right", "left"],
  [
    "modern",
    "old",
    "extended",
    "tall",
    "right",
    "left",
    "room_0",
    "room_1",
    "split_0",
    "split_1",
    "aftermath_0",
    "aftermath_1",
    "flip_0",
    "flip_1",
  ],
  ["flip_1", "room_0", "modern"],
])("passes through the graph tensor with manifest frame order %j", async (...frameNames) => {
  const vectors = {
    data: new Float32Array(frameNames.length * 128),
    dims: [frameNames.length, 128],
  }
  const search = vi.fn().mockResolvedValue({
    indices: { data: BigInt64Array.from([1n, 0n]) },
    scores: { data: Float32Array.from([0.9, 0.7]) },
  })
  const detector = vi.fn().mockResolvedValue({
    quad: { data: Float32Array.from([40, 20, 200, 20, 200, 240, 40, 240]) },
    up: { data: Float32Array.from([0, -4]) },
    centre: { data: Float32Array.from([120, 130]) },
    short: { data: Float32Array.from([160]) },
  })
  runtime.create.mockImplementation(async (bytes: Uint8Array) => ({
    run:
      bytes[0] === 1
        ? detector
        : bytes[0] === 2
          ? vi.fn().mockResolvedValue({ embeddings: vectors })
          : search,
  }))
  const front = {
    id: "uuid",
    name: "Mirror Room",
    set: "dsk",
    face: 0,
    layout: "split",
    frame: "room_0",
  }
  const back = { ...front, id: "uuid-1", name: "Fractured Realm", face: 1, frame: "room_1" }
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async (url: string) =>
        new Response(
          url.endsWith("arts.json")
            ? JSON.stringify([front, back])
            : Uint8Array.from([url.includes("detector") ? 1 : url.includes("embed") ? 2 : 3]),
        ),
    ),
  )
  const messages: WorkerResponse[] = []
  const worker: {
    postMessage: (message: WorkerResponse) => void
    onmessage?: (event: { data: WorkerRequest }) => Promise<void>
  } = {
    postMessage: (message) => messages.push(message),
  }
  vi.stubGlobal("self", worker)
  await import("./recognizer.worker")
  const bundle: BundleInfo = {
    version: "test",
    created: "test",
    constants: {
      scene: 640,
      det_input: 256,
      rotations: 4,
      refine_fill: 0.6,
      refine_min_side: 64,
      card_aspect: 1.4,
      frame_names: frameNames,
    },
    gallery: { arts: 2, dtype: "f32", embed_dim: 128, frame_penalty: 0.02, topk: 2 },
    files: {
      "manifest.json": "manifest.json",
      "arts.json": "arts.json",
      "detector.onnx": "detector.onnx",
      "embed.onnx": "embed.onnx",
      "search.onnx": "search.onnx",
    },
  }
  await worker.onmessage!({ data: { type: "load", bundle } })
  expect(messages[0]?.type).toBe("ready")
  await worker.onmessage!({
    data: {
      type: "identify",
      id: 7,
      rgba: new ArrayBuffer(32 * 32 * 4),
      width: 32,
      height: 32,
      x: 16,
      y: 16,
    },
  })
  expect(search).toHaveBeenCalledTimes(2)
  expect(search.mock.calls[1]?.[0].embeddings).toBe(vectors)
  expect(messages[1]).toMatchObject({
    type: "identified",
    id: 7,
    result: { candidates: [back, front] },
  })
})

it("loads a detection-only bundle and scans a table without any identity graphs", async () => {
  const quads = new Float32Array(40 * 4 * 2)
  // One real card at model-space (96,96)-(192,192); everything past index 0 is sorted padding.
  quads.set([96, 96, 192, 96, 192, 192, 96, 192], 0)
  const scores = new Float32Array(40)
  scores[0] = 0.9
  const tableDetector = vi
    .fn()
    .mockResolvedValue({ quads: { data: quads }, scores: { data: scores } })
  runtime.create.mockResolvedValue({ run: tableDetector })
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(Uint8Array.from([1]))),
  )
  const messages: WorkerResponse[] = []
  const worker: {
    postMessage: (message: WorkerResponse) => void
    onmessage?: (event: { data: WorkerRequest }) => Promise<void>
  } = {
    postMessage: (message) => messages.push(message),
  }
  vi.stubGlobal("self", worker)
  await import("./recognizer.worker")
  const bundle: BundleInfo = {
    version: "table-only",
    created: "test",
    files: { "table_detector.onnx": "table_detector.onnx" },
  }
  await worker.onmessage!({ data: { type: "load", bundle } })
  expect(messages[0]).toMatchObject({ type: "ready", version: "table-only", arts: 0 })
  // The warm-up call during load already exercised the graph once.
  expect(tableDetector).toHaveBeenCalledTimes(1)

  await worker.onmessage!({
    data: {
      type: "detect_table",
      id: 3,
      rgba: new ArrayBuffer(200 * 100 * 4),
      width: 200,
      height: 100,
    },
  })
  expect(tableDetector).toHaveBeenCalledTimes(2)
  const response = messages[1]
  expect(response?.type).toBe("table_detected")
  if (response?.type !== "table_detected") throw new Error("unreachable")
  // Detected in the 384x384 model space, then mapped back to the 200x100 source frame.
  expect(response.result.cards).toHaveLength(1)
  expect(response.result.cards[0]?.quad).toEqual([
    [50, 0],
    [100, 0],
    [100, 50],
    [50, 50],
  ])
  expect(response.result.cards[0]?.score).toBeCloseTo(0.9)

  await worker.onmessage!({
    data: { type: "identify", id: 4, rgba: new ArrayBuffer(16), width: 2, height: 2, x: 1, y: 1 },
  })
  expect(messages[2]).toMatchObject({
    type: "identify_failed",
    id: 4,
    message: "card identification unavailable (embedding model not published yet)",
  })
})

it("suppresses overlapping full-frame proposals by confidence", async () => {
  const worker = { postMessage: vi.fn() }
  vi.stubGlobal("self", worker)
  const { nonMaximumSuppression, quadIou } = await import("./recognizer.worker")
  const first = {
    quad: [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
    ] as Quad,
    confidence: 0.6,
  }
  const overlapping = {
    quad: [
      [1, 1],
      [11, 1],
      [11, 11],
      [1, 11],
    ] as Quad,
    confidence: 0.9,
  }
  const separate = {
    quad: [
      [20, 20],
      [30, 20],
      [30, 30],
      [20, 30],
    ] as Quad,
    confidence: 0.7,
  }
  expect(quadIou(first.quad, overlapping.quad)).toBeCloseTo(81 / 119)
  expect(nonMaximumSuppression([first, overlapping, separate], 0.45)).toEqual([
    overlapping,
    separate,
  ])
})
