/// <reference lib="webworker" />
/**
 * Runs the published card-recognition bundle with onnxruntime-web off the main thread.
 *
 * Mirrors `ml/cardid/bundle.py` step for step (see `pipeline.ts` for the maths): two detector
 * passes, one embed pass, one gallery search. Uses the WASM backend so results match the
 * Python parity check bit-for-bit modulo float rounding; WebGPU is a later switch
 * (`onnxruntime-web/webgpu` + the jsep wasm pair) once its per-op coverage is measured.
 */
import * as ort from "onnxruntime-web/wasm"
import mjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url"
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url"
import { galleryPrintings } from "./gallery"
import type {
  BundleInfo,
  FullFrameIdentification,
  FullFrameOptions,
  Identification,
  TableDetection,
  WorkerRequest,
  WorkerResponse,
} from "./messages"
import {
  fromWindow,
  letterboxToSquare,
  refineSide,
  resampleWindow,
  searchArts,
  unletterboxQuad,
  upVote,
  TABLE_DETECTOR_INPUT,
  TABLE_DETECTOR_MAX_DETECTIONS,
  TABLE_DETECTOR_MIN_SCORE,
  type BundleConstants,
  type Candidate,
  type GalleryArt,
  type Point,
  type Quad,
  type RgbaImage,
} from "./pipeline"

// Same-origin copies of the runtime (Vite emits them as assets); the page is not
// cross-origin isolated, so a single wasm thread.
ort.env.wasm.wasmPaths = { wasm: wasmUrl, mjs: mjsUrl }
ort.env.wasm.numThreads = 1
ort.env.logLevel = "warning"

interface Loaded {
  version: string
  /** Absent when the bundle ships the table detector without the embedding/search graphs. */
  constants: BundleConstants | null
  arts: GalleryArt[]
  printings: () => Promise<GalleryArt[]>
  detector: ort.InferenceSession | null
  embed: ort.InferenceSession | null
  search: ort.InferenceSession | null
  tableDetector: ort.InferenceSession | null
}

let loaded: Loaded | null = null
const cancelled = new Set<number>()

class ScanCancelled extends Error {}

function reply(message: WorkerResponse) {
  self.postMessage(message)
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url, { credentials: "same-origin" })
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`)
  return new Uint8Array(await response.arrayBuffer())
}

async function session(url: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(await fetchBytes(url), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  })
}

async function optionalSession(url: string | undefined): Promise<ort.InferenceSession | null> {
  return url ? session(url) : null
}

async function load(bundle: BundleInfo) {
  const started = performance.now()
  const [detector, embed, search, tableDetector, artsBytes] = await Promise.all([
    optionalSession(bundle.files["detector.onnx"]),
    optionalSession(bundle.files["embed.onnx"]),
    optionalSession(bundle.files["search.onnx"]),
    optionalSession(bundle.files["table_detector.onnx"]),
    bundle.files["arts.json"] ? fetchBytes(bundle.files["arts.json"]) : null,
  ])
  const arts = artsBytes ? (JSON.parse(new TextDecoder().decode(artsBytes)) as GalleryArt[]) : []
  loaded = {
    version: bundle.version,
    constants: bundle.constants ?? null,
    arts,
    detector,
    embed,
    search,
    tableDetector,
    printings: galleryPrintings(arts, bundle.files["printings.json"]),
  }
  // The first run of each graph pays for kernel setup; do it now, not on the first click/scan.
  if (detector && embed && search && loaded.constants) {
    const size = loaded.constants.det_input
    const blank: RgbaImage = {
      data: new Uint8ClampedArray(size * size * 4).fill(255),
      width: size,
      height: size,
    }
    await identify(blank, size / 2, size / 2)
  }
  if (tableDetector) {
    const blank: RgbaImage = {
      data: new Uint8ClampedArray(TABLE_DETECTOR_INPUT * TABLE_DETECTOR_INPUT * 4).fill(255),
      width: TABLE_DETECTOR_INPUT,
      height: TABLE_DETECTOR_INPUT,
    }
    await detectTable(blank)
  }
  reply({
    type: "ready",
    version: bundle.version,
    arts: arts.length,
    ms: performance.now() - started,
  })
}

/** Card identification (single-card detect/embed/search) needs all three graphs plus the
 * manifest constants that describe the detector's windowing; a bundle may ship only the table
 * detector until the embedding model exists. */
function requireIdentification(state: Loaded) {
  const { detector, embed, search, constants } = state
  if (!detector || !embed || !search || !constants)
    throw new Error("card identification unavailable (embedding model not published yet)")
  return { detector, embed, search, constants }
}

function requireTableDetector(state: Loaded): ort.InferenceSession {
  if (!state.tableDetector) throw new Error("table detector unavailable")
  return state.tableDetector
}

async function detect(
  image: RgbaImage,
  cx: number,
  cy: number,
  side: number,
): Promise<{ quad: Quad; up: [number, number]; centre: Point; short: number }> {
  if (!loaded) throw new Error("bundle not loaded")
  const { detector, constants } = requireIdentification(loaded)
  const size = constants.det_input
  const { window, scale } = resampleWindow(image, cx, cy, side, size)
  const feeds = { window: new ort.Tensor("uint8", new Uint8Array(window.buffer), [size, size, 4]) }
  const out = await detector.run(feeds)
  const quad = out.quad?.data as Float32Array
  const up = out.up?.data as Float32Array
  const centre = out.centre?.data as Float32Array
  const short = out.short?.data as Float32Array
  const corners = [0, 1, 2, 3].map((k) =>
    fromWindow([quad[k * 2] ?? 0, quad[k * 2 + 1] ?? 0], cx, cy, scale, size),
  ) as Quad
  return {
    quad: corners,
    up: [up[0] ?? 0, up[1] ?? 0],
    centre: fromWindow([centre[0] ?? 0, centre[1] ?? 0], cx, cy, scale, size),
    short: (short[0] ?? 0) / scale,
  }
}

async function identify(image: RgbaImage, x: number, y: number): Promise<Identification> {
  if (!loaded) throw new Error("bundle not loaded")
  const { constants, embed, search } = requireIdentification(loaded)
  const { arts } = loaded
  const started = performance.now()
  const coarse = await detect(image, x, y, constants.scene)
  const fine = await detect(
    image,
    coarse.centre[0],
    coarse.centre[1],
    refineSide(coarse.short, constants),
  )
  const detected = performance.now()

  const embeddings = await embed.run({
    scene: new ort.Tensor("uint8", new Uint8Array(image.data.buffer), [
      image.height,
      image.width,
      4,
    ]),
    quad: new ort.Tensor("float32", Float32Array.from(fine.quad.flat()), [4, 2]),
  })
  const embedded = performance.now()
  const vectors = Object.values(embeddings)[0]
  if (!vectors) throw new Error("embed graph returned nothing")
  const ranked = await search.run({ embeddings: vectors })
  const finished = performance.now()

  const indices = ranked.indices?.data as BigInt64Array | Int32Array
  const scores = ranked.scores?.data as Float32Array
  const candidates: Candidate[] = []
  for (let k = 0; k < indices.length; k += 1) {
    const index = Number(indices[k])
    const art = arts[index]
    if (art) candidates.push({ ...art, index, score: scores[k] ?? 0 })
  }
  return {
    quad: fine.quad,
    upVote: upVote(fine.up, constants),
    candidates,
    timings: {
      detector: detected - started,
      embed: embedded - detected,
      search: finished - embedded,
      total: finished - started,
    },
  }
}

interface ScanProposal {
  x: number
  y: number
  quad: Quad
  confidence: number
}

function gridPoints(image: RgbaImage, scene: number): Point[] {
  // Windows overlap by half their side, so a card on a cell edge remains near the middle of a
  // neighbouring detector view. Include both bounds even when the frame is smaller than scene.
  const step = Math.max(1, Math.floor(scene / 2))
  const axis = (length: number) => {
    const last = Math.max(0, length - 1)
    const points: number[] = []
    for (let point = 0; point <= last; point += step) points.push(point)
    if (points.at(-1) !== last) points.push(last)
    return points
  }
  return axis(image.width).flatMap((x) => axis(image.height).map((y) => [x, y] as Point))
}

function quadBounds(quad: Quad) {
  const xs = quad.map(([x]) => x)
  const ys = quad.map(([, y]) => y)
  return {
    left: Math.min(...xs),
    top: Math.min(...ys),
    right: Math.max(...xs),
    bottom: Math.max(...ys),
  }
}

export function quadIou(left: Quad, right: Quad): number {
  const a = quadBounds(left)
  const b = quadBounds(right)
  const width = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left))
  const height = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top))
  const intersection = width * height
  const union =
    (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection
  return union > 0 ? intersection / union : 0
}

export function nonMaximumSuppression<T extends { quad: Quad; confidence: number }>(
  proposals: T[],
  threshold: number,
): T[] {
  const kept: T[] = []
  for (const proposal of [...proposals].sort((a, b) => b.confidence - a.confidence)) {
    if (kept.every((accepted) => quadIou(proposal.quad, accepted.quad) < threshold))
      kept.push(proposal)
  }
  return kept
}

function scanOptions(options: FullFrameOptions | undefined) {
  const confidence = (value: number | undefined, fallback: number) =>
    Math.max(0, Math.min(1, value ?? fallback))
  return {
    strategy: options?.strategy ?? "hybrid",
    minDetectorConfidence: confidence(options?.minDetectorConfidence, 0.45),
    minMatchConfidence: confidence(options?.minMatchConfidence, 0.55),
    nmsIouThreshold: confidence(options?.nmsIouThreshold, 0.45),
  }
}

function assertNotCancelled(id: number) {
  if (cancelled.has(id)) throw new ScanCancelled()
}

async function identifyFrame(
  id: number,
  image: RgbaImage,
  options?: FullFrameOptions,
): Promise<FullFrameIdentification> {
  if (!loaded) throw new Error("bundle not loaded")
  const { constants } = requireIdentification(loaded)
  const started = performance.now()
  const settings = scanOptions(options)
  const proposals: ScanProposal[] = []
  for (const [x, y] of gridPoints(image, constants.scene)) {
    assertNotCancelled(id)
    const coarse = await detect(image, x, y, constants.scene)
    assertNotCancelled(id)
    const confidence = upVote(coarse.up, constants)
    if (confidence >= settings.minDetectorConfidence) {
      proposals.push({ x, y, quad: coarse.quad, confidence })
    }
  }
  const cards: Identification[] = []
  for (const proposal of nonMaximumSuppression(proposals, settings.nmsIouThreshold)) {
    assertNotCancelled(id)
    const point =
      settings.strategy === "hybrid"
        ? proposal.quad.reduce<[number, number]>(
            ([x, y], point) => [x + point[0] / 4, y + point[1] / 4],
            [0, 0],
          )
        : [proposal.x, proposal.y]
    const result = await identify(image, point[0], point[1])
    assertNotCancelled(id)
    if (
      result.upVote >= settings.minDetectorConfidence &&
      (result.candidates[0]?.score ?? 0) >= settings.minMatchConfidence
    ) {
      cards.push(result)
    }
  }
  const deduplicated = nonMaximumSuppression(
    cards.map((card) => ({ ...card, confidence: card.candidates[0]?.score ?? 0 })),
    settings.nmsIouThreshold,
  )
  return { cards: deduplicated, totalMs: performance.now() - started }
}

/** One dense pass of `table_detector.onnx`: every card's box and score, no identity attached. */
async function detectTable(image: RgbaImage): Promise<TableDetection> {
  if (!loaded) throw new Error("bundle not loaded")
  const tableDetector = requireTableDetector(loaded)
  const started = performance.now()
  const { input, transform } = letterboxToSquare(image, TABLE_DETECTOR_INPUT)
  const feeds = {
    table: new ort.Tensor("uint8", new Uint8Array(input.data.buffer), [
      TABLE_DETECTOR_INPUT,
      TABLE_DETECTOR_INPUT,
      4,
    ]),
  }
  const out = await tableDetector.run(feeds)
  const quads = out.quads?.data as Float32Array
  const scores = out.scores?.data as Float32Array
  const cards: TableDetection["cards"] = []
  // Scores sort descending; once one drops below the cutoff, the rest is padding.
  for (let i = 0; i < TABLE_DETECTOR_MAX_DETECTIONS; i += 1) {
    const score = scores[i] ?? 0
    if (score < TABLE_DETECTOR_MIN_SCORE) break
    const quad = [0, 1, 2, 3].map((corner) => [
      quads[(i * 4 + corner) * 2] ?? 0,
      quads[(i * 4 + corner) * 2 + 1] ?? 0,
    ]) as Quad
    cards.push({ quad: unletterboxQuad(quad, transform), score })
  }
  return { cards, totalMs: performance.now() - started }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  try {
    if (request.type === "load") {
      await load(request.bundle)
    } else if (request.type === "cancel") {
      cancelled.add(request.id)
    } else if (request.type === "identify") {
      const image: RgbaImage = {
        data: new Uint8ClampedArray(request.rgba),
        width: request.width,
        height: request.height,
      }
      reply({
        type: "identified",
        id: request.id,
        result: await identify(image, request.x, request.y),
      })
    } else if (request.type === "identify_frame") {
      const image: RgbaImage = {
        data: new Uint8ClampedArray(request.rgba),
        width: request.width,
        height: request.height,
      }
      const result = await identifyFrame(request.id, image, request.options)
      if (!cancelled.has(request.id)) reply({ type: "frame_identified", id: request.id, result })
    } else if (request.type === "detect_table") {
      const image: RgbaImage = {
        data: new Uint8ClampedArray(request.rgba),
        width: request.width,
        height: request.height,
      }
      reply({ type: "table_detected", id: request.id, result: await detectTable(image) })
    } else if (request.type === "search") {
      reply({
        type: "matches",
        id: request.id,
        arts: loaded ? searchArts(await loaded.printings(), request.query) : [],
      })
    } else if (request.type === "locate") {
      const wanted = new Set(request.printingIds)
      const arts = (await loaded?.printings()) ?? []
      reply({
        type: "matches",
        id: request.id,
        arts: arts.flatMap((art) => {
          const hits = (art.printings ?? [art]).filter((printing) => wanted.has(printing.id))
          return hits.length > 0 ? [{ ...art, printings: hits }] : []
        }),
      })
    } else if (request.type === "printings") {
      const art = (await loaded?.printings())?.find((art) => art.id === request.artId)
      reply({
        type: "matches",
        id: request.id,
        arts: art
          ? (art.printings ?? [art]).map((printing) => ({ ...printing, frame: art.frame }))
          : [],
      })
    }
  } catch (error) {
    if (error instanceof ScanCancelled) return
    const message = error instanceof Error ? error.message : String(error)
    if (request.type === "load") reply({ type: "load_failed", message })
    else reply({ type: "identify_failed", id: request.id, message })
  } finally {
    if (request.type === "identify_frame") cancelled.delete(request.id)
  }
}
