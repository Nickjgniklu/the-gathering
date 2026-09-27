import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vite-plus/test"
import type { WorkerResponse } from "./messages"
import { useRecognizer } from "./use-recognizer"

class FakeWorker {
  static instances: FakeWorker[] = []
  onmessage?: (event: { data: WorkerResponse }) => void
  postMessage = vi.fn()
  terminate = vi.fn()
  constructor() {
    FakeWorker.instances.push(this)
  }
  reply(data: WorkerResponse) {
    this.onmessage?.({ data })
  }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.useRealTimers()
  FakeWorker.instances = []
})

it("downloads nothing on mount and shares warmup across first actions, before timing inference", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: { version: "v1" } })))
  vi.stubGlobal("fetch", fetch)
  vi.stubGlobal("Worker", FakeWorker)
  const { result, unmount } = renderHook(useRecognizer)
  expect(result.current.state.status).toBe("idle")
  expect(fetch).not.toHaveBeenCalled()
  expect(FakeWorker.instances).toHaveLength(0)
  let identification!: ReturnType<typeof result.current.identify>
  let search!: ReturnType<typeof result.current.search>
  act(() => {
    identification = result.current.identify(
      { data: new Uint8ClampedArray(4), width: 1, height: 1 },
      0,
      0,
    )
    search = result.current.search("forest")
  })
  const worker = FakeWorker.instances[0]!
  await waitFor(() =>
    expect(worker.postMessage).toHaveBeenCalledWith({ type: "load", bundle: { version: "v1" } }),
  )
  expect(FakeWorker.instances).toHaveLength(1)
  expect(fetch).toHaveBeenCalledTimes(1)
  vi.useFakeTimers()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000)
  })
  expect(worker.postMessage).toHaveBeenCalledTimes(1)
  await act(async () => {
    worker.reply({ type: "ready", version: "v1", arts: 2, ms: 10_000 })
  })
  expect(worker.postMessage).toHaveBeenCalledTimes(3)
  worker.reply({ type: "matches", id: 2, arts: [] })
  await expect(search).resolves.toEqual([])
  const timeout = expect(identification).rejects.toThrow("no result within 2000 ms")
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2000)
  })
  await timeout
  unmount()
  expect(worker.terminate).toHaveBeenCalledOnce()
})

it("rejects waiting actions when the bundle cannot load or the table closes", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: { version: "v1" } }))),
  )
  vi.stubGlobal("Worker", FakeWorker)
  const { result, unmount } = renderHook(useRecognizer)
  let request!: ReturnType<typeof result.current.search>
  act(() => {
    request = result.current.search("forest")
  })
  await waitFor(() => expect(result.current.state.status).toBe("loading"))
  const rejected = expect(request).rejects.toThrow("missing graph")
  await act(async () => {
    FakeWorker.instances[0]!.reply({ type: "load_failed", message: "missing graph" })
  })
  await rejected
  expect(result.current.state.status).toBe("failed")
  unmount()
  const next = renderHook(useRecognizer)
  act(() => {
    request = next.result.current.search("forest")
  })
  const closed = expect(request).rejects.toThrow("table closed")
  next.unmount()
  await closed
})

it("warms the bundle once preload turns on and reuses that worker for the first click", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ data: { version: "v1" } })))
  vi.stubGlobal("fetch", fetch)
  vi.stubGlobal("Worker", FakeWorker)
  const { result, rerender } = renderHook((preload: boolean) => useRecognizer(preload), {
    initialProps: false,
  })
  expect(fetch).not.toHaveBeenCalled()
  expect(FakeWorker.instances).toHaveLength(0)
  rerender(true)
  await waitFor(() => expect(result.current.state.status).toBe("loading"))
  expect(fetch).toHaveBeenCalledTimes(1)
  const worker = FakeWorker.instances[0]!
  expect(worker.postMessage).toHaveBeenCalledWith({ type: "load", bundle: { version: "v1" } })
  rerender(true)
  rerender(false)
  rerender(true)
  await act(async () => {
    worker.reply({ type: "ready", version: "v1", arts: 2, ms: 10 })
  })
  expect(result.current.state.status).toBe("ready")
  let search!: ReturnType<typeof result.current.search>
  act(() => {
    search = result.current.search("forest")
  })
  await waitFor(() => expect(worker.postMessage).toHaveBeenCalledTimes(2))
  expect(FakeWorker.instances).toHaveLength(1)
  expect(fetch).toHaveBeenCalledTimes(1)
  worker.reply({ type: "matches", id: 1, arts: [] })
  await expect(search).resolves.toEqual([])
})
