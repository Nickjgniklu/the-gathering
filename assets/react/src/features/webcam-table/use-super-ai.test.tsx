import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vite-plus/test"
import type { DataMessage } from "./data-messages"
import type { RoomLink } from "./room-link"
import type { TableParticipant } from "./room-types"
import { SUPER_AI_SCAN_INTERVAL_MS, SUPER_AI_TIMEOUT_MS, useSuperAi } from "./use-super-ai"

const owner = {
  peer_id: "owner",
  player_id: 1,
  player_name: "Owner",
  camera_off: false,
  reveal_to: null,
} as TableParticipant

function frame(bytes: Uint8Array) {
  return {
    image: `data:image/jpeg;base64,${btoa(String.fromCharCode(...bytes))}`,
    width: 2,
    height: 2,
  }
}

async function sha256(bytes: Uint8Array) {
  const hash = await crypto.subtle.digest("SHA-256", bytes)
  return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, "0")).join("")
}

function harness(participants: TableParticipant[] = [owner]) {
  const link: RoomLink = { peerId: "viewer", channel: null, spectator: false, participants }
  let listener: {
    message?: (peerId: string, message: DataMessage) => void
    left?: (peerId: string) => void
  }
  const send = vi.fn(() => true)
  const listen = vi.fn((next) => {
    listener = next
    return () => {}
  })
  const onFrame = vi.fn(async () => {})
  const hook = renderHook(() =>
    useSuperAi(
      link,
      { frame: () => frame(new Uint8Array([1, 2, 3])), videoEnabled: () => true },
      { send, listen, revealTarget: () => null } as never,
      onFrame,
    ),
  )
  return {
    hook,
    link,
    send,
    onFrame,
    message: (message: DataMessage) => act(() => listener.message?.("owner", message)),
    left: () => act(() => listener.left?.("owner")),
  }
}

beforeEach(() => {
  vi.stubGlobal("crypto", { ...crypto, subtle: crypto.subtle, randomUUID: () => "request" })
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

it("reassembles out-of-order chunks, tolerates exact duplicates, and validates SHA-256", async () => {
  const transport = harness()
  const data = new Uint8Array([1, 2, 3, 4, 5, 6])
  const digest = await sha256(data)
  act(() => expect(transport.hook.result.current.request("owner")).toBe(true))
  transport.message({
    type: "super_ai_frame_start",
    requestId: "request",
    width: 2,
    height: 3,
    bytes: 6,
    chunks: 2,
    digest,
    private: false,
  })
  transport.message({ type: "super_ai_frame_chunk", requestId: "request", index: 1, data: "BAUG" })
  transport.message({ type: "super_ai_frame_chunk", requestId: "request", index: 1, data: "BAUG" })
  transport.message({ type: "super_ai_frame_chunk", requestId: "request", index: 0, data: "AQID" })
  transport.message({ type: "super_ai_frame_end", requestId: "request" })
  await waitFor(() =>
    expect(transport.onFrame).toHaveBeenCalledWith({ bytes: data, width: 2, height: 3 }),
  )

  const bad = harness()
  act(() => expect(bad.hook.result.current.request("owner")).toBe(true))
  bad.message({
    type: "super_ai_frame_start",
    requestId: "request",
    width: 2,
    height: 3,
    bytes: 3,
    chunks: 1,
    digest: "0".repeat(64),
    private: false,
  })
  bad.message({ type: "super_ai_frame_chunk", requestId: "request", index: 0, data: "AQID" })
  bad.message({ type: "super_ai_frame_end", requestId: "request" })
  await Promise.resolve()
  expect(bad.onFrame).not.toHaveBeenCalled()
})

it("rejects conflicting duplicates and oversized reassembly", () => {
  const transport = harness()
  act(() => transport.hook.result.current.request("owner"))
  const start = {
    type: "super_ai_frame_start" as const,
    requestId: "request",
    width: 2,
    height: 3,
    bytes: 3,
    chunks: 1,
    digest: "0".repeat(64),
    private: false,
  }
  transport.message(start)
  transport.message({ type: "super_ai_frame_chunk", requestId: "request", index: 0, data: "AQID" })
  transport.message({ type: "super_ai_frame_chunk", requestId: "request", index: 0, data: "BAUG" })
  transport.message({ type: "super_ai_frame_end", requestId: "request" })
  expect(transport.onFrame).not.toHaveBeenCalled()
})

it("times out, cancels, and fails pending work when the peer leaves", () => {
  vi.useFakeTimers()
  const timeout = harness()
  act(() => timeout.hook.result.current.request("owner"))
  act(() => vi.advanceTimersByTime(SUPER_AI_TIMEOUT_MS))
  expect(timeout.hook.result.current.status).toBe("failed")

  const cancelled = harness()
  act(() => cancelled.hook.result.current.request("owner"))
  act(() => cancelled.hook.result.current.cancel())
  expect(cancelled.hook.result.current.status).toBe("idle")

  const departed = harness()
  act(() => departed.hook.result.current.request("owner"))
  departed.left()
  expect(departed.hook.result.current.status).toBe("failed")
})

it("stops accepted transfers when visibility is revoked and rate-limits requests", async () => {
  vi.useFakeTimers()
  const transport = harness()
  expect(transport.hook.result.current.request("owner")).toBe(true)
  act(() => transport.hook.result.current.cancel())
  expect(transport.hook.result.current.request("owner")).toBe(false)
  act(() => vi.advanceTimersByTime(SUPER_AI_SCAN_INTERVAL_MS))
  expect(transport.hook.result.current.request("owner")).toBe(true)

  act(() => transport.hook.result.current.cancel())
  transport.link.participants = [{ ...owner, reveal_to: "someone-else" }]
  transport.hook.rerender()
  expect(transport.hook.result.current.request("owner")).toBe(false)

  const transfer = harness()
  act(() => transfer.hook.result.current.request("owner"))
  transfer.message({
    type: "super_ai_frame_start",
    requestId: "request",
    width: 2,
    height: 3,
    bytes: 3,
    chunks: 1,
    digest: "0".repeat(64),
    private: false,
  })
  transfer.link.participants = [{ ...owner, reveal_to: "someone-else" }]
  transfer.message({ type: "super_ai_frame_chunk", requestId: "request", index: 0, data: "AQID" })
  transfer.message({ type: "super_ai_frame_end", requestId: "request" })
  await Promise.resolve()
  expect(transfer.onFrame).not.toHaveBeenCalled()
})

it("rate-limits frame responses", async () => {
  const responder = harness()
  responder.message({ type: "super_ai_frame_request", requestId: "first" })
  await waitFor(() => expect(responder.send).toHaveBeenCalledTimes(3))
  responder.message({ type: "super_ai_frame_request", requestId: "second" })
  await Promise.resolve()
  expect(responder.send).toHaveBeenCalledTimes(3)
})
