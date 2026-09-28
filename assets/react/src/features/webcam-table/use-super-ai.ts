import { useCallback, useEffect, useRef, useState } from "react"
import type { DataMessage, SuperAiFrameStart } from "./data-messages"
import {
  SUPER_AI_MAX_CHUNK_LENGTH,
  SUPER_AI_MAX_CHUNKS,
  SUPER_AI_MAX_FRAME_BYTES,
} from "./data-messages"
import { canViewBoard } from "./media-policy"
import type { RoomLink } from "./room-link"
import type { LocalCamera } from "./use-local-camera"
import type { PeerConnections } from "./use-peer-connections"

export const SUPER_AI_SCAN_INTERVAL_MS = 10_000
export const SUPER_AI_TIMEOUT_MS = 12_000

interface Incoming {
  fromPeerId: string
  start: SuperAiFrameStart
  chunks: Map<number, string>
  receivedBytes: number
  timeout: number
}

function bytes(base64: string) {
  return Uint8Array.from(atob(base64), (character) => character.charCodeAt(0))
}

function base64(data: Uint8Array) {
  let text = ""
  for (const byte of data) text += String.fromCharCode(byte)
  return btoa(text)
}

async function digest(data: Uint8Array) {
  const hash = await crypto.subtle.digest("SHA-256", data)
  return [...new Uint8Array(hash)].map((byte) => byte.toString(16).padStart(2, "0")).join("")
}

/** Exchanges full-table JPEGs only between the owner and a viewer. Frames never leave either
 * browser except on the WebRTC data channel; callers run recognition locally in `onFrame`. */
export function useSuperAi(
  link: RoomLink,
  { frame, videoEnabled }: Pick<LocalCamera, "frame" | "videoEnabled">,
  { send, listen, revealTarget }: Pick<PeerConnections, "send" | "listen" | "revealTarget">,
  onFrame: (frame: { bytes: Uint8Array; width: number; height: number }) => Promise<void>,
) {
  const incomingRef = useRef<Incoming | null>(null)
  const pendingRef = useRef<{ target: string; requestId: string; timeout: number } | null>(null)
  const lastRequestRef = useRef(new Map<string, number>())
  const lastAnsweredRef = useRef(new Map<string, number>())
  const [status, setStatus] = useState<"idle" | "requesting" | "scanning" | "failed">("idle")
  const [lastCompleted, setLastCompleted] = useState<number | null>(null)

  const clearIncoming = useCallback(() => {
    const incoming = incomingRef.current
    if (incoming) window.clearTimeout(incoming.timeout)
    incomingRef.current = null
  }, [])
  const clearPending = useCallback(() => {
    const pending = pendingRef.current
    if (pending) window.clearTimeout(pending.timeout)
    pendingRef.current = null
  }, [])

  const answer = useCallback(
    async (peerId: string, requestId: string) => {
      if (!videoEnabled() || !canViewBoard(link.peerId, peerId, revealTarget())) return
      const now = Date.now()
      // A little under the viewer's own polling interval: jitter in data-channel delivery
      // between consecutive requests should never make an on-time request look "too soon".
      if (
        now - (lastAnsweredRef.current.get(peerId) ?? -Infinity) <
        SUPER_AI_SCAN_INTERVAL_MS - 2_000
      )
        return
      lastAnsweredRef.current.set(peerId, now)
      const captured = frame()
      if (!captured) return
      const data = bytes(captured.image.slice(captured.image.indexOf(",") + 1))
      if (data.byteLength > SUPER_AI_MAX_FRAME_BYTES) return
      const chunkSize = Math.min(SUPER_AI_MAX_CHUNK_LENGTH, 24 * 1024)
      const chunks = Math.ceil(data.byteLength / chunkSize)
      if (!chunks || chunks > SUPER_AI_MAX_CHUNKS) return
      const start: SuperAiFrameStart = {
        type: "super_ai_frame_start",
        requestId,
        width: captured.width,
        height: captured.height,
        bytes: data.byteLength,
        chunks,
        digest: await digest(data),
        private: !!revealTarget(),
      }
      // Capturing and hashing are asynchronous; consent may have been revoked while waiting.
      if (!videoEnabled() || !canViewBoard(link.peerId, peerId, revealTarget())) return
      if (!send(peerId, start)) return
      for (let index = 0; index < chunks; index += 1) {
        if (!videoEnabled() || !canViewBoard(link.peerId, peerId, revealTarget())) return
        if (
          !send(peerId, {
            type: "super_ai_frame_chunk",
            requestId,
            index,
            data: base64(data.slice(index * chunkSize, (index + 1) * chunkSize)),
          })
        )
          return
      }
      if (videoEnabled() && canViewBoard(link.peerId, peerId, revealTarget()))
        send(peerId, { type: "super_ai_frame_end", requestId })
    },
    [frame, link, revealTarget, send, videoEnabled],
  )

  const finish = useCallback(
    async (fromPeerId: string) => {
      const incoming = incomingRef.current
      if (
        !incoming ||
        incoming.fromPeerId !== fromPeerId ||
        incoming.chunks.size !== incoming.start.chunks
      )
        return
      const encoded = [...Array(incoming.start.chunks)].map((_, index) =>
        incoming.chunks.get(index),
      )
      if (encoded.some((chunk) => !chunk)) return clearIncoming()
      let data: Uint8Array
      try {
        data = Uint8Array.from(encoded.flatMap((chunk) => [...bytes(chunk!)]))
      } catch {
        return clearIncoming()
      }
      clearIncoming()
      const owner = link.participants.find((participant) => participant.peer_id === fromPeerId)
      if (
        data.byteLength !== incoming.start.bytes ||
        !owner ||
        owner.camera_off ||
        !canViewBoard(fromPeerId, link.peerId, owner.reveal_to) ||
        (await digest(data)) !== incoming.start.digest
      )
        return
      setStatus("scanning")
      try {
        await onFrame({ bytes: data, width: incoming.start.width, height: incoming.start.height })
        setLastCompleted(Date.now())
        setStatus("idle")
      } catch {
        setStatus("failed")
      }
    },
    [clearIncoming, link, onFrame],
  )

  const receive = useCallback(
    (fromPeerId: string, message: DataMessage) => {
      if (message.type === "super_ai_frame_request")
        return void answer(fromPeerId, message.requestId)
      if (message.type === "super_ai_frame_start") {
        const pending = pendingRef.current
        const owner = link.participants.find((participant) => participant.peer_id === fromPeerId)
        if (
          !pending ||
          pending.target !== fromPeerId ||
          pending.requestId !== message.requestId ||
          !owner ||
          owner.camera_off ||
          !canViewBoard(fromPeerId, link.peerId, owner.reveal_to)
        )
          return
        clearPending()
        clearIncoming()
        incomingRef.current = {
          fromPeerId,
          start: message,
          chunks: new Map(),
          receivedBytes: 0,
          timeout: window.setTimeout(clearIncoming, SUPER_AI_TIMEOUT_MS),
        }
      } else if (message.type === "super_ai_frame_chunk") {
        const incoming = incomingRef.current
        if (incoming?.fromPeerId !== fromPeerId || incoming.start.requestId !== message.requestId)
          return
        const owner = link.participants.find((participant) => participant.peer_id === fromPeerId)
        if (!owner || owner.camera_off || !canViewBoard(fromPeerId, link.peerId, owner.reveal_to))
          return clearIncoming()
        let chunkBytes: Uint8Array
        try {
          chunkBytes = bytes(message.data)
        } catch {
          return clearIncoming()
        }
        if (message.index >= incoming.start.chunks) return clearIncoming()
        const existing = incoming.chunks.get(message.index)
        if (existing) {
          if (existing !== message.data) clearIncoming()
          return
        }
        if (incoming.receivedBytes + chunkBytes.byteLength > incoming.start.bytes)
          return clearIncoming()
        incoming.chunks.set(message.index, message.data)
        incoming.receivedBytes += chunkBytes.byteLength
      } else if (message.type === "super_ai_frame_end") {
        void finish(fromPeerId)
      }
    },
    [answer, clearIncoming, clearPending, finish, link],
  )

  useEffect(
    () =>
      listen({
        message: receive,
        left: (peerId) => {
          if (pendingRef.current?.target === peerId) {
            clearPending()
            setStatus("failed")
          }
          if (incomingRef.current?.fromPeerId === peerId) clearIncoming()
        },
      }),
    [clearIncoming, clearPending, listen, receive],
  )
  useEffect(
    () => () => {
      clearIncoming()
      clearPending()
    },
    [clearIncoming, clearPending],
  )

  const request = useCallback(
    (target: string) => {
      if (pendingRef.current) return false
      const owner = link.participants.find((participant) => participant.peer_id === target)
      if (!owner || owner.camera_off || !canViewBoard(target, link.peerId, owner.reveal_to))
        return false
      if (target === link.peerId) return false
      const now = Date.now()
      if (now - (lastRequestRef.current.get(target) ?? -Infinity) < SUPER_AI_SCAN_INTERVAL_MS)
        return false
      const requestId = crypto.randomUUID()
      pendingRef.current = {
        target,
        requestId,
        timeout: window.setTimeout(() => {
          clearPending()
          setStatus("failed")
        }, SUPER_AI_TIMEOUT_MS),
      }
      if (!send(target, { type: "super_ai_frame_request", requestId })) {
        clearPending()
        return false
      }
      lastRequestRef.current.set(target, now)
      setStatus("requesting")
      return true
    },
    [clearPending, link, send],
  )

  const cancel = useCallback(() => {
    clearIncoming()
    clearPending()
    setStatus("idle")
  }, [clearIncoming, clearPending])

  return { request, status, lastCompleted, cancel }
}
