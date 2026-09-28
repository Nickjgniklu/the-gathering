import { useCallback, useRef, useState } from "react"
import { openCamera } from "./camera"
import type { RoomLink } from "./room-link"
import { sharesCorrections } from "./use-correction-upload"

const CROP_SIZE = 640
const SUPER_AI_LONG_EDGE = 1280

/** Replaces the camera track in every peer's sender after the local camera changes. */
export type TrackSwap = (track: MediaStreamTrack) => Promise<unknown>

function captureCrop(video: HTMLVideoElement, x: number, y: number) {
  const width = video.videoWidth
  const height = video.videoHeight
  const size = Math.min(CROP_SIZE, width, height)
  const left = Math.max(0, Math.min(width - size, Math.round(x * width - size / 2)))
  const top = Math.max(0, Math.min(height - size, Math.round(y * height - size / 2)))
  const canvas = document.createElement("canvas")
  canvas.width = size
  canvas.height = size
  canvas.getContext("2d")?.drawImage(video, left, top, size, size, 0, 0, size, size)
  return {
    image: canvas.toDataURL("image/jpeg", 0.82),
    nativeWidth: width,
    nativeHeight: height,
    cropSize: size,
    clickX: x * width - left,
    clickY: y * height - top,
    shareCorrections: sharesCorrections(),
  }
}

function captureFrame(video: HTMLVideoElement) {
  const nativeWidth = video.videoWidth
  const nativeHeight = video.videoHeight
  if (!nativeWidth || !nativeHeight) return null
  const scale = Math.min(1, SUPER_AI_LONG_EDGE / Math.max(nativeWidth, nativeHeight))
  const width = Math.max(1, Math.round(nativeWidth * scale))
  const height = Math.max(1, Math.round(nativeHeight * scale))
  const canvas = document.createElement("canvas")
  canvas.width = width
  canvas.height = height
  canvas.getContext("2d")?.drawImage(video, 0, 0, nativeWidth, nativeHeight, 0, 0, width, height)
  return { image: canvas.toDataURL("image/jpeg", 0.7), width, height }
}

/** The local camera: a black placeholder until the seat is admitted, then the chosen device,
 * plus a hidden, always-playing video that native crops are drawn from. */
export function useLocalCamera(link: RoomLink, deviceId: string, cameraEnabled: boolean) {
  const deviceIdRef = useRef(deviceId)
  const requestRef = useRef(0)
  const startedRef = useRef(false)
  const changingRef = useRef(false)
  const [cameraChanging, setCameraChanging] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [localStream, setLocalStream] = useState<MediaStream | null>(null)
  const offRef = useRef(!cameraEnabled)
  const [cameraOff, setCameraOff] = useState(!cameraEnabled)

  const stream = useCallback(() => streamRef.current, [])
  const isOff = useCallback(() => offRef.current, [])
  const videoEnabled = useCallback(() => !!streamRef.current?.getVideoTracks()[0]?.enabled, [])

  /** Negotiates a video sender before admission without asking spectators for camera
   * permission. A seated client's real camera replaces this track. */
  const startPlaceholder = useCallback(() => {
    const placeholder = document.createElement("canvas")
    placeholder.width = 1920
    placeholder.height = 1080
    placeholder.getContext("2d")?.fillRect(0, 0, 1920, 1080)
    const media = placeholder.captureStream(1)
    media.getVideoTracks().forEach((track) => {
      track.enabled = !offRef.current
    })
    streamRef.current = media
    setLocalStream(media)
    const captureVideo = document.createElement("video")
    captureVideo.muted = true
    captureVideo.playsInline = true
    captureVideo.srcObject = media
    void captureVideo.play().catch(() => {})
    videoRef.current = captureVideo
  }, [])

  const changeCamera = useCallback(
    async (nextDeviceId: string, swap: TrackSwap): Promise<boolean> => {
      if (link.spectator || changingRef.current || !streamRef.current) return false
      changingRef.current = true
      setCameraChanging(true)
      setCameraError(null)
      const request = (requestRef.current += 1)
      try {
        const media = await openCamera(nextDeviceId).catch((reason: unknown) => {
          if (
            nextDeviceId &&
            reason instanceof DOMException &&
            ["NotFoundError", "OverconstrainedError"].includes(reason.name)
          ) {
            setCameraError("Saved camera is unavailable; using the system default.")
            return openCamera("")
          }
          throw reason
        })
        if (request !== requestRef.current) {
          media.getTracks().forEach((track) => track.stop())
          return false
        }
        const previous = streamRef.current
        const track = media.getVideoTracks()[0]!
        track.enabled = previous?.getVideoTracks()[0]?.enabled ?? false
        streamRef.current = media
        setLocalStream(media)
        if (videoRef.current) {
          // Assigning srcObject pauses the element; left paused, the hidden capture video
          // would freeze on the camera's first frame and every crop would repeat it.
          videoRef.current.srcObject = media
          void videoRef.current.play().catch(() => {})
        }
        const swapped = swap(track)
        previous?.getTracks().forEach((oldTrack) => oldTrack.stop())
        deviceIdRef.current = nextDeviceId
        await swapped.catch(() => {
          setCameraError(
            "Camera changed, but a peer's video could not be updated. Try switching again.",
          )
        })
        return true
      } catch (reason) {
        setCameraError(reason instanceof Error ? reason.message : "Could not switch camera")
        return false
      } finally {
        changingRef.current = false
        setCameraChanging(false)
      }
    },
    [link],
  )

  /** Opens the saved camera the first time the seat is admitted; rejoins keep the camera. */
  const startCamera = useCallback(
    (swap: TrackSwap) => {
      if (startedRef.current) return
      startedRef.current = true
      void changeCamera(deviceIdRef.current, swap)
    },
    [changeCamera],
  )

  /** Flips the camera on or off locally and returns whether it is now off. */
  const toggleCamera = useCallback(() => {
    const next = !offRef.current
    offRef.current = next
    streamRef.current?.getVideoTracks().forEach((track) => {
      track.enabled = !next
    })
    setCameraOff(next)
    return next
  }, [])

  const crop = useCallback(
    (x: number, y: number) => (videoRef.current ? captureCrop(videoRef.current, x, y) : null),
    [],
  )
  const frame = useCallback(() => (videoRef.current ? captureFrame(videoRef.current) : null), [])

  /** Stops every local track and abandons a camera that is still opening. */
  const stop = useCallback(() => {
    requestRef.current += 1
    streamRef.current?.getTracks().forEach((track) => track.stop())
  }, [])

  return {
    localStream,
    cameraOff,
    cameraChanging,
    cameraError,
    stream,
    isOff,
    videoEnabled,
    startPlaceholder,
    changeCamera,
    startCamera,
    toggleCamera,
    crop,
    frame,
    stop,
  }
}

export type LocalCamera = ReturnType<typeof useLocalCamera>
