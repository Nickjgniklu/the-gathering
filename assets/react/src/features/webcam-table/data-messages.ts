/** Peer-to-peer messages on the WebRTC `table` data channel. Only the native camera crop RPC
 * travels peer to peer; everything shared with the table goes through the server. Anything
 * a peer sends is untrusted until `parseDataMessage` accepts it. */

export interface CaptureRequest {
  type: "capture_request"
  requestId: string
  /** The click as a fraction of the camera frame. */
  x: number
  y: number
}

export interface CaptureResponse {
  type: "capture_response"
  requestId: string
  /** JPEG data URL of the native crop around the click. */
  image: string
  nativeWidth: number
  nativeHeight: number
  cropSize: number
  clickX: number
  clickY: number
  private: boolean
  shareCorrections: boolean
}

/** Request a full board frame for viewer-local Super AI recognition. */
export interface SuperAiFrameRequest {
  type: "super_ai_frame_request"
  requestId: string
}

export interface SuperAiFrameStart {
  type: "super_ai_frame_start"
  requestId: string
  width: number
  height: number
  bytes: number
  chunks: number
  digest: string
  private: boolean
}

export interface SuperAiFrameChunk {
  type: "super_ai_frame_chunk"
  requestId: string
  index: number
  data: string
}

export interface SuperAiFrameEnd {
  type: "super_ai_frame_end"
  requestId: string
}

export type DataMessage =
  | CaptureRequest
  | CaptureResponse
  | SuperAiFrameRequest
  | SuperAiFrameStart
  | SuperAiFrameChunk
  | SuperAiFrameEnd

export const CAPTURE_IMAGE_PREFIX = "data:image/jpeg;base64,"
/** Browsers negotiate a 256 KiB SCTP message limit; a 640 px JPEG crop fits well inside it. */
export const MAX_DATA_MESSAGE_LENGTH = 256 * 1024
const MAX_REQUEST_ID_LENGTH = 64
const MAX_FRAME_SIZE = 8192
export const SUPER_AI_MAX_CHUNKS = 128
export const SUPER_AI_MAX_CHUNK_LENGTH = 32 * 1024
/** 128 chunks of 32 KiB base64 carry at most 3 MiB of decoded JPEG bytes. */
export const SUPER_AI_MAX_FRAME_BYTES = SUPER_AI_MAX_CHUNKS * ((SUPER_AI_MAX_CHUNK_LENGTH / 4) * 3)
const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/
const DIGEST = /^[a-f0-9]{64}$/

type Fields = Record<string, unknown>

function isRecord(value: unknown): value is Fields {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

function isRequestId(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= MAX_REQUEST_ID_LENGTH
}

function inRange(value: unknown, min: number, max: number): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= min && value <= max
}

function isSize(value: unknown, max: number): value is number {
  return Number.isInteger(value) && inRange(value, 1, max)
}

function isJpegDataUrl(value: unknown): value is string {
  return (
    typeof value === "string" &&
    value.length <= MAX_DATA_MESSAGE_LENGTH &&
    value.startsWith(CAPTURE_IMAGE_PREFIX) &&
    BASE64.test(value.slice(CAPTURE_IMAGE_PREFIX.length))
  )
}

function isBase64(value: unknown, maxLength: number): value is string {
  return (
    typeof value === "string" &&
    value.length > 0 &&
    value.length <= maxLength &&
    value.length % 4 === 0 &&
    BASE64.test(value)
  )
}

function parseRequest(fields: Fields): CaptureRequest | null {
  const { requestId, x, y } = fields
  if (!isRequestId(requestId) || !inRange(x, 0, 1) || !inRange(y, 0, 1)) return null
  return { type: "capture_request", requestId, x, y }
}

function parseResponse(fields: Fields): CaptureResponse | null {
  const { requestId, image, nativeWidth, nativeHeight, cropSize, clickX, clickY } = fields
  if (
    !isRequestId(requestId) ||
    !isJpegDataUrl(image) ||
    !isSize(nativeWidth, MAX_FRAME_SIZE) ||
    !isSize(nativeHeight, MAX_FRAME_SIZE) ||
    !isSize(cropSize, Math.min(nativeWidth, nativeHeight)) ||
    !inRange(clickX, 0, cropSize) ||
    !inRange(clickY, 0, cropSize) ||
    typeof fields.private !== "boolean" ||
    typeof fields.shareCorrections !== "boolean"
  )
    return null
  return {
    type: "capture_response",
    requestId,
    image,
    nativeWidth,
    nativeHeight,
    cropSize,
    clickX,
    clickY,
    private: fields.private,
    shareCorrections: fields.shareCorrections,
  }
}

function parseSuperAiRequest(fields: Fields): SuperAiFrameRequest | null {
  return isRequestId(fields.requestId)
    ? { type: "super_ai_frame_request", requestId: fields.requestId }
    : null
}

function parseSuperAiStart(fields: Fields): SuperAiFrameStart | null {
  const { requestId, width, height, bytes, chunks, digest } = fields
  const maxChunkBytes = (SUPER_AI_MAX_CHUNK_LENGTH / 4) * 3
  if (
    !isRequestId(requestId) ||
    !isSize(width, MAX_FRAME_SIZE) ||
    !isSize(height, MAX_FRAME_SIZE) ||
    !isSize(bytes, SUPER_AI_MAX_FRAME_BYTES) ||
    !isSize(chunks, SUPER_AI_MAX_CHUNKS) ||
    chunks < Math.ceil(bytes / maxChunkBytes) ||
    typeof digest !== "string" ||
    !DIGEST.test(digest) ||
    typeof fields.private !== "boolean"
  )
    return null
  return {
    type: "super_ai_frame_start",
    requestId,
    width,
    height,
    bytes,
    chunks,
    digest,
    private: fields.private,
  }
}

function parseSuperAiChunk(fields: Fields): SuperAiFrameChunk | null {
  const { requestId, index, data } = fields
  if (
    !isRequestId(requestId) ||
    !Number.isInteger(index) ||
    !inRange(index, 0, SUPER_AI_MAX_CHUNKS - 1) ||
    !isBase64(data, SUPER_AI_MAX_CHUNK_LENGTH)
  )
    return null
  return { type: "super_ai_frame_chunk", requestId, index, data }
}

function parseSuperAiEnd(fields: Fields): SuperAiFrameEnd | null {
  return isRequestId(fields.requestId)
    ? { type: "super_ai_frame_end", requestId: fields.requestId }
    : null
}

/** Returns a well-formed message with only its known fields, or null for anything else. */
export function parseDataMessage(data: unknown): DataMessage | null {
  if (typeof data !== "string" || data.length > MAX_DATA_MESSAGE_LENGTH) return null
  let value: unknown
  try {
    value = JSON.parse(data)
  } catch {
    return null
  }
  if (!isRecord(value)) return null
  if (value.type === "capture_request") return parseRequest(value)
  if (value.type === "capture_response") return parseResponse(value)
  if (value.type === "super_ai_frame_request") return parseSuperAiRequest(value)
  if (value.type === "super_ai_frame_start") return parseSuperAiStart(value)
  if (value.type === "super_ai_frame_chunk") return parseSuperAiChunk(value)
  if (value.type === "super_ai_frame_end") return parseSuperAiEnd(value)
  return null
}
