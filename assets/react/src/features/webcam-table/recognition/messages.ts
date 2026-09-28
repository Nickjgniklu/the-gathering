import type { BundleConstants, Candidate, GalleryArt } from "./pipeline"

/** `GET /api/cardid/bundle`: the published bundle and where to fetch its files.
 *
 */
export interface BundleInfo {
  version: string
  created: string
  gallery?: { arts: number; dtype: string; embed_dim: number; frame_penalty: number; topk: number }
  constants?: BundleConstants
  files: Partial<
    Record<
      | "manifest.json"
      | "arts.json"
      | "detector.onnx"
      | "embed.onnx"
      | "search.onnx"
      | "printings.json"
      | "table_detector.onnx",
      string
    >
  >
}

export type WorkerRequest =
  | { type: "load"; bundle: BundleInfo }
  | {
      type: "identify"
      id: number
      /** RGBA pixels of the crop, transferred (not copied) to the worker. */
      rgba: ArrayBuffer
      width: number
      height: number
      /** The click, in crop pixels. */
      x: number
      y: number
    }
  | {
      type: "identify_frame"
      id: number
      /** RGBA pixels of the complete camera frame, transferred to the worker. */
      rgba: ArrayBuffer
      width: number
      height: number
      options?: FullFrameOptions
    }
  /** Stops a full-frame scan. The worker does not emit a result for a cancelled request. */
  | { type: "cancel"; id: number }
  | { type: "search"; id: number; query: string }
  | { type: "printings"; id: number; artId: string }
  /** Arts holding any of these printing IDs, each with `printings` narrowed to those hits. */
  | { type: "locate"; id: number; printingIds: string[] }

export interface Identification {
  quad: Quad
  /** Share of detector views that agreed the card is upright, 0–1. */
  upVote: number
  candidates: Candidate[]
  /** Milliseconds per stage, for the Connection panel and telemetry. */
  timings: { detector: number; embed: number; search: number; total: number }
}

/** A card-shaped region proposed by the table detector before identity matching. */
export interface FrameDetection {
  quad: Quad
  confidence: number
}

/** Controls the expensive board scan without exposing detector implementation details. */
export interface FullFrameOptions {
  /** `hybrid` adds detected centres back into the proposal set before recognition. */
  strategy?: "grid" | "hybrid"
  /** Minimum detector orientation agreement, from 0 through 1. Default: 0.45. */
  minDetectorConfidence?: number
  /** Minimum score of the best gallery candidate. Default: 0.55. */
  minMatchConfidence?: number
  /** Overlap at which two proposed cards are considered one card. Default: 0.45. */
  nmsIouThreshold?: number
  /** Square grid dimension for the table detector's additional overlapping passes. Default: 2. */
  tableDetectorTileGrid?: number
  /** Fraction of each neighboring detector tile that overlaps. Default: 0.2. */
  tableDetectorTileOverlap?: number
}

export interface FullFrameIdentification {
  cards: Identification[]
  /** Table-detector regions, including ones that did not produce a confident card match. */
  detections: FrameDetection[]
  /** Milliseconds spent scanning the complete frame. */
  totalMs: number
}

export type WorkerResponse =
  | { type: "ready"; version: string; arts: number; ms: number }
  | { type: "load_failed"; message: string }
  | { type: "identified"; id: number; result: Identification }
  | { type: "frame_identified"; id: number; result: FullFrameIdentification }
  | { type: "identify_failed"; id: number; message: string }
  | { type: "matches"; id: number; arts: GalleryArt[] }
