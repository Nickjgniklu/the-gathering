import type { BundleConstants, Candidate, GalleryArt, Quad } from "./pipeline"

/** `GET /api/cardid/bundle`: the published bundle and where to fetch its files.
 *
 * `gallery`/`constants` and the embedding-pipeline files are optional: a bundle may ship the
 * table detector (card boxes) before the embedding model (card identity) exists at all. */
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
  | {
      type: "detect_table"
      id: number
      /** RGBA pixels of the complete camera frame, transferred to the worker. */
      rgba: ArrayBuffer
      width: number
      height: number
    }
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
}

export interface FullFrameIdentification {
  cards: Identification[]
  /** Milliseconds spent scanning the complete frame. */
  totalMs: number
}

/** One box `table_detector.onnx` found, with no identity attached — that needs the embedding
 * model, which does not exist yet. */
export interface TableCard {
  quad: Quad
  /** Sigmoid probability this is a card, already filtered to `>= TABLE_DETECTOR_MIN_SCORE`. */
  score: number
}

export interface TableDetection {
  cards: TableCard[]
  /** Milliseconds spent on the single detector pass. */
  totalMs: number
}

export type WorkerResponse =
  | { type: "ready"; version: string; arts: number; ms: number }
  | { type: "load_failed"; message: string }
  | { type: "identified"; id: number; result: Identification }
  | { type: "frame_identified"; id: number; result: FullFrameIdentification }
  | { type: "table_detected"; id: number; result: TableDetection }
  | { type: "identify_failed"; id: number; message: string }
  | { type: "matches"; id: number; arts: GalleryArt[] }
