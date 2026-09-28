/**
 * The glue around the three bundle graphs, ported from `ml/cardid/bundle.py` (the reference
 * runtime). Everything here is pure so it can be unit-tested; onnxruntime calls live in
 * `recognizer.worker.ts`.
 *
 * Per click: resample the SCENE px square around the click to the detector input, run the
 * detector, map its centre/short side back to image pixels, run it again on a tighter window
 * so the card fills ~60% of the input, then embed the card at the refined quad and search.
 */

/** Constants the bundle manifest ships (`manifest.constants`); names mirror the manifest. */
export interface BundleConstants {
  scene: number
  det_input: number
  rotations: number
  refine_fill: number
  refine_min_side: number
  card_aspect: number
  frame_names: string[]
}

/** Exact selectable printing; separate printed sides use face IDs. */
export interface GalleryPrinting {
  id: string
  name: string
  set: string
  collector_number?: string
  layout?: string
  /** 0 is the original printing UUID; 1 uses `<uuid>-1`. `name` is already face-specific. */
  face?: number
  lang?: string
  border_color?: string | null
  scryfall_frame?: string | null
  frame_effects?: string[]
  promo?: boolean
}

/** One embedding per illustration, in `arts.json` order (= gallery index). */
export interface GalleryArt extends GalleryPrinting {
  frame: string
  illustration_id?: string
  url?: string
  printing_count?: number
  printings?: GalleryPrinting[]
}

export interface Candidate extends GalleryArt {
  index: number
  score: number
}

export type Point = [number, number]
/** Card corners in image pixels, printed order (top-left, top-right, bottom-right, bottom-left). */
export type Quad = [Point, Point, Point, Point]

export interface RgbaImage {
  data: Uint8ClampedArray
  width: number
  height: number
}

/** `table_detector.onnx`'s fixed contract (`ml/cardid/table_detector.py`): a dense, single-pass
 * detector over a square input, trained on synthetic scenes that were always rendered square.
 * A camera frame rarely is, so it must be letterboxed — not stretched — into that square, or
 * every card's apparent aspect ratio skews with the frame's. */
export const TABLE_DETECTOR_INPUT = 384
export const TABLE_DETECTOR_MAX_DETECTIONS = 40
/** Below this, `scores` is sorted padding rather than a real (if low-confidence) card. */
export const TABLE_DETECTOR_MIN_SCORE = 0.3

export interface LetterboxTransform {
  scale: number
  offsetX: number
  offsetY: number
}

/** Fits `image` into a `size`×`size` square, preserving aspect ratio and padding with opaque
 * black. Bilinear with edge replication, mirroring `resampleWindow`'s resampling. */
export function letterboxToSquare(
  image: RgbaImage,
  size: number,
): { input: RgbaImage; transform: LetterboxTransform } {
  const scale = Math.min(size / image.width, size / image.height)
  const width = Math.max(1, Math.round(image.width * scale))
  const height = Math.max(1, Math.round(image.height * scale))
  const offsetX = Math.floor((size - width) / 2)
  const offsetY = Math.floor((size - height) / 2)
  const { data: src, width: srcWidth, height: srcHeight } = image
  const maxX = srcWidth - 1
  const maxY = srcHeight - 1
  const data = new Uint8ClampedArray(size * size * 4)
  for (let i = 3; i < data.length; i += 4) data[i] = 255
  for (let j = 0; j < height; j += 1) {
    const sy = (j + 0.5) / scale - 0.5
    const y0 = Math.min(maxY, Math.max(0, Math.floor(sy)))
    const y1 = Math.min(maxY, y0 + 1)
    const fy = Math.min(1, Math.max(0, sy - y0))
    for (let i = 0; i < width; i += 1) {
      const sx = (i + 0.5) / scale - 0.5
      const x0 = Math.min(maxX, Math.max(0, Math.floor(sx)))
      const x1 = Math.min(maxX, x0 + 1)
      const fx = Math.min(1, Math.max(0, sx - x0))
      const p00 = (y0 * srcWidth + x0) * 4
      const p01 = (y0 * srcWidth + x1) * 4
      const p10 = (y1 * srcWidth + x0) * 4
      const p11 = (y1 * srcWidth + x1) * 4
      const w00 = (1 - fx) * (1 - fy)
      const w01 = fx * (1 - fy)
      const w10 = (1 - fx) * fy
      const w11 = fx * fy
      const dst = ((j + offsetY) * size + (i + offsetX)) * 4
      for (let c = 0; c < 3; c += 1) {
        data[dst + c] = Math.round(
          (src[p00 + c] ?? 0) * w00 +
            (src[p01 + c] ?? 0) * w01 +
            (src[p10 + c] ?? 0) * w10 +
            (src[p11 + c] ?? 0) * w11,
        )
      }
    }
  }
  return { input: { data, width: size, height: size }, transform: { scale, offsetX, offsetY } }
}

/** Maps a quad from `letterboxToSquare`'s output square back to the original image's pixels. */
export function unletterboxQuad(quad: Quad, transform: LetterboxTransform): Quad {
  return quad.map(([x, y]) => [
    (x - transform.offsetX) / transform.scale,
    (y - transform.offsetY) / transform.scale,
  ]) as Quad
}

/**
 * The `side` px square around (cx, cy) resampled to a `size` px RGBA square, bilinear with
 * edge replication, plus the scale s so that window = s * (image - (cx, cy)) + size / 2.
 * Matches `cv2.warpAffine(..., borderMode=BORDER_REPLICATE)` in `synth.window_around`
 * (OpenCV treats INTER_AREA as INTER_LINEAR for affine warps).
 */
export function resampleWindow(
  image: RgbaImage,
  cx: number,
  cy: number,
  side: number,
  size: number,
): { window: Uint8ClampedArray; scale: number } {
  const { data, width, height } = image
  const scale = size / side
  const out = new Uint8ClampedArray(size * size * 4)
  const maxX = width - 1
  const maxY = height - 1
  for (let j = 0; j < size; j += 1) {
    const sy = (j - size / 2) / scale + cy
    const y0 = Math.min(maxY, Math.max(0, Math.floor(sy)))
    const y1 = Math.min(maxY, Math.max(0, y0 + 1))
    const fy = Math.min(1, Math.max(0, sy - Math.floor(sy)))
    for (let i = 0; i < size; i += 1) {
      const sx = (i - size / 2) / scale + cx
      const x0 = Math.min(maxX, Math.max(0, Math.floor(sx)))
      const x1 = Math.min(maxX, Math.max(0, x0 + 1))
      const fx = Math.min(1, Math.max(0, sx - Math.floor(sx)))
      const o = (j * size + i) * 4
      const p00 = (y0 * width + x0) * 4
      const p01 = (y0 * width + x1) * 4
      const p10 = (y1 * width + x0) * 4
      const p11 = (y1 * width + x1) * 4
      const w00 = (1 - fx) * (1 - fy)
      const w01 = fx * (1 - fy)
      const w10 = (1 - fx) * fy
      const w11 = fx * fy
      for (let c = 0; c < 3; c += 1) {
        out[o + c] = Math.round(
          (data[p00 + c] ?? 0) * w00 +
            (data[p01 + c] ?? 0) * w01 +
            (data[p10 + c] ?? 0) * w10 +
            (data[p11 + c] ?? 0) * w11,
        )
      }
      out[o + 3] = 255
    }
  }
  return { window: out, scale }
}

/** Window pixel → image pixel for a window produced by `resampleWindow`. */
export function fromWindow(
  point: Point,
  cx: number,
  cy: number,
  scale: number,
  size: number,
): Point {
  return [(point[0] - size / 2) / scale + cx, (point[1] - size / 2) / scale + cy]
}

/** Side of the second detector window: the card's long side at `refine_fill` of the input. */
export function refineSide(short: number, constants: BundleConstants): number {
  return Math.max(
    (short * constants.card_aspect) / constants.refine_fill,
    constants.refine_min_side,
  )
}

/** The detector's `up` output is a summed unit vector over the rotations; |up| / rotations
 * is the share of views that agreed the card is upright. */
export function upVote(up: [number, number], constants: BundleConstants): number {
  return Math.hypot(up[0], up[1]) / constants.rotations
}

interface ArtQuery {
  nameTokens: string[]
  set?: string
  number?: string
  lang?: string
}

/**
 * Manual lookup over the gallery for the "that's not it" case. Whitespace-separated tokens:
 * a token equal to a set code (`fin`, `set:fin`) restricts the set, a token that is a
 * collector number (`#278`, `278`, `12a`) restricts the number, everything else must appear
 * in the card name. "forest fin" therefore lists only Final Fantasy forests.
 *
 * A bare token that happens to be a set code is ambiguous: "woe strider" is the card Woe
 * Strider, not a Wilds of Eldraine search for "strider". Such queries are read both ways;
 * set-restricted hits come first, then plain name hits, with a printing whose name equals
 * the whole query ahead of everything.
 */
export function searchArts(arts: GalleryArt[], query: string, limit = 24): GalleryArt[] {
  const tokens = query.toLowerCase().trim().split(/\s+/).filter(Boolean)
  if (tokens.length === 0) return []
  const sets = new Set(arts.flatMap((art) => (art.printings ?? [art]).map((p) => p.set)))
  const withSet: ArtQuery = { nameTokens: [] }
  const asName: ArtQuery = { nameTokens: [] }
  let ambiguous = false
  for (const token of tokens) {
    if (token.startsWith("set:")) withSet.set = asName.set = token.slice(4)
    else if (token.startsWith("lang:")) withSet.lang = asName.lang = token.slice(5)
    else if (token.startsWith("#")) withSet.number = asName.number = token.slice(1)
    else if (sets.has(token) && !withSet.set) {
      withSet.set = token
      asName.nameTokens.push(token)
      ambiguous = true
    } else if (/^\d+[a-z★†]?$/.test(token) && !withSet.number) {
      withSet.number = asName.number = token
    } else {
      withSet.nameTokens.push(token)
      asName.nameTokens.push(token)
    }
  }
  const seen = new Set<string>()
  const matches = [
    ...matchArts(arts, withSet, seen, 0),
    ...(ambiguous ? matchArts(arts, asName, seen, 1) : []),
  ]
  const exact = tokens.join(" ")
  return matches
    .sort(
      (a, b) =>
        Number(a.name.toLowerCase() !== exact) - Number(b.name.toLowerCase() !== exact) ||
        a.group - b.group ||
        a.name.localeCompare(b.name) ||
        Number(a.lang !== "en") - Number(b.lang !== "en") ||
        a.set.localeCompare(b.set) ||
        collectorOrder(a.collector_number) - collectorOrder(b.collector_number) ||
        (a.lang ?? "").localeCompare(b.lang ?? ""),
    )
    .slice(0, limit)
    .map(({ group: _group, ...art }) => art)
}

function matchArts(
  arts: GalleryArt[],
  { nameTokens, set, number, lang }: ArtQuery,
  seen: Set<string>,
  group: number,
): Array<GalleryArt & { group: number }> {
  const matches: Array<GalleryArt & { group: number }> = []
  for (const art of arts) {
    for (const printing of art.printings ?? [art]) {
      if (seen.has(printing.id)) continue
      if (set && printing.set !== set) continue
      if (lang && printing.lang !== lang) continue
      if (number && (printing.collector_number ?? "").toLowerCase() !== number) continue
      const name = printing.name.toLowerCase()
      if (!nameTokens.every((token) => name.includes(token))) continue
      seen.add(printing.id)
      matches.push({ ...printing, frame: art.frame, group })
    }
  }
  return matches
}

export function galleryPrintingCaption(art: GalleryPrinting): string {
  return [
    `${art.set.toUpperCase()}${art.collector_number ? ` #${art.collector_number}` : ""}`,
    art.lang?.toUpperCase(),
    art.border_color === "borderless" ? "borderless" : undefined,
    ...(art.frame_effects ?? []),
    art.promo ? "promo" : undefined,
  ]
    .filter(Boolean)
    .join(" · ")
}

function collectorOrder(number: string | undefined): number {
  const parsed = Number.parseInt(number ?? "", 10)
  return Number.isNaN(parsed) ? Number.MAX_SAFE_INTEGER : parsed
}

/** Where the requester clicked, in pixels of the crop the camera owner returned. */
export function clickInCrop(capture: {
  nativeWidth: number
  nativeHeight: number
  cropLeft: number
  cropTop: number
  x: number
  y: number
}): Point {
  return [
    capture.x * capture.nativeWidth - capture.cropLeft,
    capture.y * capture.nativeHeight - capture.cropTop,
  ]
}
