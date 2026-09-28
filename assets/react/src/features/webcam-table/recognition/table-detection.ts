import type { Point, Quad, RgbaImage } from "./pipeline"

export interface TableDetection {
  quad: Quad
  confidence: number
}

/** Returns overlapping row-major tile bounds that exactly cover the source image. */
export function tileBoxes(
  width: number,
  height: number,
  grid: number,
  overlap: number,
): Array<{ left: number; top: number; right: number; bottom: number }> {
  const tileWidth = width / (grid - (grid - 1) * overlap)
  const tileHeight = height / (grid - (grid - 1) * overlap)
  const stepX = tileWidth * (1 - overlap)
  const stepY = tileHeight * (1 - overlap)
  const boxes = []

  for (let row = 0; row < grid; row += 1) {
    for (let column = 0; column < grid; column += 1) {
      const left = Math.round(column * stepX)
      const top = Math.round(row * stepY)
      boxes.push({
        left,
        top,
        right: Math.min(width, Math.round(left + tileWidth)),
        bottom: Math.min(height, Math.round(top + tileHeight)),
      })
    }
  }

  return boxes
}

export function cropImage(
  image: RgbaImage,
  box: { left: number; top: number; right: number; bottom: number },
): RgbaImage {
  const width = box.right - box.left
  const height = box.bottom - box.top
  const data = new Uint8ClampedArray(width * height * 4)

  for (let y = 0; y < height; y += 1) {
    const source = ((box.top + y) * image.width + box.left) * 4
    data.set(image.data.subarray(source, source + width * 4), y * width * 4)
  }

  return { data, width, height }
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

function averageQuad(cluster: TableDetection[]): Quad {
  const weight = cluster.reduce((total, detection) => total + detection.confidence, 0)
  return [0, 1, 2, 3].map(
    (corner) =>
      [0, 1].map((axis) => {
        const coordinate = cluster.reduce(
          (total, detection) => total + detection.quad[corner][axis] * detection.confidence,
          0,
        )
        return coordinate / weight
      }) as Point,
  ) as Quad
}

/** Greedily clusters by the strongest quad, then score-weights its coordinate estimates. */
export function dedupeTableDetections(
  detections: TableDetection[],
  iouThreshold: number,
): TableDetection[] {
  const ordered = [...detections].sort((a, b) => b.confidence - a.confidence)
  const used = new Set<number>()
  const deduplicated: TableDetection[] = []

  for (const [index, seed] of ordered.entries()) {
    if (used.has(index)) continue
    used.add(index)
    const cluster = [seed]
    for (let candidateIndex = index + 1; candidateIndex < ordered.length; candidateIndex += 1) {
      const candidate = ordered[candidateIndex]!
      if (!used.has(candidateIndex) && quadIou(seed.quad, candidate.quad) > iouThreshold) {
        used.add(candidateIndex)
        cluster.push(candidate)
      }
    }
    deduplicated.push({
      quad: averageQuad(cluster),
      confidence: Math.max(...cluster.map((detection) => detection.confidence)),
    })
  }

  return deduplicated
}
