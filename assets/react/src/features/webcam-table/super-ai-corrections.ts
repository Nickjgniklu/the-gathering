import { useCallback, useEffect, useMemo, useState } from "react"
import type { GalleryArt, Quad } from "./recognition/pipeline"
import type { SuperAiOverlayCard } from "./super-ai"

export type SuperAiCorrection = {
  quad: Quad
  replacementId?: string
  hidden?: boolean
}

function polygonArea(points: readonly (readonly [number, number])[]) {
  return Math.abs(
    points.reduce((sum, [x, y], index) => {
      const next = points[(index + 1) % points.length]!
      return sum + x * next[1] - next[0] * y
    }, 0) / 2,
  )
}

function intersection(a: Quad, b: Quad) {
  let output: (readonly [number, number])[] = a
  const winding = b.reduce((sum, [x, y], index) => {
    const next = b[(index + 1) % b.length]!
    return sum + x * next[1] - next[0] * y
  }, 0)
  const inside = (
    point: readonly [number, number],
    start: readonly [number, number],
    end: readonly [number, number],
  ) =>
    (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]) >=
      0 ===
    winding >= 0
  const crossing = (
    from: readonly [number, number],
    to: readonly [number, number],
    start: readonly [number, number],
    end: readonly [number, number],
  ) => {
    const dx = to[0] - from[0]
    const dy = to[1] - from[1]
    const edgeX = end[0] - start[0]
    const edgeY = end[1] - start[1]
    const denominator = dx * edgeY - dy * edgeX
    const t = ((start[0] - from[0]) * edgeY - (start[1] - from[1]) * edgeX) / denominator
    return [from[0] + t * dx, from[1] + t * dy] as const
  }
  for (let index = 0; index < b.length; index++) {
    const start = b[index]!
    const end = b[(index + 1) % b.length]!
    const input = output
    output = []
    for (let pointIndex = 0; pointIndex < input.length; pointIndex++) {
      const current = input[pointIndex]!
      const previous = input[(pointIndex + input.length - 1) % input.length]!
      if (inside(current, start, end)) {
        if (!inside(previous, start, end)) output.push(crossing(previous, current, start, end))
        output.push(current)
      } else if (inside(previous, start, end)) {
        output.push(crossing(previous, current, start, end))
      }
    }
  }
  return output
}

export function quadOverlap(left: Quad, right: Quad) {
  const overlap = polygonArea(intersection(left, right))
  return overlap / (polygonArea(left) + polygonArea(right) - overlap)
}

/** Applies each viewer's spatial corrections independently of the recognizer's changing answer. */
export function applySuperAiCorrections(
  cards: SuperAiOverlayCard[],
  corrections: SuperAiCorrection[],
): { cards: SuperAiOverlayCard[]; corrections: SuperAiCorrection[] } {
  const unused = new Set(corrections.keys())
  const nextCorrections = [...corrections]
  const corrected = cards.flatMap((card) => {
    const match = [...unused]
      .map((index) => [index, quadOverlap(card.quad, corrections[index]!.quad)] as const)
      .filter(([, overlap]) => overlap >= 0.35)
      .sort((left, right) => right[1] - left[1])[0]
    if (!match) return [card]
    const correction = corrections[match[0]]!
    unused.delete(match[0])
    nextCorrections[match[0]] = { ...correction, quad: card.quad }
    if (correction.hidden) return []
    return [{ ...card, id: correction.replacementId ?? card.id }]
  })
  return { cards: corrected, corrections: nextCorrections }
}

export function useSuperAiCorrections(roomId: string, viewerPeerId: string, boardPeerId?: string) {
  const key = `the-gathering:super-ai-corrections:${roomId}:${viewerPeerId}:${boardPeerId ?? "none"}`
  const [corrections, setCorrections] = useState<SuperAiCorrection[]>(() => {
    try {
      return JSON.parse(localStorage.getItem(key) ?? "[]") as SuperAiCorrection[]
    } catch {
      return []
    }
  })
  useEffect(() => {
    try {
      setCorrections(JSON.parse(localStorage.getItem(key) ?? "[]") as SuperAiCorrection[])
    } catch {
      setCorrections([])
    }
  }, [key])
  const save = useCallback(
    (next: SuperAiCorrection[]) => {
      setCorrections(next)
      localStorage.setItem(key, JSON.stringify(next))
    },
    [key],
  )
  const apply = useCallback(
    (cards: SuperAiOverlayCard[]) => {
      const result = applySuperAiCorrections(cards, corrections)
      return result.cards
    },
    [corrections],
  )
  return useMemo(() => {
    const add = (card: SuperAiOverlayCard, correction: Omit<SuperAiCorrection, "quad">) =>
      save([
        ...corrections.filter((existing) => quadOverlap(existing.quad, card.quad) < 0.35),
        { ...correction, quad: card.quad },
      ])
    return {
      apply,
      hide: (card: SuperAiOverlayCard) => add(card, { hidden: true }),
      replace: (card: SuperAiOverlayCard, art: GalleryArt) => add(card, { replacementId: art.id }),
    }
  }, [apply, corrections, save])
}
