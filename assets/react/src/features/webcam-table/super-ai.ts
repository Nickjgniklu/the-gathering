import { CLEAR_MARGIN, isClear } from "./card-suggestions"
import type { FullFrameIdentification } from "./recognition/messages"
import type { Quad } from "./recognition/pipeline"

export interface SuperAiOverlayCard {
  id: string
  quad: Quad
}

export const SUPER_AI_MARGIN_DEFAULT = CLEAR_MARGIN
export const SUPER_AI_MARGIN_MIN = 0.02
export const SUPER_AI_MARGIN_MAX = 0.2

export function overlayCardsFromScan(
  result: FullFrameIdentification,
  minMargin: number = SUPER_AI_MARGIN_DEFAULT,
): SuperAiOverlayCard[] {
  return result.cards.flatMap((card) => {
    const top = card.candidates[0]
    return top && isClear(card.candidates, minMargin) ? [{ id: top.id, quad: card.quad }] : []
  })
}

function center(quad: Quad) {
  return quad.reduce(
    ([x, y], [pointX, pointY]) => [x + pointX / quad.length, y + pointY / quad.length] as const,
    [0, 0] as const,
  )
}

function containsWithMargin(quad: Quad, point: readonly [number, number]) {
  const xs = quad.map(([x]) => x)
  const ys = quad.map(([, y]) => y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const margin = Math.max(maxX - minX, maxY - minY) * 0.2
  return (
    point[0] >= minX - margin &&
    point[0] <= maxX + margin &&
    point[1] >= minY - margin &&
    point[1] <= maxY + margin
  )
}

export function stabilizeSuperAiCards(
  previous: SuperAiOverlayCard[],
  next: SuperAiOverlayCard[],
): SuperAiOverlayCard[] {
  const available = new Set(previous.keys())
  return next.map((card) => {
    const cardCenter = center(card.quad)
    const matchingIndex = [...available]
      .filter(
        (index) =>
          previous[index]?.id === card.id && containsWithMargin(previous[index].quad, cardCenter),
      )
      .sort((left, right) => {
        const [leftX, leftY] = center(previous[left].quad)
        const [rightX, rightY] = center(previous[right].quad)
        return (
          (leftX - cardCenter[0]) ** 2 +
          (leftY - cardCenter[1]) ** 2 -
          ((rightX - cardCenter[0]) ** 2 + (rightY - cardCenter[1]) ** 2)
        )
      })[0]
    if (matchingIndex === undefined) return card
    available.delete(matchingIndex)
    return previous[matchingIndex]
  })
}
