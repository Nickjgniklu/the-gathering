import { useQueries } from "@tanstack/react-query"
import { useLayoutEffect, useRef, useState } from "react"
import { getPrintingDetails } from "./card-details"
import type { FullFrameIdentification, TableCard } from "./recognition/messages"
import type { Quad } from "./recognition/pipeline"

export interface SuperAiOverlayCard {
  id: string
  quad: Quad
}

/** Only replace a card with gallery art when the match is reliable enough not to mislead a viewer. */
export function overlayCardsFromScan(result: FullFrameIdentification): SuperAiOverlayCard[] {
  return result.cards.flatMap((card) => {
    const top = card.candidates[0]
    return top && top.score >= 0.8 ? [{ id: top.id, quad: card.quad }] : []
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

/** Keeps an identified card at its last position until it exits a 20%-larger prior box. */
export function stabilizeSuperAiCards(
  previous: SuperAiOverlayCard[],
  next: SuperAiOverlayCard[],
): SuperAiOverlayCard[] {
  const available = new Set(previous.keys())
  return next.map((card) => {
    const cardCenter = center(card.quad)
    const matchingIndex = [...available]
      .filter((index) => previous[index]?.id === card.id && containsWithMargin(previous[index].quad, cardCenter))
      .sort((left, right) => {
        const [leftX, leftY] = center(previous[left].quad)
        const [rightX, rightY] = center(previous[right].quad)
        return (leftX - cardCenter[0]) ** 2 + (leftY - cardCenter[1]) ** 2 - ((rightX - cardCenter[0]) ** 2 + (rightY - cardCenter[1]) ** 2)
      })[0]
    if (matchingIndex === undefined) return card
    available.delete(matchingIndex)
    return previous[matchingIndex]
  })
}

export interface Size {
  width: number
  height: number
}

export type ViewerFlip = { horizontal: boolean; vertical: boolean }

/** Maps native camera pixels onto a stage's `object-contain` video, then mirrors for the viewer. */
export function mapSourceQuad(
  quad: Quad,
  source: Size,
  stage: Size,
  flip: ViewerFlip,
): Quad {
  const scale = Math.min(stage.width / source.width, stage.height / source.height)
  const width = source.width * scale
  const height = source.height * scale
  const left = (stage.width - width) / 2
  const top = (stage.height - height) / 2
  return quad.map(([x, y]) => [
    left + (flip.horizontal ? source.width - x : x) * scale,
    top + (flip.vertical ? source.height - y : y) * scale,
  ]) as Quad
}

/** A CSS projective matrix that maps a 1px by 1px source rectangle to a destination quad. */
export function quadTransform(quad: Quad) {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = quad
  const dx1 = x1 - x2
  const dx2 = x3 - x2
  const dx3 = x0 - x1 + x2 - x3
  const dy1 = y1 - y2
  const dy2 = y3 - y2
  const dy3 = y0 - y1 + y2 - y3
  const denominator = dx1 * dy2 - dx2 * dy1
  if (Math.abs(denominator) < 0.00001) return null
  const g = (dx3 * dy2 - dx2 * dy3) / denominator
  const h = (dx1 * dy3 - dx3 * dy1) / denominator
  const a = x1 - x0 + g * x1
  const b = x3 - x0 + h * x3
  const c = x0
  const d = y1 - y0 + g * y1
  const e = y3 - y0 + h * y3
  const f = y0
  return `matrix3d(${[a, d, 0, g, b, e, 0, h, 0, 0, 1, 0, c, f, 0, 1].join(",")})`
}

export function SuperAiArt({ src, transform }: { src: string; transform: string }) {
  return (
    <img
      src={src}
      alt=""
      className="absolute top-0 left-0 h-px w-px origin-top-left"
      style={{ transform }}
    />
  )
}

function useSize(element: React.RefObject<HTMLElement | null>) {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })
  useLayoutEffect(() => {
    const target = element.current
    if (!target) return
    const update = () => {
      const { width, height } = target.getBoundingClientRect()
      setSize({ width, height })
    }
    update()
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update)
    observer?.observe(target)
    return () => observer?.disconnect()
  }, [element])
  return size
}

/** Non-interactive Scryfall art composited over cards found in a remotely scanned camera frame. */
export function SuperAiOverlay({
  cards,
  source,
  flip,
}: {
  cards: SuperAiOverlayCard[]
  source: Size | null
  flip: ViewerFlip
}) {
  const root = useRef<HTMLDivElement>(null)
  const stage = useSize(root)
  const details = useQueries({
    queries: cards.map((card) => ({
      queryKey: ["card-printings", card.id, "details"],
      queryFn: () => getPrintingDetails(card.id),
      staleTime: 60 * 60 * 1000,
    })),
  })
  if (!source || !stage.width || !stage.height)
    return <div ref={root} className="pointer-events-none absolute inset-0" />
  return (
    <div ref={root} className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      {cards.map((card, index) => {
        const transform = quadTransform(mapSourceQuad(card.quad, source, stage, flip))
        const src = details[index]?.data?.image_uris.normal
        if (!transform || !src) return null
        return <SuperAiArt key={`${card.id}-${index}`} src={src} transform={transform} />
      })}
    </div>
  )
}

/** The table detector's own box + confidence for one card, before any identification. */
export type TableDetectionCard = TableCard

/**
 * Outlines every card the table detector found, labelled with its confidence — the detector
 * has no notion of *which* card it is (that needs the embedding model, which doesn't exist
 * yet), so unlike `SuperAiOverlay` this draws boxes, not art.
 */
export function TableDetectionOverlay({
  cards,
  source,
  flip,
}: {
  cards: TableDetectionCard[]
  source: Size | null
  flip: ViewerFlip
}) {
  const root = useRef<HTMLDivElement>(null)
  const stage = useSize(root)
  if (!source || !stage.width || !stage.height)
    return <div ref={root} className="pointer-events-none absolute inset-0" />
  return (
    <div
      ref={root}
      role="img"
      aria-label={`${cards.length} card${cards.length === 1 ? "" : "s"} detected on the table`}
      className="pointer-events-none absolute inset-0 overflow-hidden"
    >
      <svg className="absolute inset-0 h-full w-full" viewBox={`0 0 ${stage.width} ${stage.height}`}>
        {cards.map((card, index) => {
          const quad = mapSourceQuad(card.quad, source, stage, flip)
          return (
            <g key={index}>
              <polygon
                points={quad.map(([x, y]) => `${x},${y}`).join(" ")}
                className="fill-accent/10 stroke-accent"
                strokeWidth={2}
              />
              <text
                x={quad[0][0]}
                y={Math.max(12, quad[0][1] - 6)}
                className="fill-accent text-xs font-semibold"
              >
                {Math.round(card.score * 100)}%
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
