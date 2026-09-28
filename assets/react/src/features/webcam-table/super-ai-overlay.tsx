import { useQueries } from "@tanstack/react-query"
import { useLayoutEffect, useRef, useState, type ComponentPropsWithoutRef } from "react"
import { getPrintingDetails } from "./card-details"
import type { Quad } from "./recognition/pipeline"
import type { SuperAiDetection, SuperAiOverlayCard } from "./super-ai"
import { SuperAiActions } from "./super-ai-actions"

export interface Size {
  width: number
  height: number
}

export type ViewerFlip = { horizontal: boolean; vertical: boolean }

/** Maps native camera pixels onto a stage's `object-contain` video, then mirrors for the viewer. */
export function mapSourceQuad(quad: Quad, source: Size, stage: Size, flip: ViewerFlip): Quad {
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

export function SuperAiArt({
  src,
  transform,
  className,
  style,
  ...props
}: { src: string; transform: string } & Omit<ComponentPropsWithoutRef<"img">, "src">) {
  return (
    <img
      {...props}
      data-super-ai-art
      src={src}
      alt=""
      className={`pointer-events-auto absolute top-0 left-0 h-px w-px origin-top-left ${className ?? ""}`}
      style={{ ...style, transform }}
    />
  )
}

function SuperAiOutline({ quad, name }: { quad: Quad; name: string }) {
  const [labelX, labelY] = quad.reduce(
    ([bestX, bestY], [x, y]) => (y < bestY ? [x, y] : [bestX, bestY]),
    quad[0],
  )
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full text-primary"
      aria-hidden="true"
    >
      <polygon
        points={quad.map(([x, y]) => `${x},${y}`).join(" ")}
        fill="rgba(0, 0, 0, 0.12)"
        stroke="currentColor"
        strokeWidth="3"
      />
      <text
        x={labelX}
        y={Math.max(18, labelY - 6)}
        className="fill-current text-sm font-bold"
        paintOrder="stroke"
        stroke="rgba(0, 0, 0, 0.8)"
        strokeWidth="4"
      >
        {name}
      </text>
    </svg>
  )
}

function SuperAiDetectionOutline({
  detection,
  markerId,
  quad,
}: {
  detection: SuperAiDetection
  markerId: string
  quad: Quad
}) {
  const [[x0, y0], [x1, y1]] = quad
  const label = `${Math.round(detection.confidence * 100)}%`
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full text-emerald-400"
      aria-hidden="true"
    >
      <defs>
        <marker
          id={markerId}
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="3"
          orient="auto"
          markerUnits="strokeWidth"
        >
          <path d="M 0 0 L 6 3 L 0 6 Z" fill="currentColor" />
        </marker>
      </defs>
      <polygon
        points={quad.map(([x, y]) => `${x},${y}`).join(" ")}
        fill="rgba(16, 185, 129, 0.08)"
        stroke="currentColor"
        strokeWidth="3"
      />
      <path
        d={`M ${x0} ${y0} L ${x0 + (x1 - x0) * 0.45} ${y0 + (y1 - y0) * 0.45}`}
        fill="none"
        stroke="currentColor"
        strokeWidth="5"
        strokeLinecap="round"
        markerEnd={`url(#${markerId})`}
      />
      <text
        x={x0}
        y={Math.max(18, y0 - 6)}
        className="fill-current text-sm font-bold"
        paintOrder="stroke"
        stroke="rgba(0, 0, 0, 0.8)"
        strokeWidth="4"
      >
        {label}
      </text>
    </svg>
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
  detections,
  source,
  flip,
  onWrongCard,
  onNotCard,
  onChooseCard,
  onPreviewCard,
}: {
  cards: SuperAiOverlayCard[]
  detections: SuperAiDetection[]
  source: Size | null
  flip: ViewerFlip
  onWrongCard: (card: SuperAiOverlayCard) => void
  onNotCard: (card: SuperAiOverlayCard) => void
  onChooseCard: (card: SuperAiOverlayCard) => void
  onPreviewCard: (card: SuperAiOverlayCard) => void
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
    <div ref={root} className="pointer-events-none absolute inset-0 overflow-hidden">
      {cards.map((card, index) => {
        const quad = mapSourceQuad(card.quad, source, stage, flip)
        const transform = quadTransform(quad)
        const detail = details[index]?.data
        if (!transform || !detail?.image_uris.normal) return null
        return (
          <div key={`${card.id}-${index}`}>
            <SuperAiActions
              card={card}
              onWrongCard={onWrongCard}
              onNotCard={onNotCard}
              onChooseCard={onChooseCard}
              onPreview={onPreviewCard}
              controlsStyle={{ left: quad[0][0], top: quad[0][1] }}
            >
              <SuperAiArt src={detail.image_uris.normal} transform={transform} />
            </SuperAiActions>
            <SuperAiOutline quad={quad} name={detail.name} />
          </div>
        )
      })}
      {detections.map((detection, index) => (
        <SuperAiDetectionOutline
          key={`detection-${index}`}
          detection={detection}
          markerId={`super-ai-direction-${index}`}
          quad={mapSourceQuad(detection.quad, source, stage, flip)}
        />
      ))}
    </div>
  )
}
