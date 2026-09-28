import { ChevronDown, ChevronUp, Eraser, X } from "lucide-react"
import { useState } from "react"
import { GameChangerBadge } from "@/components/game-changer-badge"
import { cn } from "@/lib/cn"
import { usePrintingDetails, type PrintingDetails } from "./card-details"
import type { SuperAiOverlayCard } from "./super-ai"
import { SuperAiActions } from "./super-ai-actions"
import type { BoardCard, IdentifiedCard, TableParticipant } from "./use-webcam-room"

/** Small card image for one printing, loaded from Scryfall through the server; a grey card
 * with the name stands in until it arrives (or if the printing has no image). */
export function CardThumb({
  card,
  className,
  onClick,
}: {
  card: IdentifiedCard
  className?: string
  onClick?: () => void
}) {
  const details = usePrintingDetails(card.id)
  return (
    <CardThumbView
      card={card}
      details={details.data}
      pending={details.isPending}
      className={className}
      onClick={onClick}
    />
  )
}

function CardThumbView({
  card,
  details,
  pending,
  className,
  onClick,
}: {
  card: IdentifiedCard
  details?: PrintingDetails
  pending: boolean
  className?: string
  onClick?: () => void
}) {
  const src = details?.image_uris.small ?? details?.image_uris.normal
  const image = src ? (
    <img
      src={src}
      alt={card.name}
      loading="lazy"
      decoding="async"
      className="aspect-[5/7] w-full rounded-[4.5%] bg-black object-cover"
    />
  ) : (
    <span
      className={cn(
        "grid aspect-[5/7] w-full place-items-center rounded-[4.5%] bg-white/10 p-1 text-center text-[0.55rem] leading-tight text-white/70",
        pending && "animate-pulse",
      )}
    >
      {card.name}
    </span>
  )
  const content = (
    <>
      {image}
      <span className="absolute bottom-0.5 left-0.5">
        <GameChangerBadge gameChanger={details?.game_changer} compact />
      </span>
    </>
  )
  if (!onClick) return <span className={cn("relative block", className)}>{content}</span>
  return (
    <button
      type="button"
      className={cn(
        "relative block rounded-[4.5%] outline-offset-2 transition hover:scale-105",
        className,
      )}
      onClick={onClick}
      aria-label={`Show ${card.name}`}
    >
      {content}
    </button>
  )
}

function SuperAiCardThumb({ card }: { card: SuperAiOverlayCard }) {
  const details = usePrintingDetails(card.id)
  return (
    <CardThumbView
      card={{ id: card.id, name: details.data?.name ?? "Recognized card", set: "" }}
      details={details.data}
      pending={details.isPending}
    />
  )
}

interface TrayProps {
  participant: TableParticipant
  /** Every identified card at the table; only this board's entries are shown. */
  cards: BoardCard[]
  /** Current automatic scan results for this board. They are never added to `cards`. */
  superAiCards?: SuperAiOverlayCard[]
  onPreview: (entry: BoardCard) => void
  onRemove: (id: string) => void
  /** Present only for the local seat: clearing a whole board is the owner's call. */
  onClear?: () => void
  /** Open state when this board first appears on the stage. */
  defaultExpanded?: boolean
  onExpandedChange?: (expanded: boolean) => void
  onWrongSuperAiCard?: (card: SuperAiOverlayCard) => void
  onNotSuperAiCard?: (card: SuperAiOverlayCard) => void
  onPreviewSuperAiCard?: (card: SuperAiOverlayCard) => void
  onResetSuperAiCards?: () => void
}

/** Convoke-style tray docked to the bottom of the active board: a chevron tab that unfolds a
 * translucent shelf with click history or the current Super AI result. Only click history is a
 * shared, ephemeral notion of what is on that board; scan results are local and never recorded.
 * A wrong history entry can be removed by any seat. Rulings live in the card preview, not here. */
export function BoardCardTray({
  participant,
  cards,
  superAiCards,
  onPreview,
  onRemove,
  onClear,
  defaultExpanded = false,
  onExpandedChange,
  onWrongSuperAiCard,
  onNotSuperAiCard,
  onPreviewSuperAiCard,
  onResetSuperAiCards,
}: TrayProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [mode, setMode] = useState<"history" | "scan">("history")
  const mine = cards.filter((entry) => entry.ownerPeerId === participant.peer_id)
  const scanCards = mode === "scan" ? superAiCards : undefined
  const Chevron = expanded ? ChevronDown : ChevronUp

  return (
    <section
      className="absolute inset-x-0 bottom-0 z-10 flex flex-col items-center"
      aria-label={`Cards identified on ${participant.player_name}'s board`}
    >
      <button
        type="button"
        className="flex h-6 items-center gap-1.5 rounded-t-lg border border-b-0 border-white/15 bg-black/70 px-4 text-[0.65rem] font-bold text-white/85 backdrop-blur hover:bg-black/85"
        onClick={() => {
          setExpanded(!expanded)
          onExpandedChange?.(!expanded)
        }}
        aria-expanded={expanded}
        aria-controls={`tray-${participant.peer_id}`}
      >
        <Chevron className="size-3.5" />
        {mine.length > 0 && (
          <span className="rounded bg-white/20 px-1.5 tabular-nums">{mine.length}</span>
        )}
        <span className="sr-only">
          {expanded ? "Hide" : "Show"} identified cards on {participant.player_name}'s board
        </span>
      </button>
      {expanded && (
        <div
          id={`tray-${participant.peer_id}`}
          className="w-full border-t border-white/15 bg-black/70 px-3 py-2 backdrop-blur"
        >
          {superAiCards !== undefined && (
            <div
              className="mb-2 flex justify-center gap-1"
              role="tablist"
              aria-label="Card tray mode"
            >
              <button
                type="button"
                role="tab"
                aria-selected={scanCards === undefined}
                className={cn(
                  "btn btn-xs",
                  scanCards === undefined ? "btn-primary" : "btn-ghost text-white/75",
                )}
                onClick={() => setMode("history")}
              >
                Click history
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={scanCards !== undefined}
                className={cn(
                  "btn btn-xs",
                  scanCards !== undefined ? "btn-primary" : "btn-ghost text-white/75",
                )}
                onClick={() => setMode("scan")}
              >
                Auto-identified
              </button>
              {onResetSuperAiCards && (
                <button
                  type="button"
                  className="btn btn-ghost btn-xs text-white/75"
                  onClick={onResetSuperAiCards}
                >
                  <Eraser className="size-3" /> Reset choices
                </button>
              )}
            </div>
          )}
          {scanCards !== undefined ? (
            scanCards.length === 0 ? (
              <p className="py-2 text-center text-xs text-white/60">
                No clear auto-identified matches right now.
              </p>
            ) : (
              <ul className="flex gap-2 overflow-x-auto pt-1.5" aria-label="Current auto-identified cards">
                {scanCards.map((card, index) => (
                  <li key={`${card.id}-${index}`} className="w-16 shrink-0 md:w-20">
                    {onWrongSuperAiCard && onNotSuperAiCard ? (
                      <SuperAiActions
                        card={card}
                        onWrongCard={onWrongSuperAiCard}
                        onNotCard={onNotSuperAiCard}
                        onChooseCard={onWrongSuperAiCard}
                        onPreview={onPreviewSuperAiCard}
                      >
                        <button
                          type="button"
                          className="block w-full"
                          aria-label="Auto-identified card actions"
                        >
                          <SuperAiCardThumb card={card} />
                        </button>
                      </SuperAiActions>
                    ) : (
                      <SuperAiCardThumb card={card} />
                    )}
                  </li>
                ))}
              </ul>
            )
          ) : (
            <>
              {onClear && mine.length > 0 && (
                <div className="flex justify-end">
                  <button
                    type="button"
                    className="btn btn-ghost btn-xs gap-1 text-[0.65rem] text-white/75 hover:text-error"
                    onClick={onClear}
                  >
                    <Eraser className="size-3" /> Clear cards
                  </button>
                </div>
              )}
              {mine.length === 0 ? (
                <p className="py-2 text-center text-xs text-white/60">
                  No cards identified on this board yet. Click a card on the video to identify it.
                </p>
              ) : (
                <ul className="flex gap-2 overflow-x-auto pt-1.5" aria-label="Identified cards">
                  {mine.map((entry) => (
                    <li key={entry.id} className="relative w-16 shrink-0 md:w-20">
                      <CardThumb
                        card={entry.card}
                        onClick={() => onPreview(entry)}
                        className="w-full"
                      />
                      <button
                        type="button"
                        className="absolute -top-1.5 -right-1.5 grid size-5 place-items-center rounded-full bg-error text-white shadow ring-2 ring-black/60 hover:brightness-110"
                        onClick={() => onRemove(entry.id)}
                        aria-label={`Remove ${entry.card.name}`}
                      >
                        <X className="size-3" strokeWidth={3} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </section>
  )
}
