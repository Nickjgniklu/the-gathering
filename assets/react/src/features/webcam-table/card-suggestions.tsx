import { Search, X } from "lucide-react"
import { useQuery } from "@tanstack/react-query"
import { useEffect, useRef, useState } from "react"
import type { DeckSummary } from "@/features/decks/decks"
import { GameChangerBadge } from "@/components/game-changer-badge"
import { cn } from "@/lib/cn"
import { CardHover, CommanderHover } from "@/components/card-hover"
import { inDeck } from "./deck-hint"
import type { Identification } from "./recognition/messages"
import { galleryPrintingCaption, type GalleryArt } from "./recognition/pipeline"
import type { CapturedCard } from "./use-webcam-room"

/** What recognition did with the current capture. */
export type Recognition =
  | { status: "identifying"; loading?: boolean }
  | { status: "done"; result: Identification }
  /** Recognition did not run or did not finish; the reason is shown and deck suggestions stand in. */
  | { status: "skipped"; reason: string }

/** Top-1 leads the runner-up by at least this cosine margin: it is the answer, not a guess,
 * and a plain click records it on the board without showing this panel. From `ml/` evaluation: the margin for
 * ~99% precision on real captures with the detector. */
export const CLEAR_MARGIN = 0.08

export function isClear(
  candidates: Identification["candidates"],
  margin: number = CLEAR_MARGIN,
): boolean {
  const [first, second] = candidates
  return first !== undefined && second !== undefined && first.score - second.score >= margin
}

interface Props {
  capture: CapturedCard
  playerName: string
  recognition: Recognition
  deckSuggestions: DeckSummary[]
  /** The worker has the gallery, so the manual search can run even when a click timed out. */
  gallerySearchable: boolean
  onChooseCard: (art: GalleryArt) => void
  onChooseDeck: (deckId: number) => void
  onSearch: (query: string) => Promise<GalleryArt[]>
  onPrintings?: (artId: string) => Promise<GalleryArt[]>
  galleryVersion?: string
  /** Names in the board owner's linked deck list; matching cards are marked "In deck". */
  deckNames?: ReadonlySet<string>
  onDismiss: () => void
}

function InDeckBadge() {
  return (
    <span className="badge badge-xs badge-secondary shrink-0 font-semibold whitespace-nowrap">
      In deck
    </span>
  )
}

function PrintingChoices({
  art,
  onChoose,
  onPrintings,
  version,
}: {
  art: GalleryArt
  onChoose: (art: GalleryArt) => void
  onPrintings: Props["onPrintings"]
  version: string | undefined
}) {
  const [open, setOpen] = useState(false)
  const query = useQuery({
    queryKey: ["gallery-printings", version, art.id],
    queryFn: () => onPrintings!(art.id),
    enabled: open && !art.printings && !!onPrintings,
    staleTime: Infinity,
  })
  const printings = art.printings ?? query.data ?? []
  const count = art.printing_count ?? art.printings?.length ?? 0
  if (count < 2) return null
  return (
    <details className="ml-7 text-xs" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer py-1 text-white/60 hover:text-white">
        {count} printings of {art.name}
      </summary>
      {open && !art.printings && query.isPending && <p role="status">Loading printings…</p>}
      {open && query.isError && (
        <button type="button" onClick={() => void query.refetch()}>
          Retry loading printings
        </button>
      )}
      <ul
        className="max-h-40 overflow-y-auto rounded-md border border-white/10"
        aria-label={`Printings of ${art.name}`}
      >
        {printings.map((printing) => (
          <li key={printing.id} className="grid">
            <CardHover id={printing.id} name={printing.name}>
              <button
                type="button"
                className="w-full px-2 py-1.5 text-left text-white/75 hover:bg-white/15"
                onClick={() => onChoose({ ...printing, frame: art.frame })}
              >
                {printing.name !== art.name && (
                  <span className="block font-semibold">{printing.name}</span>
                )}
                {galleryPrintingCaption(printing)}
              </button>
            </CardHover>
          </li>
        ))}
      </ul>
    </details>
  )
}

/** Floating panel over the board when a click needs a human: the recognizer was unsure (or
 * off), or the clicker held Shift to choose for themselves. Shows the native crop with the
 * detected card outlined, the numbered top five (keys 1–5), and a gallery search for the
 * "that's not it" case. Without a published bundle the seat's decks stand in. */
export function CardSuggestions({
  capture,
  playerName,
  recognition,
  deckSuggestions,
  gallerySearchable,
  onChooseCard,
  onChooseDeck,
  onSearch,
  onPrintings,
  galleryVersion,
  deckNames,
  onDismiss,
}: Props) {
  const [query, setQuery] = useState("")
  const [matches, setMatches] = useState<GalleryArt[]>([])
  const searchRef = useRef<HTMLInputElement | null>(null)
  const candidates = recognition.status === "done" ? recognition.result.candidates : []
  const clear = isClear(candidates)

  useEffect(() => {
    if (query.trim() === "") {
      setMatches([])
      return
    }
    let stale = false
    onSearch(query)
      .then((arts) => {
        if (!stale) setMatches(arts)
      })
      .catch(() => undefined)
    return () => {
      stale = true
    }
  }, [onSearch, query])

  useEffect(() => {
    function focusSearch(event: KeyboardEvent) {
      if (event.key !== "/" || event.target instanceof HTMLInputElement) return
      event.preventDefault()
      searchRef.current?.focus()
    }
    window.addEventListener("keydown", focusSearch)
    return () => window.removeEventListener("keydown", focusSearch)
  }, [])

  return (
    <section
      className="absolute bottom-4 left-1/2 z-10 max-h-[calc(100%-2rem)] w-[min(38rem,calc(100%-2rem))] -translate-x-1/2 overflow-y-auto rounded-xl border border-white/15 bg-black/85 text-white shadow-2xl backdrop-blur-xl"
      aria-label="Card suggestions"
    >
      <header className="flex items-center justify-between px-3 pt-2">
        <span className="text-[0.65rem] font-bold tracking-wider text-white/60 uppercase">
          Identify card · {playerName}’s board
        </span>
        <button
          type="button"
          className="grid size-6 place-items-center rounded-full text-white/60 hover:bg-white/10 hover:text-white"
          onClick={onDismiss}
          aria-label="Dismiss suggestions"
        >
          <X className="size-3.5" />
        </button>
      </header>
      <div className="grid gap-3 p-3 sm:grid-cols-[7rem_1fr]">
        <figure className="relative aspect-square w-full overflow-hidden rounded-lg">
          <img
            className="size-full object-cover"
            src={capture.image}
            alt="Native camera crop around the clicked card"
          />
          {recognition.status === "done" && (
            <svg
              className="absolute inset-0 size-full"
              viewBox={`0 0 ${capture.cropSize} ${capture.cropSize}`}
              aria-label="Detected card outline"
            >
              <polygon
                points={recognition.result.quad.map(([x, y]) => `${x},${y}`).join(" ")}
                fill="none"
                stroke="oklch(85% 0.2 150)"
                strokeWidth={capture.cropSize / 120}
                strokeLinejoin="round"
              />
            </svg>
          )}
          {recognition.status === "identifying" && (
            <span
              className="loading loading-spinner loading-sm absolute right-1 bottom-1 text-white"
              aria-label="Identifying"
            />
          )}
        </figure>

        <div className="min-w-0">
          <div className="grid gap-1">
            {candidates.map((art, index) => (
              <div key={art.id} className="grid">
                <CardHover id={art.id} name={art.name}>
                  <button
                    type="button"
                    className={cn(
                      "flex h-8 w-full items-center gap-2 rounded-md border border-white/10 bg-white/5 px-2 text-left text-xs hover:bg-white/15",
                      index === 0 && clear && "border-success/60 bg-success/15",
                    )}
                    onClick={() => onChooseCard(art)}
                  >
                    <kbd className="kbd kbd-xs bg-white text-black">{index + 1}</kbd>
                    <span className="truncate font-semibold">{art.name}</span>
                    {inDeck(art.name, deckNames) && <InDeckBadge />}
                    <span className="truncate text-white/50">{galleryPrintingCaption(art)}</span>
                    <span className="ml-auto tabular-nums text-white/40">
                      {art.score.toFixed(2)}
                    </span>
                  </button>
                </CardHover>
                <PrintingChoices
                  art={art}
                  onChoose={onChooseCard}
                  onPrintings={onPrintings}
                  version={galleryVersion}
                />
              </div>
            ))}
            {recognition.status === "skipped" &&
              deckSuggestions.map((deck, index) => (
                <CommanderHover key={deck.id} deck={deck}>
                  <button
                    type="button"
                    className="flex h-8 w-full items-center gap-2 rounded-md border border-white/10 bg-white/5 px-2 text-left text-xs hover:bg-white/15"
                    onClick={() => onChooseDeck(deck.id)}
                  >
                    <kbd className="kbd kbd-xs bg-white text-black">{index + 1}</kbd>
                    <span className="truncate font-semibold">{deck.commander_name}</span>
                    <GameChangerBadge gameChanger={deck.commander_game_changer} compact />
                    <span className="ml-auto truncate text-white/50">{deck.name}</span>
                  </button>
                </CommanderHover>
              ))}
            {recognition.status === "skipped" && deckSuggestions.length === 0 && (
              <p className="text-xs text-white/65">No decks are recorded for {playerName} yet.</p>
            )}
          </div>

          {gallerySearchable && (
            <label className="mt-2 flex h-8 items-center gap-2 rounded-md border border-white/10 bg-white/5 px-2 text-xs focus-within:border-white/40">
              <Search className="size-3.5 text-white/50" />
              <input
                ref={searchRef}
                className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-white/40"
                placeholder="Search name, set:mh2, #236, lang:en  (/)"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                aria-label="Search the card gallery"
              />
            </label>
          )}
          {matches.length > 0 && (
            <ul
              className="mt-1 max-h-40 overflow-y-auto rounded-md border border-white/10 text-xs"
              aria-label="Search results"
            >
              {matches.map((art) => (
                <li key={art.id} className="grid">
                  <CardHover id={art.id} name={art.name}>
                    <button
                      type="button"
                      className="w-full px-2 py-1.5 text-left hover:bg-white/15"
                      onClick={() => onChooseCard(art)}
                    >
                      <span className="flex items-center gap-2">
                        <span className="truncate font-semibold">{art.name}</span>
                        {inDeck(art.name, deckNames) && <InDeckBadge />}
                      </span>
                      <span className="block text-white/50">{galleryPrintingCaption(art)}</span>
                    </button>
                  </CardHover>
                </li>
              ))}
            </ul>
          )}

          <p className="mt-2 text-[0.65rem] text-white/45">
            {capture.cropSize} px crop of {capture.nativeWidth}×{capture.nativeHeight}
            {recognition.status === "done" &&
              ` · detector ${recognition.result.timings.detector.toFixed(0)} ms · embed ${recognition.result.timings.embed.toFixed(0)} ms · search ${recognition.result.timings.search.toFixed(0)} ms · upright ${Math.round(recognition.result.upVote * 100)}%`}
            {recognition.status === "identifying" &&
              (recognition.loading ? " · loading card scanner…" : " · identifying…")}
            {recognition.status === "skipped" &&
              ` · recognition ${recognition.reason}; deck-based suggestions`}
          </p>
        </div>
      </div>
    </section>
  )
}
