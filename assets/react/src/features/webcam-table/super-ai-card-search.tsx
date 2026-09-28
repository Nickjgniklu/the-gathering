import { useEffect, useState } from "react"
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import type { GalleryArt } from "./recognition/pipeline"

export function SuperAiCardSearch({
  open,
  onOpenChange,
  onSearch,
  onChoose,
  candidates = [],
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSearch: (query: string) => Promise<GalleryArt[]>
  onChoose: (art: GalleryArt) => void
  candidates?: GalleryArt[]
}) {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<GalleryArt[]>([])
  useEffect(() => {
    if (!open || !query.trim()) return setResults([])
    let current = true
    const timer = window.setTimeout(() => {
      void onSearch(query).then((matches) => current && setResults(matches))
    }, 150)
    return () => {
      current = false
      window.clearTimeout(timer)
    }
  }, [open, onSearch, query])
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Choose the correct card</DialogTitle>
          <DialogClose onClose={() => onOpenChange(false)} />
        </DialogHeader>
        <div className="grid gap-3 p-5">
          {candidates.length > 0 && (
            <ul className="max-h-48 overflow-y-auto" aria-label="Close matches">
              {candidates.map((art) => (
                <li key={art.id}>
                  <button
                    type="button"
                    className="w-full rounded-field px-3 py-2 text-left hover:bg-base-200"
                    onClick={() => onChoose(art)}
                  >
                    <span className="font-semibold">{art.name}</span>
                    <span className="ml-2 text-sm opacity-60">{art.set.toUpperCase()}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          <Input
            autoFocus
            aria-label="Search for the correct card"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Card name, set, or collector number"
          />
          <ul className="max-h-80 overflow-y-auto" aria-label="Card search results">
            {results.map((art) => (
              <li key={art.id}>
                <button
                  type="button"
                  className="w-full rounded-field px-3 py-2 text-left hover:bg-base-200"
                  onClick={() => onChoose(art)}
                >
                  <span className="font-semibold">{art.name}</span>
                  <span className="ml-2 text-sm opacity-60">
                    {art.set.toUpperCase()} · #{art.collector_number}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      </DialogContent>
    </Dialog>
  )
}
