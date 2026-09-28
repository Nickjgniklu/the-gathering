import { Eye, EyeOff, ListChecks, Search, XCircle } from "lucide-react"
import {
  cloneElement,
  useState,
  type CSSProperties,
  type MouseEvent,
  type ReactElement,
} from "react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import type { SuperAiOverlayCard } from "./super-ai"

export function SuperAiActions({
  card,
  children,
  onWrongCard,
  onNotCard,
  onChooseCard,
  onPreview,
  controlsStyle,
}: {
  card: SuperAiOverlayCard
  children: ReactElement
  onWrongCard: (card: SuperAiOverlayCard) => void
  onNotCard: (card: SuperAiOverlayCard) => void
  onChooseCard?: (card: SuperAiOverlayCard) => void
  onPreview?: (card: SuperAiOverlayCard) => void
  controlsStyle?: CSSProperties
}) {
  const [open, setOpen] = useState(false)
  const [artHidden, setArtHidden] = useState(false)
  const trigger = cloneElement(children, {
    style: artHidden ? { ...children.props.style, visibility: "hidden" } : children.props.style,
  })
  return (
    <span className="relative block">
      <DropdownMenu open={open} onOpenChange={setOpen}>
        <DropdownMenuTrigger asChild>
          <span
            onContextMenuCapture={(event: MouseEvent) => {
              event.preventDefault()
              setOpen(true)
            }}
            onClick={(event: MouseEvent) => {
              if (!event.defaultPrevented) onPreview?.(card)
            }}
          >
            {trigger}
          </span>
        </DropdownMenuTrigger>
        <DropdownMenuContent aria-label="Super AI card actions">
          <DropdownMenuItem onSelect={() => onWrongCard(card)}>
            <Search className="size-4" /> Wrong card
          </DropdownMenuItem>
          <DropdownMenuItem destructive onSelect={() => onNotCard(card)}>
            <XCircle className="size-4" /> Not a card
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <span className="pointer-events-auto absolute z-20 flex gap-1" style={controlsStyle}>
        {card.ambiguous && onChooseCard && (
          <button
            type="button"
            className="btn btn-circle btn-xs"
            onClick={() => onChooseCard(card)}
            aria-label="Choose close match"
          >
            <ListChecks className="size-3.5" />
          </button>
        )}
        <button
          type="button"
          className="btn btn-circle btn-xs"
          onClick={() => setArtHidden(!artHidden)}
          aria-label={artHidden ? "Restore card art" : "Hide card art"}
        >
          {artHidden ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
        </button>
      </span>
    </span>
  )
}
