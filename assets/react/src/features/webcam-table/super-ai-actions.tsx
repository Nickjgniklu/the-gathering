import { Search, XCircle } from "lucide-react"
import { cloneElement, useState, type MouseEvent, type ReactElement } from "react"
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
}: {
  card: SuperAiOverlayCard
  children: ReactElement
  onWrongCard: (card: SuperAiOverlayCard) => void
  onNotCard: (card: SuperAiOverlayCard) => void
}) {
  const [open, setOpen] = useState(false)
  const trigger = cloneElement(children, {
    onContextMenuCapture: (event: MouseEvent) => {
      event.preventDefault()
      setOpen(true)
    },
  })
  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>{trigger}</DropdownMenuTrigger>
      <DropdownMenuContent aria-label="Super AI card actions">
        <DropdownMenuItem onSelect={() => onWrongCard(card)}>
          <Search className="size-4" /> Wrong card
        </DropdownMenuItem>
        <DropdownMenuItem destructive onSelect={() => onNotCard(card)}>
          <XCircle className="size-4" /> Not a card
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
