import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vite-plus/test"
import { BoardCardTray } from "./board-cards"
import type { SuperAiOverlayCard } from "./super-ai"
import type { BoardCard, TableParticipant } from "./use-webcam-room"

afterEach(cleanup)

const alice = { peer_id: "alice", player_id: 1, player_name: "Alice" } as TableParticipant

function entry(id: string, ownerPeerId: string, name: string): BoardCard {
  return { id, ownerPeerId, at: 1, byPlayerName: "Alice", card: { id, name, set: "lea" } }
}

function tray(onClear?: () => void, superAiCards?: SuperAiOverlayCard[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const cards = [entry("bolt", "alice", "Lightning Bolt"), entry("snap", "bob", "Counterspell")]
  render(
    <QueryClientProvider client={client}>
      <BoardCardTray
        participant={alice}
        cards={cards}
        superAiCards={superAiCards}
        onPreview={vi.fn()}
        onRemove={vi.fn()}
        onClear={onClear}
      />
    </QueryClientProvider>,
  )
  fireEvent.click(screen.getByRole("button", { expanded: false }))
}

it("offers Clear cards only to the board's owner and never a per-card Rulings button", () => {
  const onClear = vi.fn()
  tray(onClear)
  expect(screen.getByRole("button", { name: "Show Lightning Bolt" })).toBeTruthy()
  expect(screen.queryByRole("button", { name: "Show Counterspell" })).toBeNull()
  expect(screen.queryByRole("button", { name: /Rulings/ })).toBeNull()
  fireEvent.click(screen.getByRole("button", { name: "Clear cards" }))
  expect(onClear).toHaveBeenCalledOnce()

  cleanup()
  tray(undefined)
  expect(screen.queryByRole("button", { name: "Clear cards" })).toBeNull()
})

it("switches between click history and the current Super AI scan without adding scan results to history", () => {
  tray(undefined, [
    {
      id: "forest",
      quad: [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ],
    },
  ])
  expect(screen.getByRole("tab", { name: "Click history" }).getAttribute("aria-selected")).toBe(
    "true",
  )
  fireEvent.click(screen.getByRole("tab", { name: "Super AI now" }))
  expect(screen.getByRole("tab", { name: "Super AI now" }).getAttribute("aria-selected")).toBe(
    "true",
  )
  expect(screen.getByRole("list", { name: "Current Super AI cards" })).toBeTruthy()
  expect(screen.queryByRole("button", { name: "Clear cards" })).toBeNull()
})
