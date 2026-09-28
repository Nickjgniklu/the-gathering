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

function tray(
  onClear?: () => void,
  superAiCards?: SuperAiOverlayCard[],
  actions?: {
    wrong: (card: SuperAiOverlayCard) => void
    notCard: (card: SuperAiOverlayCard) => void
  },
) {
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
        onWrongSuperAiCard={actions?.wrong}
        onNotSuperAiCard={actions?.notCard}
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

it("wires both Super AI tray context actions to the selected region", async () => {
  const notCard = vi.fn()
  const card = {
    id: "forest",
    quad: [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ],
  } as SuperAiOverlayCard
  tray(undefined, [card], { wrong: vi.fn(), notCard })
  fireEvent.click(screen.getByRole("tab", { name: "Auto-identified" }))
  fireEvent.contextMenu(screen.getByRole("button", { name: "Auto-identified card actions" }))
  expect(await screen.findByText("Wrong card")).toBeTruthy()
  fireEvent.click(await screen.findByText("Not a card"))
  expect(notCard).toHaveBeenCalledWith(card)
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
  fireEvent.click(screen.getByRole("tab", { name: "Auto-identified" }))
  expect(screen.getByRole("tab", { name: "Auto-identified" }).getAttribute("aria-selected")).toBe(
    "true",
  )
  expect(screen.getByRole("list", { name: "Current auto-identified cards" })).toBeTruthy()
  expect(screen.queryByRole("button", { name: "Clear cards" })).toBeNull()
})
