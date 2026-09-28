import { afterEach, expect, it, vi } from "vite-plus/test"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { FullFrameIdentification, Identification } from "./recognition/messages"
import type { Candidate, Quad } from "./recognition/pipeline"
import { overlayCardsFromScan, stabilizeSuperAiCards } from "./super-ai"
import { applySuperAiCorrections, quadOverlap } from "./super-ai-corrections"
import { SuperAiActions } from "./super-ai-actions"
import { mapSourceQuad, quadTransform, SuperAiArt } from "./super-ai-overlay"

afterEach(cleanup)

const quad = [
  [100, 50],
  [300, 50],
  [300, 250],
  [100, 250],
] as const

function scannedCard(
  candidates: Candidate[],
  cardQuad: Quad = quad as unknown as Quad,
): Identification {
  return {
    quad: cardQuad,
    upVote: 0.9,
    candidates,
    timings: { detector: 0, embed: 0, search: 0, total: 0 },
  }
}

function candidate(id: string): Candidate {
  return { id, name: "Forest", set: "lea", frame: "1993", index: 0, score: 0.9 }
}

it("maps a source quad through object-contain letterboxing", () => {
  expect(
    mapSourceQuad(
      quad,
      { width: 400, height: 300 },
      { width: 800, height: 800 },
      { horizontal: false, vertical: false },
    ),
  ).toEqual([
    [200, 200],
    [600, 200],
    [600, 600],
    [200, 600],
  ])
})

it("maps both viewer flips after sizing", () => {
  expect(
    mapSourceQuad(
      quad,
      { width: 400, height: 300 },
      { width: 800, height: 800 },
      { horizontal: true, vertical: true },
    ),
  ).toEqual([
    [600, 600],
    [200, 600],
    [200, 200],
    [600, 200],
  ])
})

it("produces a projective transform and rejects degenerate quads", () => {
  expect(
    quadTransform([
      [0, 0],
      [100, 0],
      [90, 140],
      [10, 100],
    ]),
  ).toContain("matrix3d(")
  expect(
    quadTransform([
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ]),
  ).toBeNull()
})

it("keeps each scanned card's best match and drops cards with no candidates", () => {
  const scan: FullFrameIdentification = {
    cards: [
      scannedCard([candidate("forest"), { ...candidate("island"), score: 0.8 }]),
      scannedCard([]),
    ],
    detections: [],
    totalMs: 10,
  }
  expect(overlayCardsFromScan(scan)).toEqual([{ id: "forest", quad }])
})

it("does not replace a card with only one candidate", () => {
  const scan: FullFrameIdentification = {
    cards: [scannedCard([candidate("forest")])],
    detections: [],
    totalMs: 10,
  }
  expect(overlayCardsFromScan(scan)).toEqual([])
})

it("requires the top candidate to clearly lead and accepts a caller-supplied lead", () => {
  const scan: FullFrameIdentification = {
    cards: [scannedCard([candidate("forest"), { ...candidate("island"), score: 0.84 }])],
    detections: [],
    totalMs: 10,
  }
  expect(overlayCardsFromScan(scan)).toEqual([])
  expect(overlayCardsFromScan(scan, 0.05)).toEqual([{ id: "forest", quad }])
  expect(overlayCardsFromScan(scan, 0)).toEqual([{ id: "forest", quad }])
})

it("keeps a matching card in its prior enlarged box and updates a real move", () => {
  const previous = [{ id: "forest", quad: quad as unknown as Quad }]
  const slightAdjustment = [
    [110, 55],
    [310, 55],
    [310, 255],
    [110, 255],
  ] as unknown as Quad
  const moved = [
    [250, 55],
    [450, 55],
    [450, 255],
    [250, 255],
  ] as unknown as Quad

  expect(stabilizeSuperAiCards(previous, [{ id: "forest", quad: slightAdjustment }])).toEqual(
    previous,
  )
  expect(stabilizeSuperAiCards(previous, [{ id: "forest", quad: moved }])).toEqual([
    { id: "forest", quad: moved },
  ])
})

it("spatially carries hide and replacement corrections across scan jitter and changed answers", () => {
  const moved = quad.map(([x, y]) => [x + 12, y - 8]) as Quad
  expect(quadOverlap(quad as unknown as Quad, moved)).toBeGreaterThan(0.7)
  expect(
    applySuperAiCorrections(
      [{ id: "wrong-again", quad: moved }],
      [{ quad: quad as unknown as Quad, replacementId: "correct" }],
    ).cards,
  ).toEqual([{ id: "correct", quad: moved }])
  expect(
    applySuperAiCorrections(
      [{ id: "false-positive", quad: moved }],
      [{ quad: quad as unknown as Quad, hidden: true }],
    ).cards,
  ).toEqual([])
})

it("renders art as a non-interactive projectively transformed image", () => {
  const transform = "matrix3d(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)"
  render(<SuperAiArt src="https://cards.example/art.jpg" transform={transform} />)
  const art = screen.getByRole("presentation")
  expect(art.getAttribute("src")).toBe("https://cards.example/art.jpg")
  expect(art.style.transform).toBe(transform)
  expect(art.className).toContain("origin-top-left")
})

it("wires both overlay context actions to its detected region", async () => {
  const card = { id: "forest", quad: quad as unknown as Quad }
  const notCard = vi.fn()
  render(
    <SuperAiActions card={card} onWrongCard={vi.fn()} onNotCard={notCard}>
      <button type="button">Overlay art</button>
    </SuperAiActions>,
  )
  fireEvent.contextMenu(screen.getByRole("button", { name: "Overlay art" }))
  expect(await screen.findByText("Wrong card")).toBeTruthy()
  fireEvent.click(await screen.findByText("Not a card"))
  expect(notCard).toHaveBeenCalledWith(card)
})
