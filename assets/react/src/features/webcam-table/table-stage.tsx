import { Link } from "@tanstack/react-router"
import { LayoutGrid, Sparkles, Undo2 } from "lucide-react"
import type { Ref } from "react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/cn"
import { ActiveBoard, capturePoint } from "./board"
import { BoardCardTray } from "./board-cards"
import { cameraGridLayout } from "./camera-grid"
import { CardPreview } from "./card-preview"
import { CardSuggestions } from "./card-suggestions"
import { SuperAiOverlay } from "./super-ai-overlay"
import type { SuperAiDetection, SuperAiOverlayCard } from "./super-ai"
import type { TableParticipant } from "./room-types"
import { describeRoll } from "./table-rolls"
import { SeatActions, SeatLife, SeatTile, TeamHeader } from "./table-seat"
import {
  isCurrentTurn,
  videoFlip,
  isLocal,
  isPinned,
  revealLabels,
  streamFor,
  type TableView,
} from "./table-view"
import type { CardIdentificationFlow } from "./use-card-identification-flow"
import type { IdentifiedCard } from "./use-webcam-room"
import type { useVideoStats } from "./video-stats"

function StageBoard({
  view,
  participant,
  flow,
  superAiCards,
  superAiEnabled,
  showSuperAiDetections,
  onToggleSuperAi,
  onWrongSuperAiCard,
  onNotSuperAiCard,
  onPreviewSuperAiCard,
  onResetSuperAiCards,
}: {
  view: TableView
  participant: TableParticipant
  flow: CardIdentificationFlow
  superAiCards: {
    cards: SuperAiOverlayCard[]
    detections: SuperAiDetection[]
    source: { width: number; height: number } | null
  }
  superAiEnabled: boolean
  showSuperAiDetections: boolean
  onToggleSuperAi: () => void
  onWrongSuperAiCard: (card: SuperAiOverlayCard) => void
  onNotSuperAiCard: (card: SuperAiOverlayCard) => void
  onPreviewSuperAiCard: (card: SuperAiOverlayCard | null) => void
  onResetSuperAiCards: () => void
}) {
  const { room, preferences } = view
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="relative min-h-0 flex-1" data-stage-board>
        <ActiveBoard
          participant={participant}
          unattackable={view.protectedSeats.includes(participant.peer_id)}
          monarch={room.monarch?.peer_id === participant.peer_id}
          {...revealLabels(view, participant)}
          local={isLocal(view, participant)}
          flip={videoFlip(view, participant)}
          currentTurn={isCurrentTurn(view, participant)}
          connectionState={room.connectionStates[participant.peer_id]}
          stream={streamFor(view, participant)}
          lifeControl={<SeatLife view={view} participant={participant} size="board" />}
          release={
            !isPinned(view, participant)
              ? undefined
              : view.preferences.viewMode === "grid"
                ? {
                    label: "Back to grid",
                    title: "Show every camera again (or click this player's camera)",
                    icon: LayoutGrid,
                    onClick: view.releaseBoard,
                  }
                : {
                    label: "Follow turn",
                    title: "Go back to following the active turn (or click this player's camera)",
                    icon: Undo2,
                    onClick: view.releaseBoard,
                  }
          }
          onInspect={(event) => {
            const flip = videoFlip(view, participant)
            const point = capturePoint(event, flip)
            if (point)
              room.requestCapture(participant.peer_id, point.x, point.y, event.shiftKey, flip)
          }}
        />
        <SuperAiOverlay
          cards={
            !showSuperAiDetections && participant.peer_id === view.activeGroup[0]?.peer_id
              ? superAiCards.cards
              : []
          }
          detections={
            showSuperAiDetections && participant.peer_id === view.activeGroup[0]?.peer_id
              ? superAiCards.detections
              : []
          }
          source={superAiCards.source}
          flip={videoFlip(view, participant)}
          onWrongCard={onWrongSuperAiCard}
          onNotCard={onNotSuperAiCard}
          onChooseCard={onWrongSuperAiCard}
          onPreviewCard={onPreviewSuperAiCard}
        />
        {participant.peer_id === view.activeGroup[0]?.peer_id && (
          <Button
            type="button"
            variant={superAiEnabled ? "default" : "secondary"}
            size="sm"
            className={cn(
              "absolute right-3 z-10 shadow-lg",
              isPinned(view, participant) ? "top-12" : "top-3",
            )}
            aria-pressed={superAiEnabled}
            aria-label={`${superAiEnabled ? "Turn off" : "Turn on"} Super AI`}
            onClick={onToggleSuperAi}
          >
            <Sparkles className="size-4" aria-hidden="true" />
            Super AI {superAiEnabled ? "on" : "off"}
          </Button>
        )}
        <BoardCardTray
          participant={participant}
          cards={room.identifiedCards}
          superAiCards={
            superAiEnabled && participant.peer_id === view.activeGroup[0]?.peer_id
              ? superAiCards.cards
              : undefined
          }
          onPreview={flow.previewEntry}
          onRemove={room.removeCard}
          onClear={
            participant.peer_id === view.localParticipant.peer_id ? room.clearOwnCards : undefined
          }
          defaultExpanded={preferences.keepTrayOpen && preferences.trayOpen}
          onExpandedChange={(trayOpen) => preferences.update({ trayOpen })}
          onWrongSuperAiCard={onWrongSuperAiCard}
          onNotSuperAiCard={onNotSuperAiCard}
          onPreviewSuperAiCard={onPreviewSuperAiCard}
          onResetSuperAiCards={onResetSuperAiCards}
        />
      </div>
      <SeatActions view={view} participant={participant} size="board" />
    </div>
  )
}

/** Grid view: every seat's camera shares the stage (teams stay together in Two-Headed Giant).
 * Clicking one fills the stage with that board until it is clicked again. */
function CameraGrid({
  view,
  videoStats,
}: {
  view: TableView
  videoStats: ReturnType<typeof useVideoStats>
}) {
  const teamsMode = view.room.mode === "two_headed_giant"
  const { columns, rows, cells } = cameraGridLayout(view.groups.length)
  return (
    <div
      className="grid min-h-0 flex-1 gap-1.5 p-1.5"
      style={{
        gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
        gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
      }}
    >
      {view.groups.map((group, index) => (
        <div
          key={group[0]!.peer_id}
          className={cn(
            "flex min-h-0 min-w-0 flex-col overflow-hidden rounded-sm",
            teamsMode && "rounded-lg border border-primary/40",
          )}
          style={{ gridRow: cells[index]!.row, gridColumn: cells[index]!.column }}
        >
          {teamsMode && <TeamHeader view={view} group={group} />}
          {group.map((participant) => (
            <SeatTile
              key={participant.peer_id}
              view={view}
              participant={participant}
              videoStats={videoStats}
              fill
            />
          ))}
        </div>
      ))}
    </div>
  )
}

/** Card identification on top of the active board: progress, the picker, and the preview. */
function IdentificationOverlays({ view, flow }: { view: TableView; flow: CardIdentificationFlow }) {
  const { capture, captureOwner, recognition, recognizer, preview } = flow
  return (
    <>
      {capture && recognition.status === "identifying" && !flow.pickerOpen && !preview && (
        <p
          role="status"
          className="absolute bottom-12 left-1/2 z-10 -translate-x-1/2 rounded-lg bg-base-100/95 px-4 py-2 text-sm text-base-content shadow-lg"
        >
          {recognition.loading ? "Loading card scanner…" : "Identifying card…"}
        </p>
      )}
      {capture && captureOwner && flow.pickerOpen && (
        <CardSuggestions
          capture={capture}
          playerName={captureOwner.player_name}
          recognition={recognition}
          deckSuggestions={flow.suggestions}
          gallerySearchable
          onChooseCard={flow.chooseCard}
          onChooseDeck={flow.chooseDeckForCapture}
          onSearch={flow.search}
          onPrintings={recognizer.printings}
          galleryVersion={"version" in recognizer.state ? recognizer.state.version : undefined}
          deckNames={flow.ownerDeckNames}
          onDismiss={flow.dismissPicker}
        />
      )}
      {preview?.kind === "entry" && (
        <CardPreview
          card={preview.shown}
          ownerName={
            view.seated.find((seat) => seat.peer_id === preview.entry.ownerPeerId)?.player_name
          }
          onWrongCard={
            preview.correctable && capture ? () => flow.correctPreview(preview.entry.id) : undefined
          }
          onRemove={() => {
            view.room.removeCard(preview.entry.id)
            flow.closePreview()
          }}
          onClose={flow.closePreview}
        />
      )}
      {preview?.kind === "art" && <CardPreview card={preview.card} onClose={flow.closePreview} />}
    </>
  )
}

/** The middle of the table: the active board (or team), or every camera in grid view. */
export function TableStage({
  ref,
  view,
  flow,
  videoStats,
  superAiCards,
  superAiEnabled,
  showSuperAiDetections,
  onToggleSuperAi,
  onWrongSuperAiCard,
  onNotSuperAiCard,
  superAiPreview,
  onPreviewSuperAiCard,
  onResetSuperAiCards,
}: {
  ref?: Ref<HTMLElement>
  view: TableView
  flow: CardIdentificationFlow
  videoStats: ReturnType<typeof useVideoStats>
  superAiCards: {
    cards: SuperAiOverlayCard[]
    detections: SuperAiDetection[]
    source: { width: number; height: number } | null
  }
  superAiEnabled: boolean
  showSuperAiDetections: boolean
  onToggleSuperAi: () => void
  onWrongSuperAiCard: (card: SuperAiOverlayCard) => void
  onNotSuperAiCard: (card: SuperAiOverlayCard) => void
  superAiPreview: SuperAiOverlayCard | null
  onPreviewSuperAiCard: (card: SuperAiOverlayCard | null) => void
  onResetSuperAiCards: () => void
}) {
  const { room } = view
  return (
    <section
      ref={ref}
      className={cn(
        "relative flex min-h-0 min-w-0 flex-col",
        view.preferences.panelLeft && "lg:order-3",
      )}
      aria-label={view.showGrid ? "Camera grid" : "Active board"}
    >
      {room.spectating && (
        <p role="status" className="bg-base-200 px-4 py-2 text-sm font-semibold text-base-content">
          Spectating — this game has already started. Your camera is not shared.
          <Link to="/games" className="link ml-3">
            Leave table
          </Link>
        </p>
      )}
      {room.roll && (
        <div
          role="status"
          className="pointer-events-none absolute top-20 left-1/2 z-30 w-max max-w-[90%] -translate-x-1/2 rounded-xl border border-accent/40 bg-base-100/95 px-6 py-4 text-center text-lg font-semibold text-base-content shadow-xl"
        >
          {describeRoll(room.roll)}
        </div>
      )}
      {room.mode === "two_headed_giant" && !view.showGrid && (
        <TeamHeader view={view} group={view.activeGroup} />
      )}
      <div className="relative flex min-h-0 flex-1 flex-col">
        {view.showGrid ? (
          <CameraGrid view={view} videoStats={videoStats} />
        ) : (
          view.activeGroup.map((participant) => (
            <StageBoard
              key={participant.peer_id}
              view={view}
              participant={participant}
              flow={flow}
              superAiCards={superAiCards}
              superAiEnabled={superAiEnabled}
              showSuperAiDetections={showSuperAiDetections}
              onToggleSuperAi={onToggleSuperAi}
              onWrongSuperAiCard={onWrongSuperAiCard}
              onNotSuperAiCard={onNotSuperAiCard}
              onPreviewSuperAiCard={onPreviewSuperAiCard}
              onResetSuperAiCards={onResetSuperAiCards}
            />
          ))
        )}
        <IdentificationOverlays view={view} flow={flow} />
        {superAiPreview && (
          <CardPreview
            card={superAiPreviewCard(superAiPreview)}
            onWrongCard={() => {
              onWrongSuperAiCard(superAiPreview)
              onPreviewSuperAiCard(null)
            }}
            onRemove={() => {
              onNotSuperAiCard(superAiPreview)
              onPreviewSuperAiCard(null)
            }}
            onClose={() => onPreviewSuperAiCard(null)}
          />
        )}
      </div>
    </section>
  )
}

function superAiPreviewCard(card: SuperAiOverlayCard): IdentifiedCard {
  const candidate = card.candidates?.find((entry) => entry.id === card.id)
  return { id: card.id, name: candidate?.name ?? "Recognized card", set: candidate?.set ?? "" }
}
