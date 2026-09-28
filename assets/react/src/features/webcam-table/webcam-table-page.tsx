import { useQuery } from "@tanstack/react-query"
import { Link, useNavigate } from "@tanstack/react-router"
import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react"
import { useToast } from "@/components/ui/toast"
import { getDecks, type DeckSummary } from "@/features/decks/decks"
import { getPlayers } from "@/features/games/games"
import { useCurrentUser } from "@/lib/auth"
import { cn } from "@/lib/cn"
import { turnId } from "./game-modes"
import { MAX_PLAYERS } from "./rooms"
import { SidePanel, type PanelTab } from "./side-panel"
import { TableCameraRail } from "./table-camera-rail"
import { TableDialogs, type TableDialog } from "./table-dialogs"
import { RailResizeHandle, useTablePreferences } from "./table-preferences"
import { TableSettings } from "./table-settings"
import { TableStage } from "./table-stage"
import { decksFor, useTableView } from "./table-view"
import { useCameraRailWidth } from "./use-camera-rail-width"
import { useCardIdentificationFlow } from "./use-card-identification-flow"
import { useCorrectionUpload } from "./use-correction-upload"
import { SuperAiCardSearch } from "./super-ai-card-search"
import { useSuperAiCorrections } from "./super-ai-corrections"
import { decodeJpeg } from "./recognition/use-recognizer"
import { useRoomHotkeys } from "./use-room-hotkeys"
import { useSeatDecklists } from "./seat-decklists"
import { useTurnSound } from "./use-turn-sound"
import { useVideoStats } from "./video-stats"
import { useWebcamRoom } from "./use-webcam-room"
import {
  overlayCardsFromScan,
  stabilizeSuperAiCards,
  type SuperAiDetection,
  type SuperAiOverlayCard,
} from "./super-ai"

interface Props {
  roomId: string
}

interface LiveRoomProps extends Props {
  playerId: number
  playerName: string
  decks: DeckSummary[]
}

function useInviteLink() {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), 2500)
    return () => window.clearTimeout(timer)
  }, [copied])
  return {
    copied,
    copy: () => {
      void navigator.clipboard.writeText(window.location.href)
      setCopied(true)
    },
  }
}

/** Sends a seat back to the games list after the room owner ends the table. */
function TableEndedRedirect() {
  const navigate = useNavigate()
  const { toast } = useToast()
  // Strict Mode runs effects twice; announce once.
  const announced = useRef(false)
  useEffect(() => {
    if (!announced.current) toast({ message: "The room owner ended the game.", tone: "info" })
    announced.current = true
    void navigate({ to: "/games" })
  }, [navigate, toast])
  return null
}

/** The table layout: camera rail, active board, and side panel, with its dialogs. */
function LiveRoom({ roomId, playerId, playerName, decks }: LiveRoomProps) {
  const preferences = useTablePreferences(playerId)
  const superAiFrameHandler = useRef<
    (frame: { peerId: string; bytes: Uint8Array; width: number; height: number }) => Promise<void>
  >(async () => {})
  const superAiAbortRef = useRef<AbortController | null>(null)
  const superAiTargetRef = useRef<string | null>(null)
  const [superAiCards, setSuperAiCards] = useState<{
    cards: SuperAiOverlayCard[]
    detections: SuperAiDetection[]
    source: { width: number; height: number } | null
  }>({ cards: [], detections: [], source: null })
  const [superAiDebug, setSuperAiDebug] = useState<{
    transferMs: number; decodeMs: number; detectorMs: number; classifyMs: number; totalMs: number
  } | null>(null)
  const room = useWebcamRoom(
    roomId,
    playerId,
    null,
    preferences.deviceId,
    preferences.quality,
    preferences.cameraEnabled,
    useCallback((frame) => superAiFrameHandler.current(frame), []),
    preferences.superAiScanIntervalSeconds * 1_000,
  )
  const [dialog, setDialog] = useState<TableDialog>(null)
  const [panelOpen, setPanelOpen] = useState(true)
  const [panelTab, setPanelTab] = useState<PanelTab>("table")
  const invite = useInviteLink()
  const toggleCamera = () => {
    preferences.update({ cameraEnabled: room.cameraOff })
    room.toggleCamera()
  }
  const openHelp = useCallback(() => setDialog({ kind: "help" }), [])
  const view = useTableView({
    room,
    preferences,
    playerId,
    playerName,
    decks,
    toggleCamera,
    openReveal: () => setDialog({ kind: "reveal" }),
  })
  const videoStats = useVideoStats(preferences.stats, room.getPeerStats)
  useTurnSound(
    preferences.turnSound,
    room.turns.active_player_id,
    turnId(room.participants, playerId, room.mode) ?? playerId,
  )
  const corrections = useCorrectionUpload()
  const decklists = useSeatDecklists(view.seated, decks)
  const flow = useCardIdentificationFlow({
    room,
    seated: view.seated,
    decks,
    decklists,
    playerName,
    corrections,
    blocked: dialog?.kind === "help" || dialog?.kind === "finish",
  })
  const superAiTarget = view.activeGroup[0]?.peer_id
  const superAiCorrections = useSuperAiCorrections(roomId, room.peerId, superAiTarget)
  const [correctingSuperAiCard, setCorrectingSuperAiCard] = useState<SuperAiOverlayCard | null>(
    null,
  )
  const [previewingSuperAiCard, setPreviewingSuperAiCard] = useState<SuperAiOverlayCard | null>(null)
  const correctedSuperAiCards = {
    ...superAiCards,
    cards: superAiCorrections.apply(superAiCards.cards),
  }
  useEffect(() => {
    superAiFrameHandler.current = async (frame) => {
      const controller = superAiAbortRef.current
      if (!controller || controller.signal.aborted || superAiTargetRef.current !== frame.peerId)
        return
      const decodeStarted = performance.now()
      const image = await decodeJpeg(new Blob([frame.bytes.buffer], { type: "image/jpeg" }))
      const decodeMs = performance.now() - decodeStarted
      const result = await flow.recognizer.identifyFrame(
        image,
        { minMatchConfidence: preferences.superAiMinMatchConfidence },
        AbortSignal.any([controller.signal, AbortSignal.timeout(30_000)]),
      )
      if (controller.signal.aborted || superAiTargetRef.current !== frame.peerId) return
      const scanned = overlayCardsFromScan(result, preferences.superAiMargin)
      setSuperAiDebug({ transferMs: frame.transferMs, decodeMs, detectorMs: result.timings.detector, classifyMs: result.timings.classify, totalMs: result.totalMs })
      setSuperAiCards((previous) => ({
        cards:
          previous.source?.width === frame.width && previous.source.height === frame.height
            ? stabilizeSuperAiCards(previous.cards, scanned)
            : scanned,
        detections: result.detections,
        source: { width: frame.width, height: frame.height },
      }))
    }
    return () => {
      superAiFrameHandler.current = async () => {}
    }
  }, [
    flow.recognizer.identifyFrame,
    preferences.superAiMargin,
    preferences.superAiMinMatchConfidence,
  ])
  const { request: requestSuperAi, cancel: cancelSuperAi } = room.superAi
  const requestSuperAiRef = useRef(requestSuperAi)
  const cancelSuperAiRef = useRef(cancelSuperAi)
  requestSuperAiRef.current = requestSuperAi
  cancelSuperAiRef.current = cancelSuperAi
  useEffect(() => {
    superAiAbortRef.current?.abort()
    setSuperAiCards({ cards: [], detections: [], source: null })
    if (!preferences.superAi || !superAiTarget) {
      superAiAbortRef.current = null
      superAiTargetRef.current = null
      cancelSuperAiRef.current()
      return
    }
    const controller = new AbortController()
    superAiAbortRef.current = controller
    superAiTargetRef.current = superAiTarget
    let cancelled = false
    let timer: number | undefined
    const scan = () => {
      if (cancelled) return
      requestSuperAiRef.current(superAiTarget)
      timer = window.setTimeout(scan, preferences.superAiScanIntervalSeconds * 1_000)
    }
    scan()
    return () => {
      cancelled = true
      controller.abort()
      if (superAiAbortRef.current === controller) superAiAbortRef.current = null
      if (superAiTargetRef.current === superAiTarget) superAiTargetRef.current = null
      if (timer) window.clearTimeout(timer)
      cancelSuperAiRef.current()
    }
  }, [preferences.superAi, preferences.superAiScanIntervalSeconds, superAiTarget])
  useRoomHotkeys(view, flow, {
    togglePanel: () => setPanelOpen((open) => !open),
    showTab: (tab) => {
      setPanelTab(tab)
      setPanelOpen(true)
    },
    openHelp,
  })

  // The grid already shows every camera, so the rail and its divider step aside.
  const railShown = !view.showGrid
  // The rail may grow into the active board's pillarbox bars, but never shrink its camera.
  const stageRef = useRef<HTMLElement>(null)
  const cameraRail = useCameraRailWidth(
    stageRef,
    preferences.camera,
    `${railShown}:${view.activeGroup.map((participant) => participant.peer_id).join(",")}`,
  )
  return (
    <div
      className={cn(
        "grid h-dvh bg-black text-white lg:grid-rows-1",
        railShown ? "grid-rows-[auto_minmax(0,1fr)_auto]" : "grid-rows-[minmax(0,1fr)_auto]",
        railShown
          ? preferences.panelLeft
            ? "lg:grid-cols-[auto_auto_minmax(0,1fr)_0.375rem_var(--table-camera-width)]"
            : "lg:grid-cols-[var(--table-camera-width)_0.375rem_minmax(0,1fr)_auto_auto]"
          : preferences.panelLeft
            ? "lg:grid-cols-[auto_auto_minmax(0,1fr)]"
            : "lg:grid-cols-[minmax(0,1fr)_auto_auto]",
      )}
      style={
        {
          "--table-camera-width": `${cameraRail.width}px`,
          "--table-panel-width": `min(${preferences.panel}px, 32vw)`,
        } as CSSProperties
      }
    >
      {railShown && (
        <>
          <TableCameraRail view={view} videoStats={videoStats} />
          <RailResizeHandle
            rail="camera"
            reversed={preferences.panelLeft}
            width={cameraRail.width}
            max={cameraRail.max}
            onChange={(width) => preferences.setWidth("camera", width)}
          />
        </>
      )}

      <TableStage
        ref={stageRef}
        view={view}
        flow={flow}
        videoStats={videoStats}
        superAiCards={correctedSuperAiCards}
        superAiEnabled={preferences.superAi}
        showSuperAiDetections={preferences.superAiShowDetections}
        onToggleSuperAi={() => preferences.update({ superAi: !preferences.superAi })}
        onWrongSuperAiCard={setCorrectingSuperAiCard}
        onNotSuperAiCard={superAiCorrections.hide}
        superAiPreview={previewingSuperAiCard}
        onPreviewSuperAiCard={setPreviewingSuperAiCard}
        onResetSuperAiCards={() => {
          superAiCorrections.reset()
          setCorrectingSuperAiCard(null)
          setPreviewingSuperAiCard(null)
          setSuperAiCards({ cards: [], detections: [], source: null })
          setSuperAiDebug(null)
        }}
      />

      <SuperAiCardSearch
        open={correctingSuperAiCard !== null}
        onOpenChange={(open) => !open && setCorrectingSuperAiCard(null)}
        onSearch={flow.recognizer.search}
        candidates={correctingSuperAiCard?.candidates ?? []}
        onChoose={(art) => {
          if (correctingSuperAiCard) superAiCorrections.replace(correctingSuperAiCard, art)
          setCorrectingSuperAiCard(null)
        }}
      />

      {panelOpen ? (
        <RailResizeHandle
          rail="panel"
          reversed={preferences.panelLeft}
          width={preferences.panel}
          onChange={(width) => preferences.setWidth("panel", width)}
        />
      ) : (
        <div className={cn("hidden lg:block", preferences.panelLeft && "lg:order-2")} />
      )}

      <SidePanel
        mode={room.mode}
        onModeChange={room.setMode}
        onMoveSeat={room.moveSeat}
        spectating={room.spectating}
        isOwner={room.isOwner}
        left={preferences.panelLeft}
        onHelp={openHelp}
        settings={
          <TableSettings
            preferences={preferences}
            room={{ ...room, toggleCamera }}
            corrections={corrections}
            recognizer={flow.recognizer.state}
            superAiDebug={superAiDebug}
            onHelp={openHelp}
          />
        }
        open={panelOpen}
        tab={panelTab}
        onOpenChange={setPanelOpen}
        onTabChange={setPanelTab}
        participants={view.seated}
        spectators={room.spectators}
        localParticipant={view.localParticipant}
        maxPlayers={MAX_PLAYERS}
        playerDecks={decksFor(view, view.localParticipant)}
        decks={decks}
        events={room.events}
        status={room.status}
        error={room.error}
        connectedPeers={Object.keys(room.streams).length}
        connectionStates={room.connectionStates}
        iceServers={room.iceServers}
        recognizer={flow.recognizer.state}
        identifiedCards={room.identifiedCards}
        gallerySearchable
        onSearch={flow.recognizer.search}
        onPreviewCard={flow.previewEntry}
        onPreviewArt={flow.previewArt}
        onRemoveCard={room.removeCard}
        onClearOwnCards={room.clearOwnCards}
        inviteCopied={invite.copied}
        onInvite={invite.copy}
        onChooseDeck={room.chooseDeck}
        onStartGame={room.startGame}
        shuffleVersion={room.shuffleVersion}
        turns={room.turns}
        timer={room.timer}
        onPassTurn={room.passTurn}
        onUnpassTurn={room.unpassTurn}
        onAdjustTurn={room.adjustTurn}
        onRoll={room.rollDice}
        onChangeTimer={(action) => {
          void room.changeTimer(action)
        }}
        reveal={{ target: room.revealTo, busy: room.revealBusy, onChange: room.changeReveal }}
        onEndGame={() => {
          void room.changeTimer("pause").then((timer) => {
            if (timer) setDialog({ kind: "finish", timer })
          })
        }}
      />

      <TableDialogs view={view} dialog={dialog} onDialogChange={setDialog} />
      {room.closedByOwner && <TableEndedRedirect />}
    </div>
  )
}

export function WebcamTablePage({ roomId }: Props) {
  const session = useCurrentUser()
  const playersQuery = useQuery({ queryKey: ["players"], queryFn: getPlayers })
  const decksQuery = useQuery({ queryKey: ["decks", {}], queryFn: () => getDecks() })
  const player = playersQuery.data?.find((candidate) => candidate.user_id === session.data?.id)

  if (playersQuery.isPending || decksQuery.isPending) {
    return (
      <div className="grid min-h-dvh place-items-center bg-black">
        <span className="loading loading-spinner loading-lg" aria-label="Loading table" />
      </div>
    )
  }

  if (!player) {
    return (
      <div className="grid min-h-dvh place-items-center bg-black p-6">
        <div role="alert" className="alert alert-warning max-w-xl">
          <div>
            <h1 className="font-bold">No linked player</h1>
            <p>Your account must be linked to a player before joining a table.</p>
          </div>
          <Link to="/games" className="btn btn-sm">
            Back to games
          </Link>
        </div>
      </div>
    )
  }

  return (
    <LiveRoom
      roomId={roomId}
      playerId={player.id}
      playerName={player.name}
      decks={decksQuery.data ?? []}
    />
  )
}
