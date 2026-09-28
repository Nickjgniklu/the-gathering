import { useEffect, useRef, useState } from "react"
import type { PublisherQuality } from "./media-policy"
import { useRoomLink } from "./room-link"
import type { IdentifiedCard } from "./room-types"
import { useBoardCards } from "./use-board-cards"
import { useCardCapture } from "./use-card-capture"
import { useLocalCamera } from "./use-local-camera"
import { usePeerConnections } from "./use-peer-connections"
import { useRoomChannel } from "./use-room-channel"
import { useSuperAi } from "./use-super-ai"
import { useTableGameState } from "./use-table-game-state"

export type {
  BoardCard,
  CapturedCard,
  IdentifiedCard,
  SeatStatus,
  TableParticipant,
} from "./room-types"
export type { TableEvent } from "./table-events"
export { CAPTURE_TIMEOUT_MS } from "./use-card-capture"
export { describeConnection } from "./use-peer-connections"
export { STARTING_LIFE } from "./use-table-game-state"

/** One seat at a webcam table: wires the local camera, the peer mesh, card captures, the
 * server's card list, and game state to the room channel, and exposes them as one object. */
export function useWebcamRoom(
  roomId: string,
  playerId: number,
  deckId: number | null,
  deviceId = "",
  quality: PublisherQuality = "auto",
  cameraEnabled = true,
  onSuperAiFrame?: (frame: { bytes: Uint8Array; width: number; height: number }) => Promise<void>,
) {
  const link = useRoomLink()
  const [status, setStatus] = useState("Opening 1080p camera…")
  const [error, setError] = useState<string | null>(null)
  const camera = useLocalCamera(link, deviceId, cameraEnabled)
  const peers = usePeerConnections(link, camera, quality, setError)
  const captures = useCardCapture(link, playerId, camera, peers, setStatus)
  const superAiFrameRef = useRef(onSuperAiFrame)
  useEffect(() => {
    superAiFrameRef.current = onSuperAiFrame
  }, [onSuperAiFrame])
  const superAi = useSuperAi(link, camera, peers, async (frame) => superAiFrameRef.current?.(frame))
  const cards = useBoardCards(link)
  const game = useTableGameState(link, playerId, setError)
  // Set while this seat ends the table, so its own `table_closed` is not reported back to it.
  const endingRef = useRef(false)
  const [closedByOwner, setClosedByOwner] = useState(false)

  const { spectating } = useRoomChannel(link, roomId, playerId, deckId, {
    setStatus,
    setError,
    onConfig(iceServers) {
      peers.setIceServers(iceServers)
      camera.startPlaceholder()
    },
    bind(room) {
      game.bindChannel(room)
      cards.bindChannel(room)
    },
    onPresence(everyone) {
      game.receivePresence(everyone)
      peers.syncPeers(everyone)
    },
    onSignal: peers.receiveSignal,
    onJoined(participant, owner) {
      game.hydrate(participant, owner)
      if (participant) peers.restoreReveal(participant.reveal_to ?? null)
      game.syncTimer()
      if (!link.spectator) {
        game.updateStatus({ camera_off: camera.isOff() })
        camera.startCamera(peers.replaceSourceTrack)
      }
      peers.refreshVideo()
    },
    onChannelError() {
      captures.cancelAll()
      superAi.cancel()
      peers.reset()
    },
    onClosed() {
      captures.cancelAll()
      superAi.cancel()
      setStatus("This table has ended.")
      // The seat that ended it navigates on its own; everyone else is told.
      if (!endingRef.current) setClosedByOwner(true)
    },
    onDispose() {
      peers.closeAll()
      camera.stop()
    },
  })

  /** Owner ends the table for everyone (after recording it or instead of recording it). */
  async function endGame() {
    endingRef.current = true
    const ended = await game.endGame()
    if (!ended) endingRef.current = false
    return ended
  }

  function toggleCamera() {
    const off = camera.toggleCamera()
    peers.refreshVideo()
    game.updateStatus({ camera_off: off })
  }

  /** Names a card on `ownerPeerId`'s board: added to that board's card list at every seat.
   * The capture stays current so the clicker can still say "wrong card" and pick again from
   * the same crop; the page dismisses it when it is done with the result. */
  function announceCard(ownerPeerId: string, byPlayerName: string, card: IdentifiedCard) {
    const owner = link.participants.find((item) => item.peer_id === ownerPeerId)
    const hidden =
      !!owner?.reveal_to ||
      (ownerPeerId === link.peerId && !!peers.revealTarget()) ||
      (captures.capture?.peerId === ownerPeerId && captures.capture.private)
    return cards.announceCard(ownerPeerId, byPlayerName, card, !hidden)
  }

  return {
    spectating,
    isOwner: game.isOwner,
    peerId: link.peerId,
    participants: game.participants,
    spectators: game.spectators,
    setEliminated: game.setEliminated,
    shuffleVersion: game.shuffleVersion,
    timer: game.timer,
    turns: game.turns,
    mode: game.mode,
    setMode: game.setMode,
    teamLife: game.teamLife,
    adjustTeamLife: game.adjustTeamLife,
    moveSeat: game.moveSeat,
    passTurn: game.passTurn,
    unpassTurn: game.unpassTurn,
    adjustTurn: game.adjustTurn,
    roll: game.roll,
    changeTimer: game.changeTimer,
    endGame,
    /** Another seat (the room owner) ended the table while this one was connected. */
    closedByOwner,
    rollDice: game.rollDice,
    events: game.events,
    streams: peers.streams,
    connectionStates: peers.connectionStates,
    iceServers: peers.iceServers,
    localStream: camera.localStream,
    changeCamera: (nextDeviceId: string) =>
      camera.changeCamera(nextDeviceId, peers.replaceSourceTrack),
    cameraChanging: camera.cameraChanging,
    cameraError: camera.cameraError,
    getPeerStats: peers.getPeerStats,
    cameraOff: camera.cameraOff,
    revealTo: peers.revealTo,
    revealBusy: peers.revealBusy,
    changeReveal: peers.changeReveal,
    capture: captures.capture,
    superAi,
    identifiedCards: cards.identifiedCards,
    status,
    error,
    requestCapture: captures.requestCapture,
    announceCard,
    removeCard: cards.removeCard,
    clearOwnCards: cards.clearOwnCards,
    chooseDeck: game.chooseDeck,
    life: game.life,
    changeLife: game.changeLife,
    counters: game.counters,
    adjustCounter: game.adjustCounter,
    monarch: game.monarch,
    takeMonarch: game.takeMonarch,
    toggleCamera,
    startGame: game.startGame,
    dismissCapture: captures.dismissCapture,
  }
}
