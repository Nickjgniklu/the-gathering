import { Eye, Keyboard, ScanSearch, Video, Volume2 } from "lucide-react"
import { useState, type ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { useTheme } from "@/lib/theme"
import { useCameraDevices, videoHealth } from "./camera"
import { isPublisherQuality } from "./media-policy"
import { PanelSection } from "./panel-section"
import type { RecognizerState } from "./recognition/use-recognizer"
import { describeRecognizer } from "./side-panel-labels"
import type { useTablePreferences } from "./table-preferences"
import { CorrectionPreference, type useCorrectionUpload } from "./use-correction-upload"
import type { useWebcamRoom } from "./use-webcam-room"

function Toggle({
  children,
  checked,
  onChange,
  disabled,
}: {
  children: ReactNode
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
}) {
  return (
    <label className="flex items-center justify-between gap-3 text-xs text-base-content/80">
      {children}
      <input
        type="checkbox"
        className="toggle toggle-sm toggle-primary shrink-0"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  )
}

export function Choice({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: readonly { value: string; label: string }[]
}) {
  return (
    <div>
      <p className="mb-2 text-[0.65rem] text-base-content/60">{label}</p>
      <div className="flex rounded-lg bg-black/20 p-0.5" role="group" aria-label={label}>
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`min-w-0 flex-1 rounded-md border px-2 py-2 text-[0.65rem] ${value === option.value ? "border-primary/50 bg-primary/20 text-base-content" : "border-transparent text-base-content/60 hover:bg-white/5"}`}
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export function TableSettings({
  preferences,
  room,
  recognizer,
  corrections,
  onHelp,
}: {
  preferences: ReturnType<typeof useTablePreferences>
  room: Pick<
    ReturnType<typeof useWebcamRoom>,
    "cameraOff" | "toggleCamera" | "localStream" | "changeCamera" | "cameraChanging" | "cameraError"
  >
  recognizer: RecognizerState
  corrections: ReturnType<typeof useCorrectionUpload>
  onHelp: () => void
}) {
  const { themeStyle, setThemeStyle } = useTheme()
  const { devices, error } = useCameraDevices(room.localStream)
  const [health, setHealth] = useState<string | null>(null)
  return (
    <div
      aria-label="Table settings"
      className="grid gap-2 p-2 text-xs [&>section]:rounded-xl [&>section]:border [&>section]:border-white/10 [&>section]:bg-base-100/40 [&>section>button]:h-11"
    >
      <PanelSection title="Keyboard shortcuts" icon={Keyboard}>
        <div className="grid gap-3">
          <Button type="button" variant="outline" size="sm" className="w-full" onClick={onHelp}>
            Keyboard shortcut help <kbd className="kbd kbd-xs">?</kbd>
          </Button>
          <Toggle checked={preferences.hotkeys} onChange={preferences.setHotkeys}>
            Enable keyboard shortcuts
          </Toggle>
        </div>
      </PanelSection>
      <PanelSection title="View" icon={Eye}>
        <div className="grid gap-4">
          <Choice
            label="View mode"
            value={preferences.viewMode}
            onChange={(value) =>
              preferences.update({ viewMode: value === "grid" ? "grid" : "follow" })
            }
            options={[
              { value: "follow", label: "Follow active turn" },
              { value: "grid", label: "All cameras (grid)" },
            ]}
          />
          <Choice
            label="Side panel position"
            value={preferences.panelLeft ? "left" : "right"}
            onChange={(value) => preferences.update({ panelLeft: value === "left" })}
            options={[
              { value: "left", label: "Left" },
              { value: "right", label: "Right" },
            ]}
          />
          <Toggle
            checked={preferences.keepTrayOpen}
            onChange={(keepTrayOpen) => preferences.update({ keepTrayOpen })}
          >
            Keep card tray open between players
          </Toggle>
          <p className="text-[0.65rem] text-base-content/60">
            Click a player's camera to show their board instead of the active turn, or full size in
            grid view; click it again to go back. The camera rail sits opposite the side panel on
            desktop.
          </p>
          <Choice
            label="Theme style"
            value={themeStyle}
            onChange={(value) => setThemeStyle(value === "glass" ? "glass" : "classic")}
            options={[
              { value: "glass", label: "Glass" },
              { value: "classic", label: "Classic" },
            ]}
          />
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="w-full"
            onClick={preferences.resetWidths}
          >
            Reset rail widths
          </Button>
        </div>
      </PanelSection>
      <PanelSection title="Camera" icon={Video}>
        <div className="grid gap-4">
          <Toggle
            checked={!room.cameraOff}
            onChange={room.toggleCamera}
            disabled={!room.localStream}
          >
            Enable camera
          </Toggle>
          <label className="grid gap-2 text-[0.65rem] text-base-content/60">
            Camera device
            <select
              className="select select-sm w-full bg-base-100 text-base-content"
              value={preferences.deviceId}
              disabled={room.cameraChanging || !room.localStream}
              onChange={async (event) => {
                const deviceId = event.target.value
                if (await room.changeCamera(deviceId)) {
                  preferences.update({ deviceId })
                  setHealth(null)
                }
              }}
            >
              <option value="">System default</option>
              {preferences.deviceId &&
                !devices.some((device) => device.deviceId === preferences.deviceId) && (
                  <option value={preferences.deviceId}>Saved camera (unavailable)</option>
                )}
              {devices.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Camera ${index + 1}`}
                </option>
              ))}
            </select>
          </label>
          {(room.cameraError || error) && (
            <p role="alert" className="text-amber-300">
              {room.cameraError || error}
            </p>
          )}
          {room.cameraChanging && <p role="status">Switching camera…</p>}
          <label className="grid gap-2 text-[0.65rem] text-base-content/60">
            Publisher quality
            <select
              className="select select-sm w-full bg-base-100 text-base-content"
              value={preferences.quality}
              onChange={(event) => {
                if (isPublisherQuality(event.target.value))
                  preferences.update({ quality: event.target.value })
              }}
            >
              <option value="auto">Auto — adapts to seat count</option>
              <option value="1080p">1080p — high quality</option>
              <option value="720p">720p — balanced</option>
              <option value="540p">540p — low bandwidth</option>
            </select>
          </label>
          <p className="text-[0.65rem] text-base-content/60">
            Lower if your video appears choppy. Takes effect immediately; card scans keep the native
            camera resolution.
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="w-full"
            onClick={() => setHealth(videoHealth(room.localStream))}
          >
            Check video health
          </Button>
          {health && (
            <p role="status" className="break-words text-[0.65rem] text-base-content/80">
              {health}
            </p>
          )}
          <Toggle checked={preferences.stats} onChange={(stats) => preferences.update({ stats })}>
            Connection stats overlay
          </Toggle>
        </div>
      </PanelSection>
      <PanelSection title="Sound" icon={Volume2}>
        <Toggle
          checked={preferences.turnSound}
          onChange={(turnSound) => preferences.update({ turnSound })}
        >
          Sound when it becomes your turn
        </Toggle>
        <p className="mt-2 text-[0.65rem] text-base-content/60">
          A short tone after you interact with the table. No microphone audio is captured.
        </p>
      </PanelSection>
      <PanelSection title="Card scan" icon={ScanSearch}>
        <Toggle
          checked={preferences.superAi}
          onChange={(superAi) => preferences.update({ superAi })}
        >
          Super AI board scan
        </Toggle>
        <p className="mt-2 text-[0.65rem] text-base-content/60">
          Requests a full camera frame from the visible board every 10 seconds. Frames and card
          matches stay in the two browsers and are never sent to the server.
        </p>
        <CorrectionPreference upload={corrections} />
        <p className="mt-3 break-words text-[0.65rem] text-base-content/60">
          {describeRecognizer(recognizer)}
        </p>
      </PanelSection>
      <p className="px-2 py-1 text-[0.65rem] text-base-content/50">
        Preferences are saved in this browser.
      </p>
    </div>
  )
}
