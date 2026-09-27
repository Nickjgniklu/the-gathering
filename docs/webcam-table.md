# Webcam table

## Product slice

A webcam table is a temporary room for two to ten signed-in members. Each member is seated as
the player linked to their account, chooses a deck while in the room, shares their board camera,
and ends the game with a result form that records through `Games.create_game/2`. The resulting
`Game` and `GamePlayer` rows are deliberately
indistinguishable from a manually logged game, so existing history and statistics need no
special cases.

The owner selects a game mode before starting. Commander is the default. Two-Headed Giant
Commander requires an even roster of at least four: adjacent seats form teams, randomization
shuffles whole pairs, each team starts at 60 shared life, and turns/counts/timing are shared.
Use the lobby's seat arrows to arrange teammates. Shared life reaching zero eliminates the whole
team; eliminating or restoring a player also applies to their whole team. Counters and
commanders remain individual. The result picker records both teammates as winners.

Five Star requires exactly five seats. Each seated viewer sees “Can't attack yet” on their
original previous/next neighbours. An eliminated neighbour loses its badge, and all badges clear
once only three players remain. These are
advisory badges, not enforced attacks or alliances. Spectators see no restrictions. Seat order
is fixed after starting either new mode, including across reconnects.

Recorded games store `format` (`commander`, `two_headed_giant`, or `five_star`), also selectable
in the manual game form. Existing games and imports default to Commander. Run `mix ecto.migrate`
when upgrading to apply the `add_format_to_games` migration. Room snapshot version 2 stores mode
and team life; version 1 snapshots continue loading as Commander.

Card clicks assist the room rather than define its durable record. A click on any board fetches
a native-resolution crop from the camera owner's browser, runs the card recognizer in the
clicking browser, and shows five numbered candidates plus a gallery search; confirming one posts
an "identified" line to every seat's log. The recognizer bundle is published to the server from
`ml/` (see **Recognition** and `ml/README.md`); when the server has none, the panel falls back
to your own known commanders as deck-based suggestions (only on clicks over your own board, since
only a seat's owner may choose its commander).

## Decisions

### Ten-seat adaptive mesh now, SFU seam later

Use one `RTCPeerConnection` per pair and Phoenix Channels only for authenticated presence,
SDP, and ICE signaling. At the maximum of ten, a room has 45 connections and each
browser sends at most nine streams. This retains the one-container architecture and keeps
media end-to-end between browsers. `ex_webrtc` implements a peer connection, not an SFU by
itself; a Membrane RTC engine would add server-side media routing, UDP port exposure, process
supervision, and materially more deployment work.

`media-policy.ts` budgets each outgoing sender by **total seated players, including self**:

| Seats | `scaleResolutionDownBy` | From a 1080p camera | `maxBitrate` per receiver |
| --- | --- | --- | --- |
| 1–4 | 1 | 1920 × 1080 | 2,500,000 bps |
| 5–7 | 1.5 | 1280 × 720 | 1,200,000 bps |
| 8–10 | 2 | 960 × 540 | 600,000 bps |

The hook applies these caps through `RTCRtpSender.setParameters` when a connection becomes live
and when membership changes (including restoring the higher tier after departures). Congestion
control may reduce quality further. At ten seats the video budget is at most 5.4 Mbps per
sender before transport overhead, still nine encodes and potentially nine TURN relays. Measure
sender CPU, available outgoing bitrate, frame dimensions, packet loss, and TURN use; these
tiers are budgets, not a guarantee of ten-seat performance on every device.

The SFU migration seam is `useWebcamRoom`: replace peer creation/signaling and stream delivery
behind that hook, preserving participant, stream, capture, reveal, and result APIs. The SFU
must enforce reveal subscriptions server-side and preserve the native crop RPC (a targeted data
channel or equivalent), rather than forwarding hidden video and relying on UI overlays. Move
there when measured CPU, uplink, or relay costs make the mesh impractical.

Phoenix's own Channels documentation confirms that signaling is application-defined and that
signed-token authentication belongs in `connect`; the authenticated config endpoint signs the
existing tracked session token for the socket handshake. MDN documents SDP and ICE exchange
through such a signaling service. References:

- <https://hexdocs.pm/phoenix/channels.html>
- <https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Signaling_and_video_calling>
- <https://github.com/elixir-webrtc/ex_webrtc>

### ICE, STUN, TURN, and proxies

The normal HTTPS reverse proxy must pass WebSocket upgrades for `/socket`; it never carries RTP
in the mesh case. Host candidates alone only work on one LAN: two browsers on different networks
each sit at "Connecting…" forever because neither learns a reachable address. `WEBRTC_STUN_URLS`
therefore defaults to public STUN servers (Google and Cloudflare); override it with your own or
set it to `none` for a LAN-only install. STUN gets through ordinary home routers; players behind
symmetric NAT or carrier-grade NAT also need a separately reachable TURN server configured with
`WEBRTC_TURN_URLS`, `WEBRTC_TURN_USERNAME`, and `WEBRTC_TURN_CREDENTIAL`. When a peer connection
fails the offering side restarts ICE once; a peer that stays failed is labelled "Couldn't
connect" on its tile and the Connection section points at TURN. TURN cannot be hidden behind an ordinary HTTP reverse proxy: expose its
UDP/TCP listener (commonly 3478) and preferably TURN-over-TLS (commonly 5349/443) from coturn or
another relay. Credentials are returned only from the authenticated config endpoint.

The hosted alternative is Cloudflare Realtime TURN: set `CLOUDFLARE_TURN_KEY_ID` and
`CLOUDFLARE_TURN_API_TOKEN` (create the key under Realtime → TURN in the Cloudflare dashboard) and
`TheGathering.CloudflareTurn` exchanges that long-lived key for per-join credentials via
`POST https://rtc.live.cloudflare.com/v1/turn/keys/:id/credentials/generate-ice-servers`. The
credentials expire after `CLOUDFLARE_TURN_TTL_SECONDS` (default six hours, longer than a game;
refreshing mid-session would need `RTCPeerConnection.setConfiguration`). The config endpoint
appends Cloudflare's servers after the static ones, dropping URLs the static list already covers,
and falls back to the static list with a logged warning if Cloudflare is unreachable, so a
Cloudflare outage degrades to STUN-only rather than blocking the room. Only pairs that cannot
connect directly use the relay; Cloudflare bills relayed egress at $0.05/GB after the first
1,000 GB each month (STUN at `stun.cloudflare.com` is free and unlimited). One relayed 1080p
player in a four-seat, three-hour game is roughly 20 GB.

### Browser inference in the clicking browser

The detector, perspective warp, art crop, embedder, and cosine search run in the browser of the
player who clicked, on the crop the camera owner already returns over the data channel (see the
next section). Running it there rather than in the owner's browser costs nothing extra: the crop
transfer already solves the resolution problem, each browser loads the bundle once, and the
result needs no second round trip before it can be shown, corrected, and announced. ONNX Runtime
Web (`onnxruntime-web`) executes the three exported graphs on its single-threaded WASM backend;
the glue around them (`recognition/pipeline.ts`) is a line-for-line port of the Python
`cardid.bundle` reference runtime, and the two give the same top five on rendered scenes. WebGPU
is a later optimisation. References:

- <https://onnxruntime.ai/docs/tutorials/web/>
- <https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html>

Do not run inference in Phoenix: Ortex/Nx would make the application host the hot compute path,
complicate CPU portability, and upload imagery. A Python sidecar could remain in one container,
but would add a second supervised runtime and duplicate the spike runtime in production.

### Remote clicks use the source camera's native frame

`getUserMedia` requests a hard minimum of 1920 × 1080. A click on a remote tile is sent as
normalized coordinates over that pair's WebRTC data channel. The camera owner's browser maps the
coordinates to its native `videoWidth`/`videoHeight`, captures the same 640 px JPEG crop used by
`cardid.capture`, and returns it on the data channel together with the click position inside the
crop. The requester recognizes the card from that crop.

Video flips are a viewer-only preference (players can flip their own preview too; the sent
stream is never flipped), so the click is mapped back to unflipped source
coordinates before it is sent, and the owner always crops native pixels. The requester then
mirrors the returned crop and its click position to match its own flip of that board
(`orient-crop.ts`), so recognition, the picker thumbnail, and correction uploads see the card as
it appears on screen.

This protocol does not depend on the resolution selected by WebRTC congestion control (a
960 × 540 received stream still yields a crop of the owner's 1920 × 1080 frame) and keeps the
click-to-candidate latency budget local: capture + detector + embedder + gallery search, with no
server image round trip.

## Lifecycle and ownership

```text
Games page Play / Join button ────▶ /table/:roomId
                                         │
                             auto-seat linked player
                                         │
                              choose deck in the room
                                         │
                            browser WebRTC mesh (≤ 10)
                                         │
                                 click End game
                                         │
                ╭────────────────────────┴───────────────────────╮
       complete result form                             End without recording
                │                                                │
POST /api/games → Games.create_game/2                            │
                │                                                │
                ╰──────────────▶ end_game: room closes ◀─────────╯
                                         │
                   recorded: saved game page · otherwise: games list
```

Rooms are UUID-addressed and durable in SQLite's `webcam_table_sessions`. The serialized state
server loads a room lazily on join and commits a versioned server-owned snapshot before acknowledging
each mutation. Seats, life/counters/damage, selected commanders, elimination, monarch, order,
turn counts/times, timer, identified card lists and the table log survive reloads and server
restarts. Media is not stored. Immediate writes avoid a debounce data-loss window;
this remains a single-server design, not a distributed room coordinator.

Ending the game closes the room. After the owner records the result, or chooses End without
recording (confirmed inline), the browser sends `end_game`. `WebcamTables.close/1` deletes the
snapshot and stops the room with `{:shutdown, :closed}`. Every joined channel sees that exit on its
room monitor, pushes `table_closed`, and stops normally, so browsers leave instead of rejoining a
fresh room under the same id. Other seats return to the games list with a notice. The table drops
off the Play/Join list at once, and its seats no longer count as taken.

Rooms keep running after their last seat leaves. `TheGathering.WebcamTables.Pruner` runs every
minute: it closes rooms that have had no connections and no activity (joins, saved changes or
disconnects) for 30 minutes, deleting their snapshots, and deletes snapshots that have expired.
Snapshots expire seven days after their last write; running rooms refresh theirs hourly, so expiry
only matters for sessions left behind by a server restart. Joins reject expired snapshots.
Finished games use the same idle policy; recording a result does not delete the room. A closed or
expired UUID opens a fresh lobby.

The authenticated player ID owns the seat, not the transient peer ID or Presence entry. A newer
connection takes over the same seat, stops the old channel, and remaps order, monarch and cards.
Each channel retry uses a fresh media peer ID so browsers rebuild the WebRTC mesh rather than
keeping mismatched connections after a signaling restart. Video elements are muted (there is no
table audio), allowing spectators to autoplay without a camera grant or a prior click.
Disconnecting does not advance the turn. The first timer start (through `start_game` or legacy
`seat_order`) locks the roster: returning players reclaim their seats, everyone else spectates.
Spectators receive boards/cameras without requesting camera permission and cannot mutate the game.
They stay in Presence (with `spectator: true`) but take no seat; the Table tab lists them under
turn order, and the Setup header shows their count.
The first seated player owns table setup, timer and turn-count corrections; players retain their
own life/counter/commander controls. Any seated player can pass the turn.

Join replies carry the authoritative seat, and `table_state` includes `seats`, `owner_id`,
`monarch` and `cards`. Clients refresh the one-day socket token (encrypted with
`Phoenix.Token.encrypt`, so page scripts cannot read the session token inside it) on connection
failure. Game state survives restarts: rooms reload their saved session on the next join.

The server bounds untrusted input: `peer_id` must be a canonical UUID (clients use
`crypto.randomUUID()`), WebRTC `signal` payloads are capped at 64 KB of JSON, and the websocket
refuses frames over 128 KB. Every channel event spends a token from a per-connection bucket
(signals have their own, larger bucket) and replies `{reason: "rate limited"}` when it is empty;
joins and TURN credential requests (`GET /api/webcam-table/config`) are limited per account.
Limits live under `config :the_gathering, TheGatheringWeb.RateLimit`.

The Games page still finds open tables without the URL: `GET /api/webcam-table/rooms`
(`TheGatheringWeb.WebcamTableRooms`) lists every running room, and every seated channel process
also tracks itself on one lobby presence topic that supplies each room's connected players (join
order, `full` at ten seats). `PlayActions` (`features/webcam-table/play-actions.tsx`) polls it
every 15 s: with no open table the header shows **Play**; with one it shows **Join** naming the
seated players (or "Empty table") plus a smaller **New table**; with several, Join becomes a menu
of tables. An empty room stays listed until the pruner closes it; after a server restart, rooms
reappear once someone opens their saved URL.

## Table view layout

The room is laid out like a webcam play surface rather than a video-call grid: by default one
board is large, everyone else is small, and controls live in a collapsible column. Grid view
(Settings → View, or `G`) shows every camera at once instead.

```text
┌──────────┬─────────────────────────────────────────────┬──┬──────────────┐
│ rail     │ 40                                          │  │ Setup  3/10  │
│ ┌──────┐ │                                             │▪ │ Invite       │
│ │40    │ │                                             │▪ │ Commander    │
│ └──────┘ │              active board                   │▪ │ Turn order   │
│ Mara ⋯ 📷│           (click = identify card)           │  │ Randomize    │
│ Select…  │                                             │  │ End game     │
│ ┌──────┐ │                                             │  │ Leave table  │
│ │37    │ │                                             │  ├──────────────┤
│ └──────┘ │                                             │  │ Identify  ▸  │
│ Cody ⋯ 📷│                                             │  │ Connection ▸ │
│ Open seat├─────────────────────────────────────────────┤  │              │
│          │ Theo ⋯                       📷  Select cmd │  │              │
└──────────┴─────────────────────────────────────────────┴──┴──────────────┘
```

- **Camera rail** (left, `lg:` defaults to 15 rem): every seat as a 16:9 tile with one life control
  over the video and a single-line name / ⋯ menu / camera indicator / commander bar. Hover or
  focus your life box (tap on touch screens) to reveal stacked ±1 buttons and the counters chevron.
  Your life box is also a text field: type a new total and press Enter (or click away) to apply it
  as one change; Escape or a non-number discards the edit.
  Other seats have read-only life and an always-visible chevron to inspect their counters.
  Once a commander is chosen, a tax badge (`+4`, or `+4/+2` for partners in name order) sits just
  left of the commander name on every seat; adjust it from the counters panel or tax hotkeys.
  The ⋯ menu offers flip video (vertical or horizontal, your own preview included) and
  eliminate/restore for every seat; your own menu also has
  camera on/off and Reveal hand, opening the existing private-reveal flow in a dialog.
  Commander names truncate when necessary; the full name remains in the title/hover preview.
  Empty seats up to ten render as dashed "Open seat" placeholders. The rail scrolls vertically
  on desktop and horizontally in 240px tiles at narrow widths rather than shrinking ten cameras until their
  names and controls are unreadable. Clicking a tile soft-pins that board over the active turn;
  clicking the same tile again releases it.
- **Commander identity** colors both the rail and active-board name bars: one muted solid for
  mono-color, a WUBRG-ordered gradient for multiple colors, and neutral for colorless or unknown.
  The active-board commander name shows identity pips using the existing mana symbols; compact
  rail bars omit pips and the YOU tag to preserve name space at minimum width. The source is the decks
  API's stored `color_identity` (including partners), not a browser Scryfall request. An empty
  identity is treated as unknown, not falsely labelled colorless; explicit `C` shows its pip.
- **Active board** (center): the stage fills the remaining viewport. The same life control sits
  top-left and the matching name bar underneath carries the seat menu, camera state, and "Select
  commander" popover without repeating life. The stage follows the active turn; before turns
  start it shows the newest remote joiner, and when the shown player leaves it falls back to your
  own board. While a board is soft-pinned, a Follow turn button top-right releases it. Clicking the
  video starts the click-to-identify flow, and the suggestion card floats bottom-center over the
  stage (keys 1–5 still pick).
- **Grid view** replaces the rail and active board with every seat's camera in a near-square
  grid (Two-Headed Giant teams stay together under their shared life). Tiles letterbox rather
  than crop so each whole board stays visible. Clicking a tile fills the stage with that board,
  restoring the rail and card identification, until the same tile or the Back to grid button is
  clicked. A soft pin belongs to the view it was made in, so switching views starts unpinned.
- **Side panel** (right): a narrow icon strip (Table, Decks, Cards, Log, Settings) plus a collapse chevron and shortcut help. The
  Table tab holds the Setup section (elapsed-time badge and players count in the header, Invite
  players copies the room URL, Select your commander, a turn-order table with #/Player/Turn/Time
  (life and commander under the name, and up/down arrows for the owner), then before the match
  the primary Start match button beside Randomize, and after it Pass turn, Un-pass and Pause/Resume
  timer; Reveal hand to…, the red End game button, Leave table), dice/coin controls, and collapsed Identify
  cards and Connection sections. Decks lists your commanders; Log shows the table event log. Collapsing the
  panel leaves only the icon strip so the board grows.
- **Resize dividers** on desktop drag the camera rail (176–360 px, default 240) and panel content
  (240–480 px, default 288); widths also cap at 24vw / 32vw to preserve board space. Double-click
  resets one divider. Focus a divider and use Left/Right to resize by 16 px, Home to reset.
  Settings offers a reset for both widths and a hotkeys toggle. Preferences persist per player
  in this browser under `the-gathering:table-preferences:<playerId>`; they are not room state.
- **Keyboard shortcuts** (Convoke-compatible where the table has the same feature):
  `Space` passes the turn after the match starts and `Shift+Space` un-passes it; `↑` / `↓` gains/loses one life and
  `Shift+↑` / `Shift+↓` gains/loses ten life (always your own seat). `[` / `]` subtracts/adds
  two commander tax by changing your primary commander's cast count by one; partner commanders
  retain individual counter rows. `C` toggles your camera, `B` collapses/expands the panel,
  `T` / `D` / `A` / `L` / `S` opens Table / Decks / Cards / Log / Settings, and `,` / `.`
  soft-pins the previous/next board, and `G` toggles grid view. `?` or `H` toggles the grouped
  shortcut dialog.
  All table actions, including Space, honor the enable preference and pause while typing,
  using a keyboard widget, or while an overlay/picker is open. Ctrl/Alt/Meta, composing and
  repeat events are ignored; Space preserves native button/link activation. The card picker
  retains `1`–`5` and `/` gallery search. Escape closes overlays even with shortcuts disabled.
- **Settings** stacks collapsible Keyboard shortcuts, View, Camera, Sound and Card scan sections.
  View switches between following the active turn and the all-cameras grid; clicking or cycling
  a board soft-pins it without changing the saved view. Left/Right swaps the side panel and camera rail on desktop;
  narrow layouts keep cameras above and controls below. Glass/Classic uses the existing global
  theme-style preference. Camera lists available devices, remembers the choice and enabled state,
  and replaces outgoing tracks on existing peer connections without leaving the room. Camera-off state and
  private-reveal restrictions survive switching. Missing saved cameras fall back to the system
  default with a warning. Capture requests ideal 1080p (lower-resolution devices are accepted).
  Publisher quality offers Auto (the existing seat-count tiers), 1080p, 720p or 540p ceilings;
  it changes sender scaling/bitrate without lowering native card-crop resolution. Stats sample
  each connection every two seconds while enabled: remote tiles show received resolution, fps,
  bitrate and the selected remote ICE candidate type; the local tile shows native capture
  resolution/fps (no network hop). Check video health reports local track settings and state.
  Turn sound is an opt-in WebAudio tone, unlocked by interaction, on transitions to your turn.
  Card scan contains recognition bundle status/version and the existing corrections-sharing
  opt-out (`the-gathering:share-card-corrections`). Other table preferences use the per-player
  browser key above; sound starts on, stats start off, and the view starts on follow-turn. There is no microphone, hand-count
  or token-copy binding because those features do not exist here.
- The `/table/*` routes force the dark theme (`TableShell` in `routes/__root.tsx` swaps
  `data-theme` on mount and restores the user's choice on unmount) so portalled popovers and
  dialogs match the black stage. They render no application header.

### Creating and editing commanders in the room

The commander popover and Decks tab offer **New commander…** and, for the selected deck,
**Edit commander…** to its owner or an administrator. The dialog reuses the Decks feature's card
search and printing picker. The optional second card can be a partner, Background, companion, or
casual pairing; the table does not enforce deck-building legality. The deck name follows the
commander names until edited. Saving creates or updates the player's ordinary deck through the
decks API, then selects it in the room without navigation. Editing changes the saved deck, not
just this session's appearance.

Both names appear in the rail, board bar, and turn order. `DeckSummary.commander_image_url` and
`partner_image_url` expose full-card images, preferring the selected printing and falling back to
the catalog; the existing `*_art_crop_url` fields use the same printing preference for crops.
After a validated `choose_deck`, the channel broadcasts `deck_selected` so every browser refreshes
its deck query, including when the same deck ID is selected after an art or partner edit.

### Shared seat state

Presence metadata carries, per seat, `life` (starts at 40), `camera_off`, `eliminated` (false), and a
server-stamped `joined_at`. Players publish their own changes through the channel's
`update_status` event (validated: life −999…999, camera boolean, and counters below; no other keys) and the channel merges
them into presence, so every browser shows the same totals without another round trip. Your own
life is also tracked locally so rapid ± clicks compound before presence echoes back. On every
rejoin, authoritative life and counters hydrate those local controls before editing resumes.

The chevron below each life box opens **Counters**. Everyone can inspect a seat; only its
owner can change its counters. `poison` and `rad` start at zero. `commander_casts` maps commander
names to command-zone cast counts. Explicit −2 / tax / +2 rows replace the small art tax buttons,
with art thumbnails and separate rows for partners/backgrounds. These still publish ±1 cast deltas and display twice
the cast count, bounded at zero and 999 casts. Poison, rad, commander damage and the monarch action
share this panel. Other players' tax is read-only. Custom counters are not supported.
Commander-name text uses the deck's color identity (gold for multicolor). Both commander
and partner/background names come from the selected deck. `commander_damage` maps opposing
player IDs to commander-name/count maps, keeping identical commanders at different seats separate.
Recorded damage stays visible when a source changes deck or leaves. Damage does not adjust life
automatically. Poison at 10 and damage of 21 from any one commander are flagged in red; damage
from separate commanders is never combined for the threshold. The server accepts only integers
0…999, maps of at most 100 entries, and names of 1…300 bytes. Counter changes use optimistic local
deltas and restore from the server on rejoin and full page reload.

**Take the monarch** claims the crown for your own seat. `take_monarch` has an empty payload;
The room process serializes claims and broadcasts one `monarch` holder, never per-seat flags.
Late joiners receive `monarch_state`; server revisions prevent stale snapshots from replacing
newer claims. A crown appears on the holder's tile and active board and survives disconnects
and server restarts along with the rest of the room snapshot.
Counter changes and monarch transfers are added to the shared Log.

Once the match has started, the room owner can eliminate or restore any present seat, and each
player their own, from the ⋯ seat menu on the video tile or board bar; nothing is eliminable in
the lobby.
`set_eliminated` validates a present peer ID and a boolean; the target channel merges it into
its own presence so later life/camera updates cannot overwrite elimination. Players may also
publish their own `eliminated` through `update_status`. In every format, a life update that
reaches zero or below marks the seat eliminated automatically; gaining life back does not restore
it, so a player is brought back only through an explicit restore. Out seats stay visible with dimmed video,
an Eliminated badge, and a struck-through name. They receive no order number; only eligible seats
are numbered, while the recording order still includes everyone. Eliminating
the current player advances the turn to the next eligible seat; disconnecting does not. If none remain there is no active
turn; restoring a player starts their next turn. Counts and accumulated time are kept.

Eliminated seats are retained in the room state and shared through `eliminated_seats`, so
leaving does not drop them from the result or from a late joiner's view. Rejoining as the same
player takes back the retained position and elimination flag. A departed seat must rejoin before
it can be restored. All seats survive the last disconnect in the durable snapshot. The End game form
suggests the only non-eliminated player as winner (when at least two players took part), with all
others defaulting to losses in recorded seat order. The result remains editable; choosing Draw
explicitly records the whole table as draws, as required by the existing game schema.

Default turn order is join order (`joined_at`, then peer id) so every browser agrees. Before the
match the owner can move any seat with the up/down arrows in the turn-order table (`arrange_seats`),
then either **Start match** (`start_game {randomize: false}`, keeps the shown order) or
**Randomize** (`start_game {randomize: true}`, shuffles the present peers — whole pairs in
Two-Headed Giant — then starts). A bare `start_game {}` still falls back to the shared
`turn_settings` option `auto_randomize`, which older clients may set. After the start the arrows
stay available in Commander only and push `seat_order`; Two-Headed Giant and Five Star lock their
order at start because team life and turns are keyed by seat position. The channel verifies each
list names exactly the retained seated peers (never spectators), then broadcasts it. Rows shuffle
visibly for about one second before settling into a randomized order (instant with reduced motion,
no animation for an ordered start or a manual move). The End game form numbers seats in that order
and records them the same way. Reordering mid-game **never resets or resumes the timer or changes
the current turn**.

Any seat can use **Pass turn** or **Space**. Turns advance in the shared order, skip eliminated seats and
wrap around. Temporarily disconnected seats keep their turns. TURN counts increment when a turn starts, including the first
turn; small −/+ controls send `adjust_turn` corrections (0–999) without changing time or active
player. `pass_turn` requires the last seen turn revision, so simultaneous passes only advance
once. TIME is the player's accumulated time in m:ss, including their current turn. The server
banks time against game elapsed time rather than wall time, so pausing freezes both game and
player clocks; passing while paused changes the turn but adds no paused time. A single remaining
player can continue taking turns; elimination does not automatically end the game.

Any seat can also **Un-pass** (**Shift+Space**) to undo a mistaken pass. The turn state keeps the
last 20 passes (who passed, when their turn started, who received it); `unpass_turn` takes the
same revision as `pass_turn` and pops the newest one: the previous player's turn resumes from its
original start, the time banked by the pass is removed, and the receiver's TURN count drops by one.
Time spent since the pass counts toward the resumed turn. Repeated un-passes walk further back.
The server refuses when the newest pass did not lead to the current turn (for example after every
seat went out and play restarted) or when the previous player has since been eliminated or left.

An amber dot and highlight mark the current player's row; an amber Current turn badge marks their
rail tile and board. This is independent of the violet active-board border. Space does
not pass while typing, using a control, holding a modifier other than Shift (which un-passes), repeating a key, composing text, or
while a card picker/dialog is open. It shares the guarded registry and enable preference in
`table-hotkeys.tsx` with the other table shortcuts.

Each room process serializes its shared timer, turns and order.
The first valid `seat_order` starts it. Only the room owner can send `timer` with `pause` or `resume`;
only the server writes `started_at`, `paused_at`, and accumulated `paused_ms`. The timer bar
above the active board's name bar derives elapsed time excluding pauses. Browsers interpolate
from a server sample using `performance.now()`, not their wall clock, and resync every 15 seconds
with `timer_sync` (half-round-trip latency compensation). New/rejoining seats receive the
current timer, order, settings, counts and per-player times via `table_state`. A running timer
includes server downtime; a paused timer remains paused. Multi-node room state is not supported.

End game pauses the timer for everyone and captures that server response for the result form.
Duration is editable, prefilled in whole minutes rounded to nearest (minimum one minute, matching
the game schema). Without a started timer it stays blank. Going back leaves the timer paused;
the owner can resume explicitly. `played_at` uses the shared start timestamp when available.

The Table tab offers d6, d20, custom dice with 2–1000 integer sides, and coin flips. The `roll`
event validates the request, generates the result on the server, stamps it with the authenticated
seat's name/id and server time, then broadcasts to everyone. Results appear in a five-second
overlay and in the Log. Clients cannot supply a result or impersonate the roller.

The Log tab is owned by the room (`TheGathering.WebcamTables.Log`): the room process writes
entries for joins and leaves, deck/life/camera/counter changes, eliminations, the monarch, seat
order and rolls, keeps the newest 200 in the saved snapshot, and broadcasts each new or merged
entry as `log_entry`. Every (re)join receives the whole log as `table_log`, so all seats see the
same history and a reload restores it. A reload or reconnect within ten seconds logs neither a
leave nor a join. Consecutive events of the same kind and actor within two seconds coalesce
(merged entries keep their id): life keeps the first and final totals, dice/coins retain every
result, and deck/camera changes show the latest state with a count. Different actors, event
kinds, and intervening entries break the group.

Audio is not part of the webcam table: no microphone is captured and there are no mute
controls. Players use their usual voice app alongside the table.

### Private hand reveal

Choose another seated player in the Table tab's **Reveal hand to…** action, and wait for the confirmation before
showing your hand. Per-peer cloned video tracks are immediately disabled for non-targets, then
their senders use `replaceTrack(null)`; the native camera and target's sender stay live.
Late joiners also start with no outgoing video track. Presence carries `reveal_to`, validated
by the channel as another current peer, so hidden seats render **Revealing to name** instead of
a video element and the target sees **name is revealing to you**. The owner still sees their
own camera. Camera off overrides reveal and stays off when reveal ends.

**End reveal** restores eligible senders in one click. A target departure automatically ends
the reveal in both channel presence and the owner's media policy, restoring public video;
put the hand down before ending or leaving. Signaling reconnects preserve the local restriction.
Admission is serialized so simultaneous final-seat joins cannot overfill the room. Another player
cannot reuse a seat's peer ID; the same authenticated player replaces their old connection.

Hidden viewers cannot identify the board: both the requesting UI and the owner's native-crop
handler enforce visibility. Authorized viewers still receive the owner's native 640 px JPEG
crop from the untouched 1080p source, unaffected by the sender tiers. Identifications during
a reveal stay local to the clicker rather than entering the shared card tray. A chosen viewer
can still save or share what they saw; this feature cannot revoke frames already delivered.

## File and component structure

- `TheGatheringWeb.UserSocket` issues and decrypts a short-lived encrypted token wrapping the tracked
  cookie session.
- `TheGatheringWeb.WebcamTableChannel` caps rooms at ten, relays targeted WebRTC signals,
  merges `update_status`/`set_eliminated` into presence, validates `seat_order`/`timer`/`timer_sync`,
  `start_game`/`turn_settings`/`pass_turn`/`unpass_turn`/`adjust_turn`, and generates
  and broadcasts validated `roll` results.
- `TheGathering.WebcamTables` is the context API for admission and game state. Each room is a
  `WebcamTables.Room` process (one per room, started on first join under
  `WebcamTables.RoomSupervisor`, registered in `WebcamTables.Registry`) that loads its
  `WebcamTables.Session` snapshot on start, saves every change before broadcasting it, and stops
  when its last connection leaves. A crashing room only disconnects its own channels, which rejoin
  from the saved snapshot. `WebcamTables.Pruner` deletes expired sessions hourly; running rooms
  refresh their own expiry (covered by `webcam_table_channel_test.exs`).
- `TheGathering.WebcamTables.Turns` owns pure turn advancement, elimination skipping, counts and
  accumulated-time accounting, and `WebcamTables.Timer` the pause-aware game clock
  (`test/the_gathering/webcam_tables/`).
- `TheGatheringWeb.Presence` owns ephemeral room membership and seat status.
- `WebcamTableConfigController` exposes authenticated ICE configuration.
- `features/webcam-table/use-webcam-room.ts` owns camera, mesh, signaling, native crop RPC,
  seat status (life, camera, elimination), seat order, timer/turn synchronization, roll overlays, and the event log.
- `features/webcam-table/use-correction-upload.tsx` uploads explicit picker labels and owns
  the crop-sharing preference/save note. Both camera owner and clicker must allow sharing.
- `features/webcam-table/webcam-table-page.tsx` composes the rail, stage, and side panel and owns
  the active-board selection (`useActiveBoard`).
- `features/webcam-table/board.tsx` — `ActiveBoard`, `CameraTile`, `OpenSeat`,
  elimination/current-turn overlays, and `capturePoint` (click → normalized coordinates).
- `features/webcam-table/life-control.tsx` — hover/focus life controls and counter entry point.
- `features/webcam-table/seat-bar.tsx` — the name/actions/camera/commander bar under a board or tile.
- `features/webcam-table/commander-picker.tsx` — popover listing your own decks; other seats see a
  read-only commander label, and the server rejects `choose_deck` for decks you do not own.
- `features/webcam-table/card-suggestions.tsx` — the click-to-identify overlay: crop with the
  detected quad, five numbered candidates, gallery search, timings, and "In deck" marks.
- `features/webcam-table/seat-decklists.ts` — loads every seated deck's linked list and warms its
  images through a small concurrency-limited queue; `deck-hint.ts` is the pure deck-list prior
  (`applyDeckHint`, `deckFirst`, exact-printing lookup); `decklist-dialog.tsx` is **Decks →
  View decklist**. `features/decks/decklist-cards.ts` holds the `GET /api/decks/:id/decklist`
  query and deck-builder grouping.
- `features/webcam-table/recognition/` — `use-recognizer.ts` (hook owning the worker and its
  checking/loading/ready/unavailable/failed state; also exposes `identifyFrame`),
  `recognizer.worker.ts` (ONNX Runtime Web sessions, warm-up, identify and full-frame scan),
  `pipeline.ts` (pure port of `ml/cardid/bundle.py`, unit-tested in `pipeline.test.ts`), and
  `messages.ts` (worker protocol types).
- `TheGathering.CardId` + `CardIdBundleController` serve the published bundle from
  `DATA_DIR/cardid/current` (`GET /api/cardid/bundle` for the manifest and file URLs,
  `GET /api/cardid/bundles/:version/:name` for the immutable files). `404` on `bundle` means no
  recognition bundle is published and the UI falls back to deck suggestions.
- `features/webcam-table/use-super-ai.ts` + `data-messages.ts` — the Super AI frame
  request/chunk/end protocol described above, validated the same way as the click-crop RPC.
- `features/webcam-table/super-ai.ts` — scan-result selection and stabilization; `super-ai-overlay.tsx`
  renders the corresponding art crops with `mapSourceQuad`'s stage-mapping math.
- `features/webcam-table/side-panel.tsx` — icon strip and Table/Decks/Cards/Log tabs
  (`cards-tab.tsx` holds the Cards tab; `panel-section.tsx` the collapsible section).
- `features/webcam-table/finish-game.tsx` — the End game dialog: record the result or end without
  recording; both close the table.
- `features/webcam-table/game-result.ts` — winner suggestion and normal recorded-game payload,
  including eliminated seats (`game-result.test.ts`).
- `features/webcam-table/table-timer.tsx` — elapsed-time badge in the Table tab header plus the
  pause/resume action; both appear once the match has started.
- `features/webcam-table/use-timer-elapsed.ts` — shared monotonic timer display hook.
- `features/webcam-table/seat-order-table.tsx` — animated order, current-turn highlight, counts,
  per-player time and elimination controls.
- `features/webcam-table/turns.ts` — next-seat suggestion and turn display (`turns.test.ts`).
- `features/webcam-table/table-hotkeys.tsx` — guarded table bindings and grouped help dialog.
- `features/webcam-table/table-settings.tsx` — collapsible browser preferences and camera controls.
- `features/webcam-table/camera.ts`, `video-stats.tsx`, `use-turn-sound.ts` — device acquisition,
  connection sampling and opt-in turn notifications.
- `features/webcam-table/game-timer.ts` — elapsed-time interpolation, formatting, and duration
  prefill helpers (`game-timer.test.ts` also covers roll descriptions).
- `features/webcam-table/table-rolls.tsx` — dice/coin controls, wire types, and result descriptions.
- `features/webcam-table/table-events.ts` — pure helpers for log lines, seat ordering, and
  shuffling, active-turn filtering, departed eliminated seats, and event coalescing
  (unit-tested in `table-events.test.ts`).
- `features/webcam-table/board-cards.tsx` — the card tray docked to the bottom of the active board
  and the shared `CardThumb`.
- `features/webcam-table/card-preview.tsx` — the card image + rules text overlay; `card-details.ts`
  fetches one printing's details from `GET /api/card-printings/:id/details`.
- `routes/table.new.tsx` and `routes/table.$roomId.tsx` are thin route adapters.

The finish mutation posts the normal game payload (`played_at`, optional duration/turns/win
condition/notes, and consecutive seats with player/deck/result) to `/api/games`. The controller
already strips provenance fields and delegates to `Games.RecordGame`.
`Game` accepts 2–10 distinct participants with consecutive seats; `GamePlayer` accepts seat
numbers 1–10 (and up to nine kills). The scrollable End game dialog and turn-order table both
render every participant in the shared order.

## Recognition

The recognizer ships as a **bundle** exported and published from `ml/` (`cardid.export`,
`cardid.publish`; see `ml/README.md`, "Shipping"). Phoenix serves whatever
`DATA_DIR/cardid/current` points at. Model bundles are neither committed to this repository nor
baked into the container image, so a new recognition or gallery bundle is a `publish` away and
browsers pick it up on their next table
because they cache bundle files by version.

Gallery coverage includes paper artwork in any language (including Japanese-only alternate
art), prepare cards, meld cards, and both scanned sides of transform/MDFC/reversible cards
and double-faced tokens. Separate sides use face names and IDs `<scryfall UUID>` (front)
and `<scryfall UUID>-1` (back); prepare/adventure share one art and keep their combined name.
Rooms, classic split, aftermath and flip cards also expose two named halves with those IDs.
Their gallery crops come from regions of the whole front scan, rotated upright before
embedding. Both visible halves can appear as candidates; this does not infer unlocked Room
doors or a flip card's active rules. Details show the chosen half's rules with the shared
whole-card image, not a fictitious reverse image. Cycling alternate printings keeps the same
split/flip half selected; full-card deck-picker requests retain combined names. Art series, battles and novelty splits
with more than two parts remain excluded. See `ml/README.md`, **Gallery coverage,
printings and face IDs**, for crop geometry and the refresh/export/publish commands. Deploying this
code alone does not rebuild the gallery bundle.

The gallery uses Scryfall's **all_cards** metadata to attach all paper printing/language
siblings to each distinct illustration, instead of letting `unique_artwork` choose the
only selectable printing. Regular-frame Sol Talisman and Essence Channeler, Nettlecyst
reprints, and early core sets such as Revised (`3ed`) and Unlimited (`2ed`) are searchable
even when another printing supplies the embedding. Placeholder-scan siblings are included
when that illustration has a usable scan elsewhere; artworks with no usable scan remain
absent. Each illustration still has one embedding/crop. Existing IDs and splits are kept;
duplicate legacy rows are retained as aliases in training metadata but not embedded twice.
Exact-printing corrections map back to the shared artwork for training and evaluation.
The separate deck printing picker remains English-only.

Room entry does not create a recognition worker or fetch models. The first card click or
gallery search starts `useRecognizer`: it fetches `GET /api/cardid/bundle`, creates the worker,
loads the three graphs plus compact `arts.json`, and warms up. Concurrent actions share this
load. The first action waits, displaying "loading card scanner…" or "Loading gallery…";
only then does the two-second inference timeout begin. Settings > Card scan shows "Not loaded"
until scanning starts, then `checking`, `loading`, `ready` (version, artwork count and load
time), `unavailable`, or `failed`. Leaving the room terminates the worker and pending actions.

New exports put sibling printing records in the separate, versioned `printings.json` file.
Only the first gallery search or expansion of a candidate's printing choices downloads it;
ordinary image identification needs only `arts.json`. The optional file is shared between
search/expansion requests and cached by bundle version. Expanding displays loading/retry
states, and choosing a sibling preserves its exact face, language, set and collector number.
Old bundles (embedded siblings or representative-only) still work. Re-export/publish a new
version on the training box to get the split; see `ml/README.md`. No room protocol changes.

Each capture runs identify with a two second timeout: detector pass over the 640 px crop, a
refine pass on the detected card, upright vote, embed all bundled art cuts (14 in new bundles;
older six-frame bundles still work), gallery search. A top-1
that leads the runner-up by at least `CLEAR_MARGIN` (0.08 cosine) is treated as the answer to a
plain click: it is recorded immediately and the **card preview** opens over the board — the
card image with its set and collector number, and (on wider screens) a box with mana cost, type
line, P/T or loyalty, and oracle text. That text is not in the catalog (which keeps one printing
per card), so `card-details.ts` asks `GET /api/card-printings/:id/details`, which
`Catalog.Printings.details/1` answers by fetching the exact printing from Scryfall once and
caching it in `card_printings`. "Wrong card?" on the preview reopens the picker for the same
crop and the chosen card replaces the entry.

For separate-side gallery entries, details select that face's name, image, mana cost, type
and rules text. The suffix is stripped only for the Scryfall request, not the details/query
cache key or correction label. The cached-printing `show` endpoint can also return either
face after details has populated it. Rulings use the base card's shared ruling list and
face-keyed cache rows. Malformed gallery IDs return HTTP 400 from details/rulings.

The **picker** (`card-suggestions.tsx`) is user-initiated only: it opens for a near-tie, when
no bundle is published (deck suggestions stand in), on "Wrong card?", or when the clicker
Shift+clicks to choose for themselves. It shows five numbered candidates (`1`–`5`), the crop
with the detected quad and per-stage timings; low similarity never suppresses results. `/`
focuses a gallery search that understands names, set codes (`forest fin`, `set:fin`),
collector numbers (`#280`) and language (`lang:ja`). A bare word that is also a set code is
read both ways, so `woe strider` finds Woe Strider as well as any WOE "strider"; use `set:`
to force the set. English results sort first. Expand a
candidate's printing count to select its exact set/number/language; choices also show frame
effects, borderless treatment and promos. The Cards tab searches the same printing choices.
Ranks/keyboard `1`–`5` remain one per artwork, not one per reprint. Identical art cannot tell
printings apart automatically; the user chooses a sibling, and its own ID drives details,
hover images, rulings and correction labels. Name-based board deduplication is unchanged.

### Super AI board scan

A second, independent recognition path alongside click-to-identify: instead of naming one card
a player clicked, it periodically identifies the visible board. It is opt-in (Settings → Card scan
→ **Super AI board scan**, off by default) and, like
clicks, runs entirely in the two browsers — the server only relays WebRTC signaling and never
sees a frame.

**Transport** (`use-super-ai.ts`, `data-messages.ts`): while enabled, the viewer's browser asks
the currently-shown board's owner for a frame every ten seconds (`super_ai_frame_request`,
rate-limited to one per ten seconds per peer either direction). The owner's browser captures its
own camera frame as a JPEG and, if the viewer is allowed to see that board right now
(`canViewBoard`: camera on, no private reveal in the way) sends it back over the existing
per-pair WebRTC data channel, base64-chunked (`super_ai_frame_start` → N ×
`super_ai_frame_chunk` → `super_ai_frame_end`, 32 KiB chunks, 4 MiB/128-chunk ceiling). The
requester reassembles the chunks, verifies the declared byte count and a SHA-256 digest before
using anything, and gives up after 12 seconds. Visibility is re-checked on both ends at every
step, so a reveal ending or a camera going off mid-transfer stops the frame going out or being
used. `parseDataMessage` (shared with the older capture-crop RPC) rejects anything outside these
shapes, sizes and ranges before it reaches application code.

**Recognition and rendering** (`recognizer.worker.ts`, `super-ai.ts`, `super-ai-overlay.tsx`):
the requester decodes the JPEG and runs the published identity pipeline over the whole frame.
Only a clear top match is retained; its Scryfall art is projectively warped onto the identified
card through the same `object-contain` mapping used by the stage video. Results are scoped to the
currently viewed board, so a completed scan cannot appear after the viewer switches boards.

### Linked deck lists

When a seat's chosen deck links to Moxfield, Archidekt or ManaVault, every seat fetches the
list from `GET /api/decks/:id/decklist` (TanStack Query, 30-minute stale time) and preloads
its `small` then `normal` images, four at a time, so neither the list nor a later preview waits on
Scryfall. The server fetches the list live (five-minute cache) and joins it to the catalog; the
app stores no copy. Images come from the list's own printing when the site records one.

**Decks → View decklist** shows your own chosen deck's list, grouped like a deck builder, with
a list/images toggle, a filter, and a check on cards already identified on your board.
Opponents' lists are loaded only for recognition and are not shown, though anyone watching the
network tab can read them.

The clicked board owner's list nudges recognition after the bundle answers
(`deck-hint.ts`): each top-5 candidate whose name, or either face name, is in the list gets
`DECK_PRIOR` (0.03) added before re-ranking. The clear-answer rule, keys `1`–`5` and the picker
all see the hinted order, and the picker marks those candidates "In deck". Gallery search from
the picker lists deck cards first. A deck card outside the recognizer's top five is not
recovered: a gallery-wide bias would need a new search-graph input and a new bundle. When a
recognized art holds the list's exact printing, found once per list by the worker's `locate`
request, that printing is recorded. A printing picked by hand is never replaced. Corrections
keep the raw result, so training data is not biased by the hint. `ml/README.md`, "Deck-list
prior", describes how to check the prior against real captures.

Hover or keyboard-focus a candidate (including gallery search results) to see a larger card
image beside the picker before choosing. The same hover preview shows a seat's commander in
the rail, board name bar, turn-order table, commander picker, and Decks tab. Commander hover
uses the deck serializer's `commander_image_url` (selected printing, or catalog default),
falling back to `commander_art_crop_url` without a Scryfall details request.

The card popup keeps **Wrong card?**, **Remove**, **Rulings**, and close together in a dark,
consistently sized toolbar. Clicking outside the visible content, including the space below
the rules text, closes it. Right-click its content or use **Rulings** to view Scryfall rulings;
the popup is the only place rulings are offered (tray entries just open the popup). The authenticated
`GET /api/card-printings/:id/rulings` endpoint fetches `/cards/:id/rulings` with Req and caches
successful results (including an empty list) in `card_rulings_cache` for one day. Errors are
not cached, and the dialog offers retry. This cache is independent of printing/catalog refreshes.

Identified cards are not game events and never appear in the Log. Confirming a card (the silent
clear match, `1`–`5`, a click, or a search result) broadcasts `card_identified` on the data
channels and every seat adds the entry to that board's **card tray** (`board-cards.tsx`): a
chevron tab at the bottom of the active board's video that unfolds a translucent shelf of card
thumbnails, each with a red × to remove it (`card_removed`, honoured at every seat) and opening
the preview when clicked. The **Cards** tab of the side panel (`cards-tab.tsx`) shows the
newest identified card with its details (Clear hides it locally), a gallery search that
previews any printing, and the detected cards grouped per player with a Shared / My board
toggle. The list is persisted through the channel's `cards` event (up to 500 entries per room)
and synchronized through `identified_cards` and `table_state`. The server stamps each entry's
`byPlayerName` from the sender's seat (any client-supplied name is ignored), and every
`identified_cards` broadcast carries the change `type` and the acting seat as
`by: {peer_id, player_name}`. Any seated player may remove any entry to correct a
misidentification; spectators cannot change the list. Data-channel messages remain for
older clients; private reveal identifications are never persisted. If the card name matches one of the owner's
commanders and they have no deck selected yet, it also selects that deck.

Only a board's owner can wipe it: the tray on your own board and your own group in the Cards
tab offer **Clear cards**, which broadcasts `cards_cleared` with your peer ID and empties that
board at every seat (other boards keep their entries). Starting a game also clears every board:
each seat watches the shared timer and drops all cards on the lobby-to-started transition, so
cards identified while waiting never carry into the game. A late joiner's first timer sample is
already started and does not count as a transition, so the `cards_sync` it receives survives.

Each board keeps **one entry per displayed card name**, trimmed and case-insensitive (the gallery
does not provide oracle IDs). Different printings of the same card do not create extra entries.
Separate face names remain distinct; full combined prepare/adventure names stay intact.
Repeat identification previews the **printing just clicked** while preserving the tray entry's
first printing and position. Remove and correction still target that existing entry;
another player's board can still hold its own entry. Picker choices use the same
rule. "Wrong card?" removes the mistaken entry, then reuses an existing replacement if present.
Incoming entries and late-join syncs also deduplicate; simultaneous discoveries choose the oldest
timestamp, then entry ID, so message arrival order does not decide which printing survives.

The popup's **printing arrows** wrap around all pages of the existing English paper printing
list, newest first. It starts on the clicked printing; if absent (for example, another language
or a face-specific gallery entry), that printing is prepended. The counter and set/collector
caption track the displayed printing; browsing never changes the tray. Neighbouring images are
prefetched (only the two neighbours, at low browser priority). Left/Right works only while the preview has focus, not while typing, using a keyboard
widget, or viewing rulings. Table shortcuts remain paused under the preview.

Preview and card hover show Scryfall USD prices: `$0.25 · Foil $1.10 · Etched $1.25`, omitting
unavailable finishes and showing `—` when none are priced. Printing details (including prices)
are fresh for one hour in TanStack Query and use `cache-control: private, max-age=3600` on the
server. The printing list also has a one-hour client cache. Loading/failure does not hide the
clicked card; failed printing lists can be retried.
Page requests are spaced by 500 ms to respect the existing Scryfall search limit. The list
excludes tokens and memorabilia, matching the catalog/details contract (Scryfall includes
memorabilia basic lands in printing searches).

All catalog card images now pass through the authenticated, same-origin image cache described
in the README. Thumbnails use `small`, previews `normal`, and art tiles `art_crop`; off-screen
images load lazily and decode asynchronously. The server shares concurrent misses across seats
and limits upstream image downloads to four at once. Disk storage is `DATA_DIR/card-images`,
bounded to 512 MiB with oldest-written eviction and 30-day expiry; browsers cache for one day
and can revalidate using ETags. No new environment variables, service worker, or image
transformations. `x-card-image-cache: hit|miss` distinguishes server disk reuse from downloads.

Backlog:

- record identified cards against the historical game (currently retained only with the room session);
- WebGPU execution provider with WASM fallback;
- shift-click manual four-corner capture when the detector misses.

Explicit picker choices (including an explicitly confirmed top-1, but never an automatic
answer) POST their native JPEG, click, quad/up vote, chosen gallery ID and original ranking
metadata to `/api/cardid/corrections`. The small checkbox below the active board opts out in
localStorage; the camera owner's preference also travels with each crop. The save note only
appears after server acknowledgement and failures never interrupt identification.

Phoenix stores the private, bounded samples under `DATA_DIR/cardid/corrections`. The offline
desktop importer creates the card warp and merges captures into `ml/data/real`; the NUC never
trains or runs Python. Admin-only export, filesystem import, manual training and optional
guarded nightly runs are documented in `ml/README.md`, **Training from in-app corrections**.
