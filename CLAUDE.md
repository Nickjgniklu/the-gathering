# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The imported `AGENTS.md` above is the primary source of truth for common commands, JSON API
conventions, frontend conventions, git commit policy, and Elixir/Phoenix/Ecto/testing guidelines.
Everything below is architecture context that isn't already covered there.

## What this app is

The Gathering is a self-hosted tracker for Commander (Magic: The Gathering) games: an
Elixir/Phoenix JSON API (`--no-html --no-live`) plus a Vite/React SPA, shipped as one container
with a SQLite database. See `README.md` for the full feature list, environment variables, and
self-hosting instructions.

## Domain contexts (`lib/the_gathering/`)

Each context's `@moduledoc` is a good starting point before editing it. The core ones:

- **`Games`** — players, decks, games, seats (`game_players`). The public context is a thin
  compatibility boundary; real workflows live in dedicated modules: `Games.RecordGame` (nested
  game/seat writes), `Games.MergePlayers` (repoints every player/deck/seat/elimination
  reference), `Games.ListGames` (filterable/paginated queries), `Games.SyncRemoteDecks`,
  `Games.LinkCatalogCards`. See `docs/data-model.md` for the full schema, seat invariants (2-6
  consecutive seats, exactly one winner or all draws), and importer contract.
- **`Accounts`** — users, sessions, Discord OAuth (`SignInWithDiscord`), registration invites,
  server-wide settings (singleton `server_settings` row, e.g. open registration,
  `detailed_stats_from`). First registered user becomes the sole password administrator; everyone
  else signs in with Discord.
- **`Imports`** — parses CSV and Mythic Track exports into normalized `Imports.Game`/`Imports.Seat`
  structs, independent of storage. `Preview` resolves the write plan without committing; `Commit`
  re-resolves and writes atomically in one transaction. `SheetPreview`/`SheetCommit` reconcile the
  original Google Sheet by explicit per-row mapping instead of re-deriving identity. Games are
  deduplicated by `{source, external_id}`.
- **`Catalog`** — offline-searchable mirror of Scryfall's `default_cards` bulk data, one row per
  `oracle_id` with a deterministically chosen representative printing. `SyncServer` refreshes it
  weekly (`mix the_gathering.catalog.sync`); `Backfill` links deck/MVP card names to catalog rows
  after sync or import. `CardImages` proxies and disk-caches Scryfall art behind
  `GET /api/card-images`.
- **`Decklists`** — resolves public Moxfield/Archidekt/ManaVault deck-list URLs
  (`Decklists.Sources.*`) into normalized metadata, with an in-memory 5-minute cache
  (`Decklists.Cache`). Each source module owns its own HTTP contract and rate/host constraints.
- **`WebcamTables`** — live game rooms. Each room is a `WebcamTables.Room` GenServer started under
  `WebcamTables.RoomSupervisor` and registered by room id in `WebcamTables.Registry`; state is
  persisted to `WebcamTables.Session` after every mutation so rooms survive with zero connections,
  and `WebcamTables.Pruner` closes rooms idle 30 minutes. Mutations must be called from the joined
  connection process — the room monitors it and pattern-matches on it for seat replacement/
  elimination messages. The Phoenix channel (`lib/the_gathering_web/channels/webcam_table_channel.ex`)
  is the transport; `WebcamTableRooms` bridges channel topics to room processes. WebRTC signaling
  (mesh video/audio, adaptive resolution, private "reveal hand to") happens client-side; the server
  only relays signaling messages and (optionally) mints Cloudflare TURN credentials
  (`CloudflareTurn`). Design notes: `docs/webcam-table.md`.
- **`CardId`** — does not run any ML models. It only locates and serves whatever versioned bundle
  `DATA_DIR/cardid/current` points at (built and published by the separate `ml/` project). The
  browser downloads the ONNX graphs and runs recognition itself.
- **`Discord`** — optional Nostrum-based bot (only starts if `bot_token` is configured):
  SpellBot game tracking, `/log`, `/summary`, `/newgame` webcam-table queues. Incoming reports are
  staged (`PendingGame`) before being resolved into real games.
- **`Stats`** — read-only aggregations (`Overview`, `Player`, `Deck`, `Commanders`). Win/loss/draw
  figures use all games; figures needing seat position/duration/turns/MVP only use games on or
  after `Accounts.ServerSettings.detailed_stats_from`, since older imports may not carry that data.

## Frontend architecture (`assets/react/src/`)

- `features/{games,decks,imports,admin,webcam-table}/` hold product code; `routes/` are thin
  TanStack Router adapters over them (`routeTree.gen.ts` is generated — don't hand-edit).
  `components/` holds shared presentation, `components/ui/` the Radix-based primitives
  (`Button`, `Card`, `Dialog`, etc., ported from ManaVault).
- Server state goes through TanStack Query exclusively (see AGENTS.md for the query-key
  convention); there is no separate client-side data store.
- `features/webcam-table/` is the most stateful feature: `use-webcam-room.ts` owns the Phoenix
  channel connection and room state machine, `use-super-ai.ts`/`recognition/` own the click-to-
  identify pipeline (a Web Worker running the ONNX models fetched from `/api/cardid/*`), and
  `table-stage.tsx`/`super-ai-overlay.tsx` render the board. Changes here often touch the worker,
  its message protocol (`recognition/messages.ts`), and the hook together — check
  `recognizer.worker.test.ts` and `use-recognizer.test.tsx` for the expected message shapes before
  changing either side.

## `ml/` (card recognition, separate from the Phoenix build)

Offline Python (uv) tooling that trains/exports the card-recognition bundle the webcam table
downloads. It is never imported by or run from the Elixir/Node app — the only coupling is the
bundle file layout `CardId` reads (`manifest.json`, `arts.json`, `detector.onnx`, `embed.onnx`,
`search.onnx`) and the deck-list prior. See `ml/README.md` for the training/export/publish
pipeline (`mise run ml:retrain|export|publish|evaluate`); changes there don't need `mix precommit`
or `vp check`, and vice versa.

## Docs worth reading before deep changes

- `docs/data-model.md` — full schema, invariants, importer contract, JSON API shapes for
  players/decks/games.
- `docs/webcam-table.md` — webcam-table design notes (rooms, signaling, recognition).
- `docs/discord-integration.md` — bot setup, permissions, `/newgame` queue behavior.
- `docs/csv-import.md`, `docs/mythic-track-import.md` — import formats and admin flow.
- `docs/stats.md` — statistics API and the `detailed_stats_from` cutoff semantics.
