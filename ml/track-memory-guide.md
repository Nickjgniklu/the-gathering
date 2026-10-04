# TrackMemory integration guide

What it is, how to wire it up, and what you still have to build yourself. Companion to
`ml/detect-and-embed-guide.md` (the detector+embedder this sits downstream of) and
`ml/cardid/track_memory.py`'s own module docstring (the design rationale -- read that first if
you want the "why", this is the "how").

## What it is

Today, identifying a card from one video frame is a one-shot guess: a single bad frame (motion
blur, a weird angle, a hand passing over the card) can flip the identification to the wrong card
even though nothing about the physical card changed. `TrackMemory` is a small GRU that carries a
running belief about each *tracked* card forward across frames, so one noisy observation nudges
that belief instead of replacing it outright.

**Measured result** (`mtg-models/runs/track-memory-c`, see that repo's README for the full
writeup): on held-out synthetic sequences, smoothing took top-1 identification from 82.1% (raw,
one frame at a time) to **94.4%**, and cut the rate at which the identification flips from frame to
frame from 26.7% down to **3.9%**.

Three pieces make up the full feature; only one of them is a trained model:

| Piece | What it does | Where it lives | Exported? |
|---|---|---|---|
| Stage 1: track association | Maps each frame's unstable, confidence-ranked detections to a persistent track ID | `track_memory.associate_detections` | No -- classical, ~12 lines, port directly |
| Stage 2: TrackMemory | The trained GRU -- smooths embedding + pose per track | `track_memory.TrackMemory` | **Yes** -- `track_memory.onnx` |
| User corrections | "This detected card is really Y" | `track_identity.TrackIdentity` | No -- plain state, no weights |

## Building the full pipeline

Per video frame, per tracked card:

1. **Detect + embed** (`detect_and_embed.DetectAndEmbed`, already exported/deployed as part of the
   main app's bundle): produces, per detected slot, `scores`, `quads`, and 14 frame-hypothesis
   embeddings.
2. **Pick one embedding per slot** -- `TrackMemory` takes a single 128-d vector, not 14. Use
   whichever hypothesis the real gallery search would actually rank first (same math
   `evaluate_detect_and_embed.search`/`graphs.SearchGraph` already does: cosine similarity of each
   gallery row to *its own* frame hypothesis, minus that row's penalty, argmax). See
   `track_memory_dataset._best_frame_embedding` for the reference implementation -- this is **not**
   optional pre-processing, feeding all 14 hypotheses or the wrong one will not run correctly, it
   will just run and produce wrong answers.
3. **Build this frame's `pose_t`**: `(cx, cy, short, cos(2*angle), sin(2*angle))`. If you're calling
   `DetectAndEmbed.forward` yourself, pass `return_pose=True` to get this directly. If you're
   consuming the already-deployed three-graph bundle (`detector.onnx` + `embed.onnx` +
   `search.onnx`, see `recognizer.worker.ts`), you'll need to derive it from the quad `detect.py`'s
   corner-order convention gives you (`atan2` of the corner-1-minus-corner-0 vector is the same
   `angle` -- see `track_memory_dataset._quad_pose` for the exact formula) -- that bundle doesn't
   expose pose directly today.
4. **Associate detections to tracks** (`associate_detections`): nearest-position matching against
   last frame's tracked positions, within a gate distance (roughly half a card-width). A detection
   with no close match starts a new track (`hidden` = zero vector). A track with no match this
   frame might be a momentary occlusion -- keep it alive a couple of frames before dropping it
   rather than deleting on the first miss.
5. **(Optional) apply a pending correction** (`track_identity.step`): if this track is pinned, this
   returns the pinned card's own gallery embedding instead of step 2's raw one -- feed *that* into
   step 6, not the raw embedding. See `track_identity.py`'s module docstring for the full
   pin/auto-release contract.
6. **Run `track_memory.onnx`** for each track: `(hidden, embedding, pose, score) -> (new_hidden,
   refined_embedding, refined_pose)`. Store `new_hidden` back on the track for next frame. If a
   track is pinned (step 5), you can skip this call entirely and just display the pinned identity
   directly -- but still run it if you want the hidden state to stay coherent for when the pin
   releases (see `track_identity.py`'s docstring for why that matters).
7. **Search the gallery** against `refined_embedding` (plain cosine + frame-penalty, same as
   `search.onnx`/`graphs.SearchGraph` -- `refined_embedding` is a single already-resolved vector,
   so you need the single-query form of that search, not the 14-hypothesis one).

## `track_memory.onnx`'s contract

No batch dimension -- one track, one timestep, per call.

```
inputs:
  hidden     float32[128]  zero vector for a brand-new track
  embedding  float32[128]  L2-normalized (step 2 above)
  pose       float32[5]    cx, cy, short, cos(2*angle), sin(2*angle)
  score      float32[1]    detector confidence, sigmoid-space

outputs:
  new_hidden         float32[128]  carry forward as next frame's `hidden` for this track
  refined_embedding  float32[128]  L2-normalized; search the gallery against this
  refined_pose       float32[5]    same layout as the input pose
```

Exported with `ml/cardid/export_track_memory.py` (mirrors `export_table_detector.py`'s
conventions): opset 17, `manifest.json` with per-file sha256, verified against the torch model
over 50 unrolled random steps (feeding `new_hidden` back in each step, the real usage pattern --
max |diff| ~1e-7, floating-point noise).

```sh
uv run python -m cardid.export_track_memory --checkpoint data/runs/track-memory-c/best.pt
```

### Loading it (onnxruntime-web, same pattern `recognizer.worker.ts` already uses)

```ts
const session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] })

// per track, per frame:
const feeds = {
  hidden: new ort.Tensor("float32", track.hidden, [128]),
  embedding: new ort.Tensor("float32", bestFrameEmbedding, [128]),
  pose: new ort.Tensor("float32", poseT, [5]),
  score: new ort.Tensor("float32", Float32Array.of(detectorScore), [1]),
}
const out = await session.run(feeds)
track.hidden = out.new_hidden.data // carry forward
// search the gallery against out.refined_embedding.data
```

## What you still have to build

- **Stage 1 association, ported to your runtime.** It's classical (no ONNX export needed) but it
  is *not* optional -- without it, "track A's hidden state" and "track B's new observation" get
  blended together the moment the detector's confidence-sorted ranks shuffle, which happens
  constantly even when no card moves. See `associate_detections`'s own docstring for the exact
  failure mode and a worked example.
- **The single-query gallery search** (step 7) -- the deployed `search.onnx` expects a 14-hypothesis
  query; `refined_embedding` is a single already-resolved vector. Reuse the cosine+penalty math,
  not the graph, for this one.
- **Deriving `pose_t` from the deployed bundle's quads**, if you're not calling `DetectAndEmbed`
  directly (see step 3).
- **`TrackIdentity`, if you want corrections** -- it's tiny (a dataclass plus three functions, no
  model weights) and ports trivially to any language; see `track_identity.py`'s module docstring
  for the full pin/auto-release design and `test_track_identity.py` for worked examples of every
  code path.
- **UI/message-protocol work** for surfacing a correction affordance and showing per-track
  identification -- out of scope for this guide entirely.

## Testing this yourself

- `uv run python -m unittest cardid.test_track_memory cardid.test_track_identity` -- Stage 1 and
  the correction layer, hand-built tensors, no checkpoint needed.
- `uv run python -m unittest cardid.test_train_track_memory` -- the training-time unroll/loss
  logic, including the hidden-state-frozen-through-padding invariant that's easy to get subtly
  wrong if you ever reimplement the unroll yourself.
- `uv run python -m cardid.export_track_memory --checkpoint <path> --verify 200` to re-verify the
  ONNX export against the torch model with more unrolled steps than the default.
