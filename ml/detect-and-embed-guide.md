# `DetectAndEmbed` implementation guide

Code: `ml/cardid/detect_and_embed.py`, branch `feat/reproduce-table-detector-training`. Tests:
`ml/cardid/test_detect_and_embed.py` (geometry only, no checkpoint needed).

## What it is

One `nn.Module` (and one ONNX graph) that does detection, cropping, and embedding in a single
forward pass: given a full frame, find every card, warp each one straight to the recogniser's
input, and embed all of them in parallel. It wires together two already-trained, frozen models —
a table detector and the existing `Embedder` — through a new batched crop layer; **it trains
nothing itself and contains no learned weights of its own.**

```python
embeddings, scores, quads = model(images)
```

- `images`: `(N, 3, native_size, native_size)` — already normalized (see Input contract below).
- `embeddings`: `(N, MAX_CARDS, 128)` — L2-normalized, one per detection slot.
- `scores`: `(N, MAX_CARDS)` — the detector's confidence for that slot, sigmoid-space, `[0, 1]`.
- `quads`: `(N, MAX_CARDS, 4, 2)` — each card's 4 corners, full-image pixel coordinates, in the
  same order `table_detector.decode_detections` produces.

`MAX_CARDS = 20` fixed slots per image, **always**, sorted by score descending. There is no
score threshold applied inside the model — every slot is populated whether or not it corresponds
to a real card (see "Fixed top-K, not a threshold" below).

## Constructing it

Two ways to build the detector half; the embedder half is always the same.

**Single-pass** (fast, ~11ms CPU / ~5ms GPU per frame — the right choice for a smaller/focused
capture area):

```python
from cardid.detect_and_embed import DetectAndEmbed

model = DetectAndEmbed(
    table_checkpoint="path/to/repro-a-hardneg-v4/best.pt",  # plain TableCenterNet weights
    embed_checkpoint="path/to/embedder/best.pt",             # a torch Embedder checkpoint
    native_size=384,                                          # must match what you feed forward()
).eval()
```

**Tiled-fusion** (slower, higher accuracy on a full-table scan — see `tiled_fusion.py`'s own
docs for why 1920 specifically):

```python
from cardid.tiled_fusion import TiledFusionDetector
from cardid.detect_and_embed import DetectAndEmbed

tiled = TiledFusionDetector(checkpoint="path/to/repro-a-hardneg-v4/best.pt", native_size=1920)
tiled.fusion.load_state_dict(torch.load("path/to/tiled-fusion-1920/best.pt", weights_only=True))

model = DetectAndEmbed(
    detector=tiled,          # pre-built module, NOT table_checkpoint
    embed_checkpoint="path/to/embedder/best.pt",
    native_size=1920,        # MUST match the TiledFusionDetector's own native_size
).eval()
```

Pass exactly one of `table_checkpoint` or `detector` — never both, never neither (raises
`ValueError`). Any `detector` you pass must return `(heat_logits, pose, up)` at stride 4 given
`native_size`-square input; both `TableCenterNet` and `TiledFusionDetector` satisfy this, so a
future third detector variant is a drop-in as long as it keeps that contract.

**Checkpoints, where to get them:** `Nickjgniklu/mtg-models`. `repro-a-hardneg-v4` (single-pass
base, used by both variants above) is `runs/repro-a-hardneg-v4/best.pt`. `tiled-fusion-1920`'s
`.pt` is the `FusionHead` *only* (~54k params) — it will load fine into `TiledFusionDetector.fusion`
but is meaningless loaded anywhere else. For `embed_checkpoint`, use
`runs/recogniser-cfbender-oracle/best.pt` — a real, currently-deployed `Embedder` checkpoint
recovered from `cfbender/oracle`'s public repo (see that directory's `PROVENANCE.md`: confirmed
byte-identical to the deployed bundle's own recogniser by sha256, and spot-checked against the
reference crop pipeline at 0.997+ cosine similarity, not just loaded and trusted blindly). A
ready-made export using it is already at
`exports/detect-and-embed-repro-a-hardneg-v4-real-embed/detect_and_embed.onnx` if you just need
the ONNX file rather than rebuilding it — see "The frame-hypothesis gotcha" below, though, before
wiring that export's embeddings to the real gallery search.

## Input contract

`images` must be normalized exactly the way `cardid.data.to_tensor` does it (ImageNet mean/std,
NCHW, RGB) — **not** the `0.5/0.5` normalization some other vision code uses. Getting this wrong
doesn't crash; it silently produces garbage detections, which is a much worse failure mode. From
a `uint8` HWC RGB numpy array:

```python
from cardid.data import to_tensor
x = to_tensor(rgb_uint8_image).unsqueeze(0)  # (1, 3, H, W)
```

`H` and `W` must equal `native_size` exactly (resize/letterbox before calling, not after — see
`confusion_matrix.letterbox_to_square` for the padding convention the rest of this project uses,
which pads to a square rather than cropping, so no content is lost).

## Output contract, in detail

**Scores and the fixed top-K, not a threshold.** `_topk_peaks` always returns exactly
`MAX_CARDS` slots — whatever the top 20 local-maxima are, real card or not, down to a score of
effectively 0 if fewer than 20 peaks exist at all. **The caller must threshold `scores` before
treating a slot as a real detection.** This project's default threshold is 0.3, *except*
`tiled-fusion-1920`, which needs 0.15-0.16 (see `tiled_fusion.py`'s docs for why — its bigger
canonical grid scores confidence on a different scale). Never assume 0.3 is correct for a new
detector variant without re-checking against the golden real-capture set.

**Quads are already fully resolved, printed-orientation-first.** `_resolve_orientation` handles
the 180-degree ambiguity in the detector's raw angle output internally; you get back a quad whose
corner order matches `decode_detections`' convention directly. No further reordering needed.

**Embeddings are a *single* frame hypothesis (`DEFAULT_FRAME = "modern"`), not all fourteen.**
This is the detail most likely to trip up a real integration. The production gallery search
(`search.onnx`, built from `graphs.SearchGraph`) expects **14** embeddings per query — one per
`detect.FRAME_NAMES` entry — because it looks up each gallery art's *own* frame index into that
14-row batch (`sims.gather(1, self.frames)`) to compare it against the correct hypothesis. Passing
`DetectAndEmbed`'s `(N, MAX_CARDS, 128)` output (one embedding per card, not 14) into the existing
`search.onnx` as-is is **not compatible** — it will either error (frame index out of range for any
gallery art whose native frame isn't index 0) or, if you reshape around that, silently compare
against the wrong hypothesis for every non-modern-frame gallery entry. Confirmed by reading
`SearchGraph.forward` directly, not assumed.

Three real options, none implemented yet:
1. **Extend the crop layer to produce all 14 frame-window crops per card** (stack more crops the
   same way tiles are stacked today — the module docstring already flags this as "a
   straightforward extension, not a redesign") and embed all 14, matching `embed.onnx`'s own
   `(scene, quad) -> (14, 128)` contract exactly.
2. **Export a simplified search graph** that skips frame-aware lookup entirely (plain top-k
   cosine similarity against a single embedding, no `frames`/`penalties` buffers) — cheaper to
   build, but throws away the frame-penalty calibration the real gallery search already has, and
   will under-perform on non-modern-frame cards specifically (this module only ever crops the
   "modern" window, so a non-modern card's crop is *already* systematically wrong before search
   even happens with option 2 -- option 1 is the only one that actually fixes that).
3. **Accept degraded matching on non-modern-frame cards** and ship as-is for now, revisiting if
   real usage shows this matters. Reasonable for point-and-click (most cards are modern-frame);
   less reasonable for a full-table Super AI scan where non-standard frames come up often enough
   that this project built a whole training phase around them.

The earlier end-to-end validation in this session's history (the "full pipeline test" scoring
92.9%/93.3% top-1 identification) used the *separate*, existing `embed.onnx` (scene, quad) call
for the embedding step, deliberately bypassing this gotcha — it validated the detection+crop
geometry, not `DetectAndEmbed`'s own embedding output wired to search end-to-end. Don't cite that
number as proof this module's embeddings work with `search.onnx` today; they don't, yet.

## Tested end-to-end (`evaluate_detect_and_embed.py`), with a real embedder

Using the recovered real `Embedder` checkpoint (`recogniser-cfbender-oracle`) and a simplified
single-embedding gallery search (bypassing the `search.onnx` incompatibility above by comparing
this module's own embedding directly against the gallery's raw vectors, extracted from
`search.onnx`'s ONNX initializers -- see `evaluate_detect_and_embed.load_gallery`), on 10
gallery-verified synthetic scenes (63 cards):

| detector | native_size | detection recall | modern-frame top1 / top5 | non-modern top1 / top5 |
|---|---|---|---|---|
| single-pass | 384 | 98.4% | 72.7% / 84.1% | 77.8% / 88.9% |
| tiled-fusion-1920 | 1920 | 93.7% | **93.0%** / 95.3% | 78.9% / 84.2% |

**A real, previously-undocumented finding, not a bug:** single-pass `DetectAndEmbed` identifies
meaningfully worse than the reference `embed.onnx` pipeline (72.7% vs. the 92.9% cited above) --
not because its crop math is wrong (verified pixel-identical to the reference crop for a
land card, cosine similarity 0.997+ against the reference embedding for the same card, see
"Constructing it" above), but because at `native_size=384` the crop has to come from an
*already-downsampled* 384px canvas: a normal card is only ~40px across on that canvas, and
upsampling that to the embedder's 128px input for a good embedding loses fine art detail the
reference pipeline never throws away (it crops from the *original*, undownsampled image). Running
the same embedding step through the tiled-fusion detector instead -- which keeps the crop source
at true native resolution -- closes almost the entire gap (93.0%, matching the reference number).
**If embedding quality matters more than latency, prefer wrapping a `TiledFusionDetector` even for
a smaller capture area, or extend `DetectAndEmbed` to crop from a higher-resolution source image
than the one fed to a single-pass detector** (not implemented -- would need decoupling the
detector's own input resolution from the crop layer's source resolution, currently the same
`native_size` for both).

One bug found and fixed *in the test harness itself* while building this, not in `DetectAndEmbed`:
an early version zipped ground-truth cards (scene order) directly against detected slots (score
order) -- two unrelated orderings -- making every identification look essentially random until
caught by visually verifying one specific card's crop against the reference pipeline and finding
the crop was correct even though the reported name wasn't. Matching each detection to its
ground-truth card by IoU first, *then* reading that matched card's name, was the fix. Worth
repeating if you extend this script: a garbled identification result is not automatically a model
problem.

## Exporting to ONNX

```python
import torch
x = torch.randn(1, 3, model.native_size, model.native_size)
torch.onnx.export(
    model, x, "detect_and_embed.onnx",
    opset_version=18,
    input_names=["image"],
    output_names=["embeddings", "scores", "quads"],
    dynamo=False,  # the legacy TorchScript-based exporter; the dynamo-based one is untested here
)
```

Verified (both the single-pass and tiled-fusion variants) to match the torch model to
floating-point noise via onnxruntime:

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("detect_and_embed.onnx")
onnx_out = sess.run(None, {"image": x.numpy()})
torch_out = model(x)
for o, t in zip(onnx_out, torch_out):
    print(np.abs(o - t.numpy()).max())  # ~1e-8 (embeddings/scores), ~1e-2 (quad pixel coords)
```

File size with the real embedder: 8.1MB (single-pass). Not yet re-measured for the tiled-fusion
variant with the real embedder (early testing with a placeholder embedder put it at ~42MB, since
it carries 4 tile passes' worth of intermediate compute graph plus the fusion head).

## Known simplifications (all deliberate, all documented as extension points, not dead ends)

- **Fixed top-20 detections, not a threshold** — see above; caller must threshold `scores`.
- **One frame hypothesis only** — see above; blocks direct `search.onnx` compatibility today.
- **`native_size` is fixed at construction, not runtime.** Feeding a different-sized image than
  the one passed to `__init__` will produce wrong (not necessarily crashing) output — the crop
  geometry and canonical-grid math are baked in at construction time. Build a new instance per
  resolution you need to support, or use `tiled_fusion.TiledFusionDetector`'s own resolution
  parametrization (it takes `native_size` as a constructor arg for exactly this reason) if the
  detector half needs to vary.
- **No batching validation across different detectors.** `TableCenterNet` and `TiledFusionDetector`
  both happen to share the same `(heat, pose, up)` output contract at stride 4, but that contract
  is enforced by convention (matching docstrings), not by an interface/type check anywhere in the
  code. A future third detector variant that doesn't honor it will fail silently, not loudly.

## Testing this yourself

`test_detect_and_embed.py` covers the geometry helpers (`_resolve_orientation`, `_topk_peaks`)
without needing any checkpoint — run it first after any change here:

```sh
uv run python -m unittest cardid.test_detect_and_embed -v
```

For an end-to-end sanity check against a real checkpoint, the pattern used to validate this module
originally: render or load a real scene, run `DetectAndEmbed`, and compare its crop against
`detect.warp_card` + `detect.art_crop` on the same detected quad (should match to within a few
pixel values, explained entirely by interpolation method differences, not a bug if larger).
