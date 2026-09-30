# `DetectAndEmbed` implementation guide

Code: `ml/cardid/detect_and_embed.py`, branch `feat/reproduce-table-detector-training`. Tests:
`ml/cardid/test_detect_and_embed.py` (geometry only, no checkpoint needed).

## What it is

One `nn.Module` (and one ONNX graph) that does detection, cropping, and embedding — for every one
of the 14 frame hypotheses `search.onnx` expects — in a single forward pass: given a full frame,
find every card, warp each one straight to the recogniser's input, and embed all of them in
parallel. It wires together two already-trained, frozen models — a table detector and the existing
`Embedder` — through a new batched crop layer; **it trains nothing itself and contains no learned
weights of its own.**

```python
embeddings, scores, quads = model(images)
```

- `images`: `(N, 3, native_size, native_size)` — already normalized (see Input contract below).
- `embeddings`: `(N, MAX_CARDS, 14, 128)` — L2-normalized, 14 frame hypotheses (`detect.FRAME_NAMES`
  order) per detection slot. `embeddings[n, k]` is ready to feed `search.onnx` directly for slot
  `k`, the same contract `embed.onnx`'s own `(scene, quad) -> (14, 128)` output has.
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

**Embeddings cover all 14 frame hypotheses (`detect.FRAME_NAMES`), fixed.** An earlier version of
this module cropped only the "modern" window per card, making its output incompatible with the
real `search.onnx` (which expects 14 embeddings per query -- one per `FRAME_NAMES` entry -- and
looks up each gallery art's *own* frame index into that batch). `embeddings` is now
`(N, MAX_CARDS, 14, 128)`; `embeddings[n, k]` can be passed to `search.onnx` directly, exactly like
`embed.onnx`'s own `(scene, quad) -> (14, 128)` output. Each frame's window is cropped at its own
native (possibly non-square) aspect ratio, rotated per `detect.FRAME_ROTATIONS` (`_rot90` -- a
transpose+flip reimplementation of `torch.rot90`, which has no ONNX opset-18 exporter), *then*
resized to `INPUT_SIZE`-square -- matching `detect.frame_crop`'s crop-then-rotate-then-resize
order exactly; resizing before rotating would squash a non-square box along the wrong axis for the
six frames with a 90/270-degree rotation. Verified against `detect.art_crops` directly: cosine
similarity 0.985+ on all 14 frames, including every rotated one (`test_detect_and_embed.py`'s
`FrameGeometryTest` locks in the box/rotation tables without needing a checkpoint).

## Tested end-to-end (`evaluate_detect_and_embed.py`), with a real embedder and the real search math

Using the recovered real `Embedder` checkpoint (`recogniser-cfbender-oracle`) and a torch
reimplementation of `graphs.SearchGraph.forward`'s own per-frame-gather math (not a
simplification -- `DetectAndEmbed` now produces the same 14-embedding-per-card shape
`search.onnx` itself expects), on 10 gallery-verified synthetic scenes (63 cards):

| detector | native_size | detection recall | identification top1 / top5 (all frames) |
|---|---|---|---|
| single-pass | 384 | 98.4% | 74.2% / 85.5% |
| tiled-fusion-1920 | 1920 | 93.7% | **90.3%** / 91.9% |

Per-frame breakdown (small samples per non-modern frame, but the pattern holds): non-modern frames
(`old`, `tall`, `right`, `extended`) now score *comparably* to modern, not systematically worse --
confirming the 14-hypothesis fix closes the gap for those cards specifically, not just for modern
ones. Before the fix, an earlier (mistaken) measurement using only the modern-frame embedding
found single-pass topping out at 72.7% even on modern-frame cards specifically; **that number was
capped by `native_size=384`'s already-downsampled crop source, not the frame-hypothesis gap** --
a card is only ~40px across on that canvas, and upsampling that for the embedder loses fine art
detail the reference pipeline (which crops from the *original*, undownsampled image) never throws
away. Running the same embedding step through the tiled-fusion detector instead -- true native
resolution -- closes nearly all of that gap (90.3% vs. single-pass's 74.2%). **If embedding
quality matters more than latency, prefer wrapping a `TiledFusionDetector` even for a smaller
capture area, or extend `DetectAndEmbed` to crop from a higher-resolution source image than the
one fed to a single-pass detector** (not implemented -- would need decoupling the detector's own
input resolution from the crop layer's source resolution, currently the same `native_size` for
both).

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

File size and native PyTorch latency (this desktop, CPU 4 threads / GPU RX 7900 GRE via ROCm --
not the deployed onnxruntime-web path, a relative-cost proxy only), with the real embedder and all
14 frame hypotheses:

| detector | file size | CPU | GPU |
|---|---|---|---|
| single-pass | 10.2MB | 293ms | 142ms |
| tiled-fusion-1920 | 44.3MB | 435ms | 142ms |

Both converge to the same GPU number because embedding now dominates either detector's total cost
(20 cards x 14 frames = 280 embedder passes per image, batched into one call -- see "Known
simplifications" below). That is roughly **14x** the single-frame version's latency, by
construction (14 crop-and-embed passes per card instead of one): the earlier single-frame numbers
were single-pass 20ms / tiled-fusion 21ms on GPU. Batching all 14 frames' crops into one
`self.embed` call (rather than 14 separate calls) roughly halved the GPU number (was ~240ms/207ms)
but did not meaningfully change CPU, which was already compute- rather than kernel-launch-bound.
**At ~142ms/frame even on GPU, this is not yet real-time-per-frame** (roughly 7fps) -- worth
knowing before assuming the frame-hypothesis fix is free.

## WebGPU compatibility: measured in a real browser, not inferred from a doc table

Don't "fix" this graph against onnxruntime-web's documented WebGPU operator support table without
re-measuring end to end. That was tried once: source-level changes to remove `Mod`/`Xor`/`And`/`Not`
from `_topk_peaks` (an artifact of PyTorch's ONNX exporter defensively handling a floor-division it
couldn't prove was always non-negative) plus a post-export `ORT_ENABLE_EXTENDED` optimization pass
to clean up leftover `ConstantOfShape` nodes. On paper this looked strictly better: fewer
undocumented ops, fewer CPU-assigned nodes. Measured on a real Chromium/Dawn browser on real
hardware (AMD 7900 GRE, via a Playwright benchmark), it was **2.7x slower** (94.7ms -> 253.2ms
median per-frame) than the original, unmodified graph. Both commits were reverted.

The cause, found via onnxruntime's own verbose per-node placement log (`ort.env.logLevel =
"verbose"` before `InferenceSession.create`, then reading the `VerifyEachNodeIsAssignedToAnEp`
"Node placements" block from the console) rather than any static op table: `ORT_ENABLE_EXTENDED`
fuses the top-level gallery-search matmul's `Transpose` into it as `FusedMatMul`
(`MatMulTransposeFusion`, an extended-level-only optimizer pass). `FusedMatMul` has no WebGPU kernel
in onnxruntime-web, so the single largest op in the entire graph -- the 51417x128 gallery matched
against up to 280 query columns -- silently fell back to single-threaded CPU/wasm. That one op
accounted for essentially the whole regression. Meanwhile the *original* graph's CPU-assigned nodes
(`Div`, `Mod`, `Xor`, `Cast`, `Expand`, 13 total, all tiny shape/index ops) are cheap and, per
onnxruntime's own log, entirely intentional: "ORT explicitly assigns shape related ops to CPU to
improve perf." Removing that fallback bought nothing; the optimization pass meant to help
introduced a far more expensive one in its place.

**The lesson, not just the specific bug**: a WebGPU op-support table only tells you whether an op
*already in the graph* has a GPU kernel. It says nothing about whether a graph optimizer -- ORT's
own `ORT_ENABLE_EXTENDED`, or onnxruntime-web's runtime `graphOptimizationLevel: "all"` -- will
*introduce a new, fused, unsupported op* on top of an otherwise-fine graph. Static op-list auditing
cannot catch this; only running the actual export through the actual runtime and execution provider
and reading its own node-placement log can. Re-run that check after any export-side optimization
change, not just once per source-code change -- the optimizer's behavior on the specific graph shape
is what determines placement, not the op list alone. `TopK` and `GridSample` were both flagged as
uncertain from documentation alone during the original (over-)fix; in practice neither caused a
problem -- the actual problem was an op that wasn't even in the graph until an optimization pass put
it there.

## Known simplifications (all deliberate, all documented as extension points, not dead ends)

- **Fixed top-20 detections, not a threshold** — see above; caller must threshold `scores`.
- **~142ms/frame on GPU even after batching** — see the latency table above. The MAX_CARDS=20
  budget is the other lever besides frame count: most real scenes have far fewer real cards, so a
  smaller fixed budget (fewer wasted embed passes on padding slots) is the next thing to try if
  this needs to be faster, before considering an architectural change.
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
