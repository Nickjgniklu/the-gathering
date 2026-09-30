# webgpu_bench

A/B benchmark for `DetectAndEmbed`/`DetectEmbedAndSearch` ONNX exports on onnxruntime-web's WebGPU
execution provider, in a real Chromium browser on real hardware. See
`ml/detect-and-embed-guide.md`'s "WebGPU compatibility" section for why this exists: a static
WebGPU operator-support table cannot tell you whether ORT's own graph optimizer will fuse a new,
unsupported op into an otherwise-fine graph. Only running the actual export through the actual
runtime and reading its own per-node execution-provider placement log catches that. This script
does exactly that, for two exports at once, so an export change can be checked against its
predecessor before shipping.

## Setup (once)

```sh
cd ml/cardid/webgpu_bench
npm install   # installs playwright; Chromium is reused from any other Playwright install on this
              # machine's shared cache, or downloaded fresh
```

## Usage

```sh
node run-bench.mjs <model-a.onnx> <model-b.onnx> [nativeSize] [iters]
# nativeSize defaults to 384 (repro-a-hardneg-v4 single-pass); pass 1920 for tiled-fusion-1920
# iters defaults to 15 timed runs after 4 warmup runs
```

Both models must share the `DetectAndEmbed`/`DetectEmbedAndSearch` input contract: a single input
named `image`, shape `(1, 3, nativeSize, nativeSize)`, float32. Reports for each model: fetch/
session-create/run latency (min/median/mean/max), any onnxruntime CPU-fallback warning, and --
critically -- the exact per-provider node list from onnxruntime's own verbose log, so a regression
like the `FusedMatMul`-on-CPU one this tool found is visible immediately rather than only showing up
as an unexplained latency number.

Runs headed (not headless) by default since older headless Chromium builds have had inconsistent
WebGPU support; this also lets you glance at the browser window if something looks wrong.

## What to look for

- Any provider other than the one you asked for handling something expensive (a big MatMul, a
  Conv). Small shape/index ops (`Cast`, `Expand`, `Reshape`, `Div` on tiny tensors) landing on
  `CPUExecutionProvider` is normal and, per onnxruntime's own log message, intentional -- not a
  problem to fix.
- A large jump in median run time between two exports that both claim to be "the same graph, just
  cleaned up." That claim needs re-verifying here, not assuming.
