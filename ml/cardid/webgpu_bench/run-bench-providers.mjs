// Per-frame latency of ONE ONNX export under onnxruntime-web's WebGPU execution provider vs its
// WASM (CPU) one, in a real Chromium/Dawn browser on real hardware -- "how fast is this per frame
// with and without WebGPU." Same harness as run-bench.mjs (A/B between two files on WebGPU only);
// this is the other axis (one file, two providers).
//
// Usage: node run-bench-providers.mjs <model.onnx> [nativeSize=384] [iters=20]
import { chromium } from "playwright"
import http from "node:http"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PORT = 8935
const WARMUP = 4

const [, , modelArg, nativeArg, itersArg] = process.argv
if (!modelArg) {
  console.error("usage: node run-bench-providers.mjs <model.onnx> [nativeSize=384] [iters=20]")
  process.exit(1)
}
const MODEL = path.resolve(modelArg)
const NATIVE_SIZE = Number(nativeArg ?? 384)
const ITERS = Number(itersArg ?? 20)

function serveStatic(extraFiles) {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split("?")[0])
    const filePath = extraFiles[url] ?? path.join(HERE, url)
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404)
        res.end()
        return
      }
      const type = filePath.endsWith(".html") ? "text/html" : "application/octet-stream"
      res.writeHead(200, { "Content-Type": type })
      res.end(data)
    })
  })
  return new Promise((resolve) => server.listen(PORT, () => resolve(server)))
}

async function bench(page, provider) {
  return page.evaluate(
    async ({ native, iters, warmup, provider }) => {
      const fetchStart = performance.now()
      const bytes = new Uint8Array(await (await fetch("/model.onnx")).arrayBuffer())
      const fetchMs = performance.now() - fetchStart

      const loadStart = performance.now()
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      })
      const loadMs = performance.now() - loadStart

      const n = native * native * 3
      const data = new Float32Array(n)
      for (let i = 0; i < n; i += 1) data[i] = Math.random() * 2 - 1
      const feeds = { image: new ort.Tensor("float32", data, [1, 3, native, native]) }

      for (let i = 0; i < warmup; i += 1) await session.run(feeds)

      const times = []
      for (let i = 0; i < iters; i += 1) {
        const t0 = performance.now()
        await session.run(feeds)
        times.push(performance.now() - t0)
      }
      times.sort((a, b) => a - b)
      const mean = times.reduce((a, b) => a + b, 0) / times.length

      return {
        fetchMs: Math.round(fetchMs),
        loadMs: Math.round(loadMs),
        bytes: bytes.length,
        min: +times[0].toFixed(1),
        median: +times[Math.floor(times.length / 2)].toFixed(1),
        mean: +mean.toFixed(1),
        max: +times[times.length - 1].toFixed(1),
      }
    },
    { native: NATIVE_SIZE, iters: ITERS, warmup: WARMUP, provider },
  )
}

function fmt(r) {
  return `session.create ${r.loadMs}ms | per-frame: min ${r.min} median ${r.median} mean ${r.mean} max ${r.max} ms (n=${ITERS}, fps~${(1000 / r.median).toFixed(1)})`
}

async function run() {
  const server = await serveStatic({ "/model.onnx": MODEL })
  const browser = await chromium.launch({ headless: false })
  const page = await browser.newPage()
  page.on("pageerror", (err) => console.error("PAGE ERROR:", err.message))

  await page.goto(`http://localhost:${PORT}/bench.html`)
  await page.waitForFunction(() => typeof ort !== "undefined", { timeout: 30000 })

  const adapterInfo = await page.evaluate(async () => {
    if (!navigator.gpu) return { webgpu: false }
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return { webgpu: false }
    const info = adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {}
    return { webgpu: true, ...info }
  })
  console.log(`model: ${modelArg} (native ${NATIVE_SIZE}x${NATIVE_SIZE})`)
  console.log("GPU adapter:", adapterInfo)

  console.log("\n=== WITH WebGPU ===")
  const gpu = await bench(page, "webgpu")
  console.log(fmt(gpu))

  console.log("\n=== WITHOUT WebGPU (wasm/CPU) ===")
  const cpu = await bench(page, "wasm")
  console.log(fmt(cpu))

  console.log("\n=== summary ===")
  console.log(`median per-frame: webgpu ${gpu.median}ms  vs  wasm ${cpu.median}ms  ->  webgpu is ${(cpu.median / gpu.median).toFixed(1)}x faster`)

  await browser.close()
  server.close()
}

run().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
