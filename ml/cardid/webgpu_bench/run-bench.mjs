// A/B benchmark for two DetectAndEmbed/DetectEmbedAndSearch ONNX exports on onnxruntime-web's
// WebGPU execution provider, in a real Chromium/Dawn browser on real hardware -- see this
// directory's README.md and ml/detect-and-embed-guide.md's "WebGPU compatibility" section for why
// this exists: a static WebGPU operator-support table cannot tell you whether ORT's own graph
// optimizer will fuse a new, unsupported op into an otherwise-fine graph (found exactly this way:
// ORT_ENABLE_EXTENDED fusing a Transpose into an unsupported FusedMatMul, forcing the graph's
// single largest op onto CPU -- a 2.7x regression an op-list audit alone could not have caught).
//
// Usage: node run-bench.mjs <model-a.onnx> <model-b.onnx> [nativeSize=384] [iters=15]
import { chromium } from "playwright"
import http from "node:http"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const HERE = path.dirname(fileURLToPath(import.meta.url))
const PORT = 8934
const WARMUP = 4

const [, , modelAArg, modelBArg, nativeArg, itersArg] = process.argv
if (!modelAArg || !modelBArg) {
  console.error("usage: node run-bench.mjs <model-a.onnx> <model-b.onnx> [nativeSize=384] [iters=15]")
  process.exit(1)
}
const MODEL_A = path.resolve(modelAArg)
const MODEL_B = path.resolve(modelBArg)
const NATIVE_SIZE = Number(nativeArg ?? 384)
const ITERS = Number(itersArg ?? 15)

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

async function bench(page, modelUrl, provider) {
  return page.evaluate(
    async ({ modelUrl, native, iters, warmup, provider }) => {
      ort.env.logLevel = "verbose"
      ort.env.debug = true

      const fetchStart = performance.now()
      const bytes = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer())
      const fetchMs = performance.now() - fetchStart

      const loadStart = performance.now()
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
        logSeverityLevel: 0,
        logVerbosityLevel: 0,
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
    { modelUrl, native: NATIVE_SIZE, iters: ITERS, warmup: WARMUP, provider },
  )
}

function fmt(r) {
  return `fetch ${r.fetchMs}ms | session.create ${r.loadMs}ms | run: min ${r.min} median ${r.median} mean ${r.mean} max ${r.max} ms (n=${ITERS}) | ${(r.bytes / 1e6).toFixed(1)}MB`
}

async function run() {
  const server = await serveStatic({ "/a.onnx": MODEL_A, "/b.onnx": MODEL_B })
  const browser = await chromium.launch({ headless: false })
  const page = await browser.newPage()

  const fallbackLines = []
  const allLines = []
  page.on("console", (msg) => {
    const text = msg.text()
    allLines.push(text)
    if (/not assigned to the preferred execution provider|fallback/i.test(text)) fallbackLines.push(text)
  })
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
  console.log("GPU adapter:", adapterInfo)

  async function scenario(label, url) {
    console.log(`\n=== ${label} ===`)
    fallbackLines.length = 0
    allLines.length = 0
    const r = await bench(page, url, "webgpu")
    console.log(fmt(r))
    console.log(fallbackLines.length ? `CPU-fallback warnings:\n  ${fallbackLines.join("\n  ")}` : "no CPU-fallback warning logged")

    // Parse "Node(s) placed on [X]. Number of nodes: N" blocks and the node list that follows --
    // an exact, complete per-provider node list straight from onnxruntime's own log, not a
    // truncated guess -- specifically to catch anything heavy (a big MatMul, a Conv) on CPU.
    const placements = {}
    let current = null
    for (const line of allLines) {
      const header = line.match(/Node\(s\) placed on \[(\w+)\]\.\s*Number of nodes:\s*(\d+)/)
      if (header) {
        current = header[1]
        placements[current] = { count: Number(header[2]), nodes: [] }
        continue
      }
      const allHeader = line.match(/All nodes placed on \[(\w+)\]\.\s*Number of nodes:\s*(\d+)/)
      if (allHeader) {
        placements[allHeader[1]] = { count: Number(allHeader[2]), nodes: [] }
        current = null
        continue
      }
      const node = line.match(/VerifyEachNodeIsAssignedToAnEp\]\s+(\S+)\s*\(([^)]*)\)\s*$/)
      if (node && current) placements[current].nodes.push(`${node[1]}(${node[2]})`)
    }
    for (const [provider, { count, nodes }] of Object.entries(placements)) {
      console.log(`  ${provider}: ${count} nodes${nodes.length && nodes.length <= 30 ? "\n    " + nodes.join("\n    ") : ""}`)
    }
    return r
  }

  const a = await scenario(`A: ${modelAArg}`, "/a.onnx")
  const b = await scenario(`B: ${modelBArg}`, "/b.onnx")

  console.log("\n=== summary ===")
  console.log(`webgpu median: A ${a.median}ms -> B ${b.median}ms (${b.median >= a.median ? "+" : ""}${(b.median - a.median).toFixed(1)}ms)`)
  console.log(`session.create: A ${a.loadMs}ms -> B ${b.loadMs}ms`)

  await browser.close()
  server.close()
}

run().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
