import { describe, expect, it } from "vite-plus/test"
import {
  clickInCrop,
  fromWindow,
  refineSide,
  resampleWindow,
  searchArts,
  upVote,
  type BundleConstants,
  type GalleryArt,
  type RgbaImage,
} from "./pipeline"

const constants: BundleConstants = {
  scene: 640,
  det_input: 256,
  rotations: 4,
  refine_fill: 0.6,
  refine_min_side: 64,
  card_aspect: 88 / 63,
  frame_names: ["modern"],
}

/** A width×height RGBA image whose red channel is x and green channel is y. */
function gradient(width: number, height: number): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4)
  for (let y = 0; y < height; y += 1)
    for (let x = 0; x < width; x += 1) {
      const o = (y * width + x) * 4
      data[o] = x
      data[o + 1] = y
      data[o + 2] = 7
      data[o + 3] = 255
    }
  return { data, width, height }
}

function pixel(window: Uint8ClampedArray, size: number, x: number, y: number) {
  const o = (y * size + x) * 4
  return [window[o], window[o + 1], window[o + 2], window[o + 3]]
}

describe("resampleWindow", () => {
  it("copies pixels one-to-one when the window side equals the output size", () => {
    const image = gradient(100, 80)
    const { window, scale } = resampleWindow(image, 50, 40, 16, 16)
    expect(scale).toBe(1)
    // window pixel (8, 8) is the click itself, (0, 0) is eight pixels up-left of it
    expect(pixel(window, 16, 8, 8)).toEqual([50, 40, 7, 255])
    expect(pixel(window, 16, 0, 0)).toEqual([42, 32, 7, 255])
  })

  it("interpolates linearly when downscaling", () => {
    const image = gradient(100, 80)
    const { window, scale } = resampleWindow(image, 50, 40, 32, 16)
    expect(scale).toBe(0.5)
    // output i samples source x = (i - 8) / 0.5 + 50, i.e. every second source pixel
    expect(pixel(window, 16, 0, 0)).toEqual([34, 24, 7, 255])
    expect(pixel(window, 16, 15, 15)).toEqual([64, 54, 7, 255])
  })

  it("samples between pixels when upscaling", () => {
    const image = gradient(100, 80)
    const { window } = resampleWindow(image, 50, 40, 8, 16)
    // output i=9 samples x = (9 - 8) / 2 + 50 = 50.5 → halfway between 50 and 51
    expect(pixel(window, 16, 9, 8)[0]).toBe(51) // 50.5 rounds up
    expect(pixel(window, 16, 8, 8)[0]).toBe(50)
  })

  it("replicates the edge outside the image", () => {
    const image = gradient(100, 80)
    const { window } = resampleWindow(image, 2, 78, 16, 16)
    expect(pixel(window, 16, 0, 15)).toEqual([0, 79, 7, 255])
    expect(pixel(window, 16, 15, 15)).toEqual([9, 79, 7, 255])
  })
})

describe("fromWindow", () => {
  it("inverts the window transform", () => {
    const { scale } = resampleWindow(gradient(640, 640), 300.5, 200.25, 640, 256)
    const [x, y] = fromWindow([128, 128], 300.5, 200.25, scale, 256)
    expect(x).toBeCloseTo(300.5)
    expect(y).toBeCloseTo(200.25)
    const [ex, ey] = fromWindow([256, 0], 300.5, 200.25, scale, 256)
    expect(ex).toBeCloseTo(300.5 + 320)
    expect(ey).toBeCloseTo(200.25 - 320)
  })
})

describe("refineSide and upVote", () => {
  it("fits the card's long side at the fill ratio, with a floor", () => {
    expect(refineSide(120, constants)).toBeCloseTo((120 * 88) / 63 / 0.6)
    expect(refineSide(10, constants)).toBe(64)
  })

  it("normalises the summed up vector by the rotation count", () => {
    expect(upVote([0, -4], constants)).toBe(1)
    expect(upVote([0, 0], constants)).toBe(0)
  })
})

describe("searchArts", () => {
  const arts: GalleryArt[] = [
    { id: "1", name: "Forest", set: "fin", collector_number: "292", frame: "tall" },
    { id: "2", name: "Forest", set: "fin", collector_number: "291", frame: "tall" },
    { id: "3", name: "Forest", set: "blb", collector_number: "278", frame: "modern" },
    { id: "4", name: "Forest Bear", set: "m19", collector_number: "12", frame: "modern" },
    { id: "5", name: "Command Tower", set: "fic", collector_number: "301", frame: "modern" },
    { id: "6", name: "Woe Strider", set: "thb", collector_number: "123", frame: "modern" },
    { id: "7", name: "Restless Fortress", set: "woe", collector_number: "258", frame: "modern" },
    { id: "8", name: "Woe Strider", set: "woe", collector_number: "9", frame: "modern" },
  ]

  it("reads a bare set code as part of the name when the card is named that way", () => {
    expect(searchArts(arts, "woe strider").map((art) => art.id)).toEqual(["8", "6"])
    expect(searchArts(arts, "Woe").map((art) => art.id)).toEqual(["7", "8", "6"])
    expect(searchArts(arts, "set:woe strider").map((art) => art.id)).toEqual(["8"])
  })

  it("narrows by set code and orders by collector number", () => {
    expect(searchArts(arts, "forest fin").map((art) => art.id)).toEqual(["2", "1"])
    expect(searchArts(arts, "set:blb forest").map((art) => art.id)).toEqual(["3"])
  })

  it("narrows by collector number", () => {
    expect(searchArts(arts, "forest #278").map((art) => art.id)).toEqual(["3"])
    expect(searchArts(arts, "forest 291").map((art) => art.id)).toEqual(["2"])
  })

  it("matches every name token and ignores case", () => {
    expect(searchArts(arts, "Tower command").map((art) => art.id)).toEqual(["5"])
    expect(searchArts(arts, "bear").map((art) => art.id)).toEqual(["4"])
    expect(searchArts(arts, "   ")).toEqual([])
  })

  it("caps the result list", () => {
    expect(searchArts(arts, "forest", 2)).toHaveLength(2)
  })

  it("searches sibling sets, numbers and languages without adding ranked artwork rows", () => {
    const art: GalleryArt = {
      id: "extended",
      name: "Nettlecyst",
      set: "mh2",
      collector_number: "471",
      frame: "extended",
      printings: [
        { id: "jp", name: "Nettlecyst", set: "mkc", collector_number: "233", lang: "ja" },
        { id: "reprint", name: "Nettlecyst", set: "mkc", collector_number: "233", lang: "en" },
        { id: "regular", name: "Nettlecyst", set: "mh2", collector_number: "231", lang: "en" },
        {
          id: "extended",
          name: "Nettlecyst",
          set: "mh2",
          collector_number: "471",
          lang: "en",
          frame_effects: ["extendedart"],
        },
      ],
    }
    expect(searchArts([art], "Nettlecyst mkc #233").map((p) => p.id)).toEqual(["reprint", "jp"])
    expect(searchArts([art], "Nettlecyst set:mkc lang:ja")).toEqual([
      { ...art.printings![0], frame: "extended" },
    ])
    expect(searchArts([art], "Nettlecyst mh2 231").map((p) => p.id)).toEqual(["regular"])
    expect(searchArts([art, art], "Nettlecyst", 20)).toHaveLength(4)
    expect(searchArts([art], "Nettlecyst mkc lang:de")).toEqual([])
  })

  it("finds Revised and Unlimited printings even when the gallery representative is Alpha", () => {
    const art: GalleryArt = {
      id: "alpha",
      name: "Serra Angel",
      set: "lea",
      collector_number: "40",
      frame: "old",
      printings: [
        { id: "alpha", name: "Serra Angel", set: "lea", collector_number: "40", lang: "en" },
        { id: "revised", name: "Serra Angel", set: "3ed", collector_number: "40", lang: "en" },
        { id: "unlimited", name: "Serra Angel", set: "2ed", collector_number: "40", lang: "en" },
      ],
    }
    expect(searchArts([art], "serra 3ed #40").map((p) => p.id)).toEqual(["revised"])
    expect(searchArts([art], "serra set:2ed").map((p) => p.id)).toEqual(["unlimited"])
  })

  it("searches and returns face metadata without collapsing sides", () => {
    const front: GalleryArt = {
      id: "b0a96416-9ee5-4202-a99f-e09db8794567",
      name: "Jadzi, Oracle of Arcavios",
      set: "stx",
      collector_number: "325",
      face: 0,
      lang: "en",
      frame: "modern",
    }
    const back = { ...front, id: `${front.id}-1`, name: "Journey to the Oracle", face: 1 }
    expect(searchArts([front, back], "Journey stx #325")).toEqual([back])
    expect(searchArts([front, back], "Jadzi")).toEqual([front])
  })
})

describe("clickInCrop", () => {
  it("maps the normalized click through the crop origin", () => {
    // 1920×1080 camera, click at (0.9, 0.5) → native (1728, 540); a 640 crop clamped to the
    // right edge starts at 1280, so the click sits 448 px into the crop, not at its centre.
    expect(
      clickInCrop({
        nativeWidth: 1920,
        nativeHeight: 1080,
        cropLeft: 1280,
        cropTop: 220,
        x: 0.9,
        y: 0.5,
      }),
    ).toEqual([448, 320])
  })
})
