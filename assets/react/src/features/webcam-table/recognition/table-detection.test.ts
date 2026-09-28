import { expect, it } from "vite-plus/test"
import { cropImage, dedupeTableDetections, tileBoxes } from "./table-detection"

it("covers a frame with a 2x2 grid that overlaps each neighboring tile by 20%", () => {
  expect(tileBoxes(100, 80, 2, 0.2)).toEqual([
    { left: 0, top: 0, right: 56, bottom: 44 },
    { left: 44, top: 0, right: 100, bottom: 44 },
    { left: 0, top: 36, right: 56, bottom: 80 },
    { left: 44, top: 36, right: 100, bottom: 80 },
  ])
})

it("crops RGBA tile pixels without resampling them", () => {
  const image = {
    data: Uint8ClampedArray.from({ length: 4 * 3 * 4 }, (_, index) => index),
    width: 4,
    height: 3,
  }
  expect(cropImage(image, { left: 1, top: 1, right: 3, bottom: 3 })).toEqual({
    data: Uint8ClampedArray.from([...image.data.subarray(20, 28), ...image.data.subarray(36, 44)]),
    width: 2,
    height: 2,
  })
})

it("score-weights overlapping quads instead of retaining only the strongest one", () => {
  const quad = (offset: number) =>
    [
      [offset, offset],
      [offset + 10, offset],
      [offset + 10, offset + 10],
      [offset, offset + 10],
    ] as const
  expect(
    dedupeTableDetections(
      [
        { quad: quad(0), confidence: 0.8 },
        { quad: quad(2), confidence: 0.4 },
        { quad: quad(30), confidence: 0.9 },
      ],
      0.4,
    ),
  ).toEqual([
    { quad: quad(30), confidence: 0.9 },
    { quad: quad(2 / 3), confidence: 0.8 },
  ])
})
