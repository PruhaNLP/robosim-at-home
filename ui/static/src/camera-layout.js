export const CAMERA_GAP = 8;
export const CAMERA_PAD = 16;
export const CAMERA_MIN = 96;

export function cameraLayouts(count) {
  const cell = (column, row) => [column, row];
  if (count <= 1) return [{ cols: 1, rows: 1, positions: [cell(0, 0)] }];
  if (count === 2) {
    return [
      { cols: 2, rows: 1, positions: [cell(0, 0), cell(1, 0)] },
      { cols: 1, rows: 2, positions: [cell(0, 0), cell(0, 1)] },
    ];
  }
  if (count === 3) {
    return [
      { cols: 3, rows: 1, positions: [cell(0, 0), cell(1, 0), cell(2, 0)] },
      { cols: 1, rows: 3, positions: [cell(0, 0), cell(0, 1), cell(0, 2)] },
    ];
  }
  if (count === 4) {
    return [
      { cols: 2, rows: 2, positions: [cell(0, 0), cell(1, 0), cell(0, 1), cell(1, 1)] },
      { cols: 4, rows: 1, positions: [cell(0, 0), cell(1, 0), cell(2, 0), cell(3, 0)] },
      { cols: 1, rows: 4, positions: [cell(0, 0), cell(0, 1), cell(0, 2), cell(0, 3)] },
    ];
  }
  return [
    {
      cols: 3,
      rows: 2,
      positions: [cell(0, 0), cell(1, 0), cell(2, 0), cell(0.5, 1), cell(1.5, 1)],
    },
    {
      cols: 3,
      rows: 2,
      positions: [cell(0.5, 0), cell(1.5, 0), cell(0, 1), cell(1, 1), cell(2, 1)],
    },
    {
      cols: 5,
      rows: 1,
      positions: [cell(0, 0), cell(1, 0), cell(2, 0), cell(3, 0), cell(4, 0)],
    },
    {
      cols: 1,
      rows: 5,
      positions: [cell(0, 0), cell(0, 1), cell(0, 2), cell(0, 3), cell(0, 4)],
    },
  ];
}

export function pickCameraLayout(count, width, height) {
  let best = null;
  for (const layout of cameraLayouts(count)) {
    const tile = Math.min(
      (width - CAMERA_GAP * (layout.cols - 1)) / layout.cols,
      (height - CAMERA_GAP * (layout.rows - 1)) / layout.rows
    );
    if (tile < CAMERA_MIN) continue;
    const filled = count * tile * tile;
    if (!best || filled > best.filled) best = { ...layout, tile, filled };
  }
  return best;
}
