/**
 * Largest-Triangle-Three-Buckets (LTTB) downsampling algorithm.
 *
 * Reduces a time-series to `threshold` points while preserving visual shape.
 * Ideal for charting 5000+ data points with Recharts — reduces render time
 * without visible quality loss.
 *
 * Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual
 * Representation" (2013), https://skemman.is/handle/1946/15343
 */

export interface DataPoint {
  [key: string]: unknown;
}

/**
 * Calculate the area of the triangle formed by three points.
 */
function triangleArea(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number },
): number {
  return Math.abs(
    (a.x - c.x) * (b.y - a.y) - (a.x - b.x) * (c.y - a.y),
  ) / 2;
}

/**
 * Downsample a data array using LTTB.
 *
 * @param data - Full dataset (must be sorted by xKey)
 * @param threshold - Target number of points (must be >= 2)
 * @param xKey - Key for x-axis values (e.g., "date")
 * @param yKey - Key for y-axis values (e.g., "value")
 * @returns Downsampled array with original data shape preserved
 */
/**
 * Convert an x-axis value to a numeric representation.
 * ISO date strings become timestamps; numbers pass through.
 */
function toNumber(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'string') {
    const d = new Date(v).getTime();
    return isNaN(d) ? NaN : d;
  }
  return Number(v);
}

function readPointValue<T>(point: T, key: string): unknown {
  return (point as Record<string, unknown>)[key];
}

export function downsampleLTTB<T>(
  data: T[],
  threshold: number,
  xKey: string,
  yKey: string,
): T[] {
  if (data.length <= threshold || threshold < 3) {
    return data;
  }

  const sampled: T[] = [];
  // Always include first and last points
  sampled.push(data[0]);

  // Bucket size (first and last points are fixed)
  const bucketSize = (data.length - 2) / (threshold - 2);

  let aIndex = 0; // Previously selected point

  for (let i = 1; i < threshold - 1; i++) {
    // Calculate next bucket boundaries
    const avgStart = Math.floor((i + 0) * bucketSize) + 1;
    const avgEnd = Math.floor((i + 1) * bucketSize) + 1;
    const nextBucketStart = Math.min(avgEnd, data.length - 1);

    // Calculate average point in next bucket
    let avgX = 0;
    let avgY = 0;
    let count = 0;
    for (let j = avgStart; j < avgEnd && j < data.length; j++) {
      const x = toNumber(readPointValue(data[j], xKey));
      const y = Number(readPointValue(data[j], yKey));
      if (!isNaN(x) && !isNaN(y)) {
        avgX += x;
        avgY += y;
        count++;
      }
    }
    if (count === 0) continue;
    avgX /= count;
    avgY /= count;

    // Find point in current bucket that forms largest triangle
    const bucketStart = Math.floor((i - 1) * bucketSize) + 1;
    const bucketEnd = avgStart;

    const pointA = {
      x: toNumber(readPointValue(data[aIndex], xKey)),
      y: Number(readPointValue(data[aIndex], yKey)),
    };

    let maxArea = -1;
    let maxIndex = bucketStart;

    for (let j = bucketStart; j < bucketEnd && j < data.length; j++) {
      const pointB = {
        x: toNumber(readPointValue(data[j], xKey)),
        y: Number(readPointValue(data[j], yKey)),
      };
      const area = triangleArea(pointA, pointB, { x: avgX, y: avgY });
      if (area > maxArea) {
        maxArea = area;
        maxIndex = j;
      }
    }

    sampled.push(data[maxIndex]);
    aIndex = maxIndex;
  }

  // Always include last point
  sampled.push(data[data.length - 1]);

  return sampled;
}

/**
 * Auto-downsample: only apply LTTB if data exceeds a size threshold.
 * Returns original data if small enough.
 *
 * @param data - Full dataset
 * @param threshold - Target number of points (default 500)
 * @param xKey - Key for x-axis values
 * @param yKey - Key for y-axis values
 * @param minSize - Minimum data size before downsampling (default 1000)
 */
export function autoDownsample<T>(
  data: T[],
  threshold: number = 500,
  xKey: string = "date",
  yKey: string = "value",
  minSize: number = 1000,
): T[] {
  if (data.length < minSize) {
    return data;
  }
  return downsampleLTTB(data, threshold, xKey, yKey);
}
