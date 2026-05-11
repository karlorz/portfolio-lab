/**
 * Purged Cross-Validation — Implements purged k-fold CV with embargo periods
 * Prevents data leakage in financial time series by removing overlapping periods
 * Reference: Lopez de Prado, "Advances in Financial Machine Learning" (2018)
 */

export interface PurgedKFoldConfig {
  nSplits: number;
  embargoDays: number;
  minTrainPeriods: number;
}

export interface PurgedSplit {
  train: Date[];
  test: Date[];
}

export class PurgedKFold {
  private config: PurgedKFoldConfig;

  constructor(config: Partial<PurgedKFoldConfig> = {}) {
    this.config = {
      nSplits: config.nSplits ?? 5,
      embargoDays: config.embargoDays ?? 20,
      minTrainPeriods: config.minTrainPeriods ?? 252, // 1 year of trading days
    };
  }

  /**
   * Create purged splits from sorted date array
   * Ensures no overlap between train and test with embargo period
   */
  createSplits(dates: Date[]): PurgedSplit[] {
    if (dates.length < this.config.minTrainPeriods * 2) {
      throw new Error(
        `Insufficient data: ${dates.length} dates, need at least ${this.config.minTrainPeriods * 2}`
      );
    }

    // Sort dates
    const sortedDates = [...dates].sort((a, b) => a.getTime() - b.getTime());
    const n = sortedDates.length;
    
    // Calculate fold boundaries
    const foldSize = Math.floor(n / this.config.nSplits);
    const splits: PurgedSplit[] = [];

    for (let i = 0; i < this.config.nSplits; i++) {
      const testStart = i * foldSize;
      const testEnd = Math.min((i + 1) * foldSize, n);
      
      // Embargo period before and after test set
      const embargoStart = Math.max(0, testStart - this.config.embargoDays);
      const embargoEnd = Math.min(n, testEnd + this.config.embargoDays);
      
      // Training data excludes embargo period around test
      // For simplicity: use all data before embargoStart
      // In practice: can also use data after embargoEnd for growing window
      if (embargoStart < this.config.minTrainPeriods) {
        // Skip if not enough training data
        continue;
      }

      const trainDates = sortedDates.slice(0, embargoStart);
      const testDates = sortedDates.slice(testStart, testEnd);

      splits.push({
        train: trainDates,
        test: testDates,
      });
    }

    return splits;
  }

  /**
   * Create growing window splits (purged)
   * Train on [0, embargoStart), test on [testStart, testEnd)
   * Then expand train to include [embargoEnd, ...)
   */
  createGrowingWindowSplits(dates: Date[]): PurgedSplit[] {
    const sortedDates = [...dates].sort((a, b) => a.getTime() - b.getTime());
    const n = sortedDates.length;
    const splits: PurgedSplit[] = [];

    for (let i = 0; i < this.config.nSplits; i++) {
      const testStart = Math.floor(n * (i / this.config.nSplits));
      const testEnd = Math.floor(n * ((i + 1) / this.config.nSplits));
      
      // Embargo periods
      const embargoStart = Math.max(0, testStart - this.config.embargoDays);
      const embargoEnd = Math.min(n, testEnd + this.config.embargoDays);

      if (embargoStart < this.config.minTrainPeriods) {
        continue;
      }

      // Train excludes embargo period
      const trainDates = sortedDates.slice(0, embargoStart);
      const testDates = sortedDates.slice(testStart, testEnd);

      splits.push({ train: trainDates, test: testDates });
    }

    return splits;
  }

  /**
   * Create anchored walk-forward splits
   * Train always starts at 0, test moves forward with embargo
   */
  createAnchoredSplits(dates: Date[]): PurgedSplit[] {
    const sortedDates = [...dates].sort((a, b) => a.getTime() - b.getTime());
    const n = sortedDates.length;
    const splits: PurgedSplit[] = [];

    const minTrainSize = this.config.minTrainPeriods;

    for (let i = 1; i <= this.config.nSplits; i++) {
      const testEnd = Math.floor(n * (i / this.config.nSplits));
      const testStart = Math.floor(n * ((i - 1) / this.config.nSplits));
      const embargoEnd = Math.min(n, testEnd + this.config.embargoDays);

      // Train from start to testStart minus embargo
      const trainEnd = Math.max(0, testStart - this.config.embargoDays);

      if (trainEnd < minTrainSize) {
        continue;
      }

      const trainDates = sortedDates.slice(0, trainEnd);
      const testDates = sortedDates.slice(testStart, testEnd);

      splits.push({ train: trainDates, test: testDates });
    }

    return splits;
  }
}

/**
 * Calculate embargo period based on data frequency
 */
export function estimateEmbargoPeriod(
  dates: Date[],
  strategy: 'daily' | 'weekly' | 'monthly' = 'daily'
): number {
  if (dates.length < 2) return 20;

  // Calculate average days between observations
  const diffs: number[] = [];
  for (let i = 1; i < Math.min(dates.length, 100); i++) {
    const diff = (dates[i].getTime() - dates[i - 1].getTime()) / (1000 * 60 * 60 * 24);
    diffs.push(diff);
  }
  const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;

  // Embargo based on strategy and frequency
  if (strategy === 'daily') {
    return Math.max(20, Math.round(20 / avgDiff)); // ~20 trading days
  } else if (strategy === 'weekly') {
    return Math.max(4, Math.round(4 / avgDiff)); // ~4 weeks
  } else {
    return Math.max(3, Math.round(3 / avgDiff)); // ~3 months
  }
}

/**
 * Validate that splits have no leakage
 */
export function validateSplits(splits: PurgedSplit[]): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  for (let i = 0; i < splits.length; i++) {
    const split = splits[i];

    // Check no dates in both train and test
    const trainSet = new Set(split.train.map(d => d.getTime()));
    const overlap = split.test.filter(d => trainSet.has(d.getTime()));

    if (overlap.length > 0) {
      errors.push(`Split ${i}: ${overlap.length} dates overlap between train and test`);
    }

    // Check chronological ordering
    const lastTrain = Math.max(...split.train.map(d => d.getTime()));
    const firstTest = Math.min(...split.test.map(d => d.getTime()));

    if (lastTrain >= firstTest) {
      errors.push(`Split ${i}: Train data overlaps or follows test data`);
    }

    // Check sufficient data
    if (split.train.length < 100) {
      errors.push(`Split ${i}: Insufficient training data (${split.train.length} dates)`);
    }
    if (split.test.length < 20) {
      errors.push(`Split ${i}: Insufficient test data (${split.test.length} dates)`);
    }
  }

  return { valid: errors.length === 0, errors };
}
