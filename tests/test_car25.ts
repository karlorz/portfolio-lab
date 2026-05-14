/**
 * Tests for CAR25 Performance Metric
 * 
 * Tests Bandy's safe-f position sizing and CAR25 calculation
 * with Monte Carlo block-bootstrap simulation.
 */

import {
  calculateSafeF,
  calculateCAR25,
  calculateMarketCorrelation,
  analyzeCAR25,
  pricesToReturns,
  simulateDailyReturnsFromStats,
} from '../src/backtest/car25';
import type { PriceData } from '../src/backtest/engine';

describe('CAR25 Core Functions', () => {
  // Generate synthetic daily returns for testing
  function generateSyntheticReturns(
    days: number,
    meanReturn: number,
    volatility: number,
    seed: number = 42,
  ): number[] {
    // Simple seeded RNG
    let s = seed;
    const rng = () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };

    const returns: number[] = [];
    const dailyMean = meanReturn / 252;
    const dailyVol = volatility / Math.sqrt(252);

    for (let i = 0; i < days; i++) {
      // Box-Muller
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      returns.push(dailyMean + z * dailyVol);
    }
    return returns;
  }

  describe('calculateSafeF', () => {
    it('should return a safe-f between 0.01 and 4.0', () => {
      const returns = generateSyntheticReturns(500, 0.08, 0.12);
      const result = calculateSafeF(returns, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      expect(result.safeF).toBeGreaterThanOrEqual(0.01);
      expect(result.safeF).toBeLessThanOrEqual(4.0);
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.iterations).toBeLessThanOrEqual(20);
    });

    it('should respect risk tolerance - drawdown near tolerance', () => {
      const returns = generateSyntheticReturns(500, 0.10, 0.15);
      const tolerance = 0.15;
      const result = calculateSafeF(returns, {
        riskTolerance: tolerance,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      // 95th %ile drawdown should be close to tolerance (within 2%)
      expect(Math.abs(result.drawdown95 - tolerance)).toBeLessThan(0.02);
    });

    it('should produce higher safe-f for lower volatility', () => {
      const returnsLowVol = generateSyntheticReturns(500, 0.10, 0.10);
      const returnsHighVol = generateSyntheticReturns(500, 0.10, 0.25);

      const resultLow = calculateSafeF(returnsLowVol, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 300,
        seed: 42,
      });

      const resultHigh = calculateSafeF(returnsHighVol, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 300,
        seed: 42,
      });

      // Lower volatility should allow higher position size
      expect(resultLow.safeF).toBeGreaterThan(resultHigh.safeF);
    });

    it('should converge within max iterations', () => {
      const returns = generateSyntheticReturns(500, 0.08, 0.15);
      const result = calculateSafeF(returns, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      expect(result.converged || result.iterations === 20).toBe(true);
    });
  });

  describe('calculateCAR25', () => {
    it('should return valid CAR25 values', () => {
      const returns = generateSyntheticReturns(500, 0.10, 0.15);
      const safeF = calculateSafeF(returns, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      const car25 = calculateCAR25(returns, safeF.safeF, {
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      // CAR values should be reasonable (between -50% and +100%)
      expect(car25.car25).toBeGreaterThan(-0.5);
      expect(car25.car25).toBeLessThan(1.0);
      expect(car25.car50).toBeGreaterThan(-0.5);
      expect(car25.car75).toBeGreaterThan(car25.car25);
    });

    it('should have car25 < car50 < car75', () => {
      const returns = generateSyntheticReturns(500, 0.10, 0.15);
      const safeF = calculateSafeF(returns, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      const car25 = calculateCAR25(returns, safeF.safeF, {
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      expect(car25.car25).toBeLessThanOrEqual(car25.car50);
      expect(car25.car50).toBeLessThanOrEqual(car25.car75);
    });

    it('should return TWR values greater than 0', () => {
      const returns = generateSyntheticReturns(500, 0.10, 0.15);
      const safeF = calculateSafeF(returns, {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      const car25 = calculateCAR25(returns, safeF.safeF, {
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      expect(car25.twr25).toBeGreaterThan(0);
      expect(car25.twr50).toBeGreaterThan(0);
      expect(car25.twr75).toBeGreaterThan(0);
    });
  });

  describe('calculateMarketCorrelation', () => {
    it('should return correlation between -1 and 1', () => {
      const portfolioReturns = generateSyntheticReturns(100, 0.10, 0.15);
      const benchmarkReturns = generateSyntheticReturns(100, 0.12, 0.15);

      const result = calculateMarketCorrelation(portfolioReturns, benchmarkReturns);

      expect(result.correlation).toBeGreaterThanOrEqual(-1);
      expect(result.correlation).toBeLessThanOrEqual(1);
      expect(result.commonDays).toBe(100);
    });

    it('should classify correlation correctly', () => {
      // High correlation (>0.7)
      const highCorr1 = Array(100).fill(0.001);
      const highCorr2 = Array(100).fill(0.001);
      const highResult = calculateMarketCorrelation(highCorr1, highCorr2);
      expect(highResult.classification).toBe('high');

      // Low correlation (random independent)
      const low1 = generateSyntheticReturns(100, 0.10, 0.15, 1);
      const low2 = generateSyntheticReturns(100, 0.10, 0.15, 999);
      const lowResult = calculateMarketCorrelation(low1, low2);
      expect(lowResult.classification).toBe('low');
    });

    it('should handle different length arrays', () => {
      const pReturns = generateSyntheticReturns(150, 0.10, 0.15);
      const bReturns = generateSyntheticReturns(100, 0.12, 0.15);

      const result = calculateMarketCorrelation(pReturns, bReturns);

      expect(result.commonDays).toBe(100);
    });
  });

  describe('analyzeCAR25', () => {
    it('should return complete CAR25 analysis', () => {
      const returns = generateSyntheticReturns(500, 0.10, 0.15);
      const benchmarkReturns = generateSyntheticReturns(500, 0.12, 0.16);

      const result = analyzeCAR25(returns, benchmarkReturns, 'Test Portfolio', {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      expect(result.portfolio).toBe('Test Portfolio');
      expect(result.safeF.safeF).toBeGreaterThan(0);
      expect(result.car25.car25).not.toBeNaN();
      expect(result.correlation.correlation).toBeGreaterThanOrEqual(-1);
      expect(result.inputDays).toBe(500);
    });

    it('should handle null benchmark', () => {
      const returns = generateSyntheticReturns(500, 0.10, 0.15);

      const result = analyzeCAR25(returns, null, 'Test Portfolio', {
        riskTolerance: 0.20,
        horizonYears: 2,
        simulations: 500,
        seed: 42,
      });

      expect(result.correlation.correlation).toBe(0);
      expect(result.correlation.classification).toBe('low');
    });
  });

  describe('pricesToReturns', () => {
    it('should convert price data to daily returns', () => {
      const priceData: PriceData[] = [
        { date: '2024-01-01', symbol: 'SPY', price: 100 },
        { date: '2024-01-02', symbol: 'SPY', price: 101 },
        { date: '2024-01-03', symbol: 'SPY', price: 99 },
        { date: '2024-01-04', symbol: 'SPY', price: 102 },
      ];

      const returns = pricesToReturns(priceData, 'SPY');

      expect(returns.length).toBe(3);
      expect(returns[0]).toBeCloseTo(0.01, 4); // (101-100)/100
      expect(returns[1]).toBeCloseTo(-0.0198, 2); // (99-101)/101
      expect(returns[2]).toBeCloseTo(0.0303, 2); // (102-99)/99
    });

    it('should filter by symbol', () => {
      const priceData: PriceData[] = [
        { date: '2024-01-01', symbol: 'SPY', price: 100 },
        { date: '2024-01-02', symbol: 'SPY', price: 101 },
        { date: '2024-01-01', symbol: 'GLD', price: 150 },
        { date: '2024-01-02', symbol: 'GLD', price: 151 },
      ];

      const spyReturns = pricesToReturns(priceData, 'SPY');
      const gldReturns = pricesToReturns(priceData, 'GLD');

      expect(spyReturns.length).toBe(1);
      expect(gldReturns.length).toBe(1);
      expect(spyReturns[0]).toBeCloseTo(0.01, 4);
      expect(gldReturns[0]).toBeCloseTo(0.00667, 3);
    });

    it('should sort by date', () => {
      const priceData: PriceData[] = [
        { date: '2024-01-03', symbol: 'SPY', price: 102 },
        { date: '2024-01-01', symbol: 'SPY', price: 100 },
        { date: '2024-01-02', symbol: 'SPY', price: 101 },
      ];

      const returns = pricesToReturns(priceData, 'SPY');

      expect(returns.length).toBe(2);
      expect(returns[0]).toBeCloseTo(0.01, 4); // Jan 1 to Jan 2
    });
  });

  describe('simulateDailyReturnsFromStats', () => {
    it('should generate returns with approximate target statistics', () => {
      const targetCAGR = 0.10;
      const targetVol = 0.15;
      const days = 252;

      const returns = simulateDailyReturnsFromStats(targetCAGR, targetVol, days, 42);

      expect(returns.length).toBe(days);

      // Check approximate mean (should be near daily CAGR)
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const dailyTarget = targetCAGR / 252;
      expect(Math.abs(mean - dailyTarget)).toBeLessThan(0.01);

      // Check all returns are finite
      expect(returns.every(r => Number.isFinite(r))).toBe(true);
    });

    it('should be deterministic with same seed', () => {
      const returns1 = simulateDailyReturnsFromStats(0.10, 0.15, 100, 42);
      const returns2 = simulateDailyReturnsFromStats(0.10, 0.15, 100, 42);

      expect(returns1).toEqual(returns2);
    });

    it('should differ with different seeds', () => {
      const returns1 = simulateDailyReturnsFromStats(0.10, 0.15, 100, 42);
      const returns2 = simulateDailyReturnsFromStats(0.10, 0.15, 100, 999);

      expect(returns1).not.toEqual(returns2);
    });
  });
});

describe('CAR25 Integration Tests', () => {
  it('should handle realistic portfolio data', () => {
    // Simulate ~2 years of SPY-like returns (10% CAGR, 16% vol)
    const returns: number[] = [];
    let s = 42;
    const rng = () => {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };

    const dailyMean = 0.10 / 252;
    const dailyVol = 0.16 / Math.sqrt(252);

    for (let i = 0; i < 500; i++) {
      const u1 = rng();
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      returns.push(dailyMean + z * dailyVol);
    }

    const result = analyzeCAR25(returns, returns, 'SPY Benchmark Test', {
      riskTolerance: 0.20,
      horizonYears: 2,
      simulations: 1000,
      seed: 42,
    });

    // Portfolio correlated with itself should have correlation ~1
    expect(result.correlation.correlation).toBeGreaterThan(0.9);
    expect(result.correlation.classification).toBe('high');

    // CAR values should be in reasonable range
    expect(result.car25.car25).toBeGreaterThan(-0.3);
    expect(result.car25.car75).toBeLessThan(0.5);
  });

  it('should produce lower safe-f for higher risk tolerance', () => {
    const returns = generateSyntheticReturns(500, 0.10, 0.20, 42);

    const conservative = calculateSafeF(returns, {
      riskTolerance: 0.10, // 10% max DD
      horizonYears: 2,
      simulations: 500,
      seed: 42,
    });

    const aggressive = calculateSafeF(returns, {
      riskTolerance: 0.30, // 30% max DD
      horizonYears: 2,
      simulations: 500,
      seed: 42,
    });

    // Higher tolerance should allow higher position size
    expect(aggressive.safeF).toBeGreaterThan(conservative.safeF);
  });

  it('should handle edge case of very short return series', () => {
    const returns = generateSyntheticReturns(50, 0.10, 0.15, 42);

    const result = calculateSafeF(returns, {
      riskTolerance: 0.20,
      horizonYears: 1, // Shorter horizon
      simulations: 200,
      blockSize: 5, // Smaller blocks
      seed: 42,
    });

    expect(result.safeF).toBeGreaterThan(0);
    expect(result.iterations).toBeGreaterThan(0);
  });
});
