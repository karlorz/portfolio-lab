/**
 * Portfolio Backtest Engine
 * Simulates portfolio performance with rebalancing
 */

export interface AssetAllocation {
  [symbol: string]: number; // percentage (0-1)
}

export interface PortfolioConfig {
  name: string;
  allocation: AssetAllocation;
  rebalanceFrequency: 'monthly' | 'quarterly' | 'annual' | 'none';
  trendFollowing?: {
    enabled: boolean;
    lookbackMonths: number;
    movingAverageMonths: number;
  };
  volatilityTarget?: {
    enabled: boolean;
    targetVol: number; // annualized, e.g. 0.12 for 12%
  };
}

export interface PriceData {
  date: string;
  symbol: string;
  price: number;
  dividend?: number;
}

export interface BacktestResult {
  dates: string[];
  portfolioValues: number[];
  returns: number[];
  drawdowns: number[];
  holdings: Array<{ [symbol: string]: number }>;
  trades: Trade[];
}

export interface Trade {
  date: string;
  type: 'rebalance' | 'trend';
  from?: string;
  to?: string;
  amount: number;
  newHoldings?: { [symbol: string]: number };
}

export interface PerformanceMetrics {
  cagr: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  calmarRatio: number;
  sortinoRatio: number;
  positiveMonths: number;
  totalReturn: number;
}

export class BacktestEngine {
  private priceData: Map<string, Map<string, number>> = new Map();
  private dividendData: Map<string, Map<string, number>> = new Map();

  loadData(prices: PriceData[]) {
    for (const p of prices) {
      if (!this.priceData.has(p.symbol)) {
        this.priceData.set(p.symbol, new Map());
        this.dividendData.set(p.symbol, new Map());
      }
      this.priceData.get(p.symbol)!.set(p.date, p.price);
      if (p.dividend) {
        this.dividendData.get(p.symbol)!.set(p.date, p.dividend);
      }
    }
  }

  runBacktest(
    config: PortfolioConfig,
    startDate: string,
    endDate: string,
    initialValue: number = 10000
  ): BacktestResult {
    const dates = this.getTradingDays(startDate, endDate);
    const symbols = Object.keys(config.allocation);
    
    if (dates.length === 0) {
      return { dates: [], portfolioValues: [], returns: [], drawdowns: [], holdings: [], trades: [] };
    }

    let holdings: { [symbol: string]: number } = {};

    // Initialize holdings
    for (const symbol of symbols) {
      const firstDate = dates[0];
      const price = this.getPrice(symbol, firstDate) || 100; // Fallback if no data
      holdings[symbol] = (initialValue * config.allocation[symbol]) / price;
    }

    const result: BacktestResult = {
      dates: [],
      portfolioValues: [],
      returns: [],
      drawdowns: [],
      holdings: [],
      trades: [],
    };

    let peakValue = initialValue;
    let lastRebalanceIndex = 0;
    // Volatility targeting state
    const recentReturns: number[] = [];
    const volLookback = 60; // ~3 months of daily returns for vol estimate

    for (let i = 0; i < dates.length; i++) {
      const date = dates[i];

      // Apply dividends
      for (const symbol of symbols) {
        const dividend = this.dividendData.get(symbol)?.get(date);
        if (dividend && holdings[symbol]) {
          const price = this.getPrice(symbol, date) || 100;
          const sharesToBuy = (holdings[symbol] * dividend) / price;
          holdings[symbol] += sharesToBuy;
        }
      }

      // Calculate current value
      let currentValue = 0;
      for (const symbol of symbols) {
        currentValue += holdings[symbol] * (this.getPrice(symbol, date) || 100);
      }

      // Track peak and drawdown
      if (currentValue > peakValue) peakValue = currentValue;
      const drawdown = (currentValue - peakValue) / peakValue;

      // Record data
      result.dates.push(date);
      result.portfolioValues.push(currentValue);
      result.drawdowns.push(drawdown);
      result.holdings.push({ ...holdings });

      if (i > 0) {
        const dailyReturn = (currentValue - result.portfolioValues[i - 1]) / result.portfolioValues[i - 1];
        result.returns.push(dailyReturn);
      } else {
        result.returns.push(0);
      }

      // Check rebalance
      const shouldRebalance = this.shouldRebalance(
        i,
        lastRebalanceIndex,
        config.rebalanceFrequency,
        dates
      );

      // Volatility targeting: scale exposure based on realized vol
      let volTargetScale = 1.0;
      if (config.volatilityTarget?.enabled && i > volLookback) {
        recentReturns.push(result.returns[result.returns.length - 1]);
        if (recentReturns.length > volLookback) recentReturns.shift();
        const realizedVol = this.estimateAnnualVol(recentReturns);
        if (realizedVol > 0) {
          volTargetScale = Math.min(1.5, Math.max(0.3, config.volatilityTarget.targetVol / realizedVol));
        }
      } else if (i <= volLookback) {
        recentReturns.push(result.returns[result.returns.length - 1]);
        if (recentReturns.length > volLookback) recentReturns.shift();
      }

      // Check trend following (applies to equity-risky assets)
      let trendSignal: { shouldTrade: boolean; targetAllocation?: { [key: string]: number } } | null = null;
      if (config.trendFollowing?.enabled && symbols.length > 0) {
        const firstSymbol = symbols[0];
        trendSignal = this.checkTrendSignal(firstSymbol, date, config.trendFollowing, config.allocation, symbols);
      }

      if (shouldRebalance || trendSignal?.shouldTrade) {
        let targetAllocation = trendSignal?.targetAllocation || config.allocation;
        // Apply volatility targeting: scale risky assets, park remainder in safest
        if (config.volatilityTarget?.enabled && volTargetScale < 1.0) {
          const safeOrder = ['SHY', 'IEF', 'TLT', 'GLD', 'AGG'];
          const safeSymbol = safeOrder.find(s => symbols.includes(s)) || symbols[symbols.length - 1];
          const scaledAllocation: { [key: string]: number } = {};
          for (const [sym, wt] of Object.entries(targetAllocation)) {
            if (sym === safeSymbol) {
              scaledAllocation[sym] = wt + (1 - volTargetScale) * (1 - (targetAllocation[safeSymbol] || 0));
            } else {
              scaledAllocation[sym] = wt * volTargetScale;
            }
          }
          // Normalize
          const total = Object.values(scaledAllocation).reduce((a, b) => a + b, 0);
          for (const sym of Object.keys(scaledAllocation)) {
            scaledAllocation[sym] /= total;
          }
          targetAllocation = scaledAllocation;
        }
        const allSymbols = trendSignal?.targetAllocation
          ? [...new Set([...symbols, ...Object.keys(targetAllocation)])]
          : symbols;
        const trades = this.rebalance(holdings, currentValue, targetAllocation, date, allSymbols);
        result.trades.push(...trades);
        if (trades.length > 0 && trades[trades.length - 1].newHoldings) {
          holdings = trades[trades.length - 1].newHoldings!;
        }
        lastRebalanceIndex = i;
      }
    }

    return result;
  }

  private estimateAnnualVol(returns: number[]): number {
    if (returns.length < 20) return 0;
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;
    return Math.sqrt(variance) * Math.sqrt(252);
  }

  private getTradingDays(startDate: string, endDate: string): string[] {
    // Use actual dates from loaded price data (handles holidays)
    const allDates = new Set<string>();
    for (const symbolDates of this.priceData.values()) {
      for (const d of symbolDates.keys()) {
        if (d >= startDate && d <= endDate) {
          allDates.add(d);
        }
      }
    }
    return Array.from(allDates).sort();
  }

  private getPrice(symbol: string, date: string): number {
    const symbolData = this.priceData.get(symbol);
    if (!symbolData) return 0;
    const price = symbolData.get(date);
    if (price !== undefined) return price;
    // Fallback: find closest earlier date (handles missing data for some symbols)
    const dates = Array.from(symbolData.keys()).sort();
    let lastPrice = 0;
    for (const d of dates) {
      if (d > date) break;
      lastPrice = symbolData.get(d) || 0;
    }
    return lastPrice;
  }

  private shouldRebalance(
    currentIndex: number,
    lastRebalanceIndex: number,
    frequency: PortfolioConfig['rebalanceFrequency'],
    dates: string[]
  ): boolean {
    if (frequency === 'none') return false;

    const periodsSinceRebalance = currentIndex - lastRebalanceIndex;

    switch (frequency) {
      case 'monthly':
        return periodsSinceRebalance >= 21; // ~21 trading days
      case 'quarterly':
        return periodsSinceRebalance >= 63; // ~63 trading days
      case 'annual':
        return periodsSinceRebalance >= 252; // ~252 trading days
      default:
        return false;
    }
  }

  private checkTrendSignal(
    symbol: string,
    date: string,
    config: { lookbackMonths: number; movingAverageMonths: number },
    baseAllocation: AssetAllocation,
    allSymbols: string[]
  ): { shouldTrade: boolean; targetAllocation?: { [key: string]: number } } | null {
    // Trend following: check if price > moving average for the first (equity) symbol
    const price = this.getPrice(symbol, date);
    if (price === 0) return null;

    const ma = this.calculateMovingAverage(symbol, date, config.movingAverageMonths * 21);

    if (!ma) return null;

    // If price below MA, shift equity risk to safe assets already in portfolio
    if (price < ma) {
      // Redistribute: move the first symbol's weight to the safest available asset
      // Priority: SHY > IEF > TLT > GLD > remaining
      const safeOrder = ['SHY', 'IEF', 'TLT', 'GLD', 'AGG'];
      const safeSymbol = safeOrder.find(s => allSymbols.includes(s)) || allSymbols.find(s => s !== symbol);
      if (!safeSymbol) return { shouldTrade: false };

      const safeAllocation: { [key: string]: number } = {};
      for (const s of allSymbols) {
        safeAllocation[s] = s === safeSymbol
          ? (baseAllocation[s] || 0) + (baseAllocation[symbol] || 0)
          : s === symbol ? 0 : (baseAllocation[s] || 0);
      }
      return {
        shouldTrade: true,
        targetAllocation: safeAllocation,
      };
    }

    return { shouldTrade: false };
  }

  private calculateMovingAverage(symbol: string, date: string, periods: number): number | null {
    const symbolData = this.priceData.get(symbol);
    if (!symbolData) return null;
    
    const dates = Array.from(symbolData.keys()).sort();
    const dateIndex = dates.indexOf(date);
    if (dateIndex < periods) return null;

    let sum = 0;
    for (let i = dateIndex - periods + 1; i <= dateIndex; i++) {
      sum += this.getPrice(symbol, dates[i]);
    }
    return sum / periods;
  }

  private rebalance(
    currentHoldings: { [symbol: string]: number },
    totalValue: number,
    targetAllocation: AssetAllocation,
    date: string,
    allSymbols: string[]
  ): Trade[] {
    const trades: Trade[] = [];
    const newHoldings: { [symbol: string]: number } = {};

    // Initialize all symbols to 0
    for (const symbol of allSymbols) {
      newHoldings[symbol] = 0;
    }

    for (const [symbol, weight] of Object.entries(targetAllocation)) {
      const targetValue = totalValue * weight;
      const price = this.getPrice(symbol, date) || 100;
      newHoldings[symbol] = targetValue / price;

      const oldShares = currentHoldings[symbol] || 0;
      const diff = newHoldings[symbol] - oldShares;

      if (Math.abs(diff * price) > 1) { // Min $1 trade
        trades.push({
          date,
          type: 'rebalance',
          from: diff < 0 ? symbol : undefined,
          to: diff > 0 ? symbol : undefined,
          amount: Math.abs(diff * price),
        });
      }
    }

    if (trades.length > 0) {
      trades.push({ date, type: 'rebalance', amount: 0, newHoldings });
    }

    return trades;
  }

  calculateMetrics(result: BacktestResult, riskFreeRate: number = 0.02): PerformanceMetrics {
    const values = result.portfolioValues;
    const returns = result.returns.slice(1); // Skip first 0 return
    
    if (values.length < 2 || returns.length === 0) {
      return {
        cagr: 0,
        volatility: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        calmarRatio: 0,
        sortinoRatio: 0,
        positiveMonths: 0,
        totalReturn: 0,
      };
    }

    const totalReturn = (values[values.length - 1] - values[0]) / values[0];
    const years = values.length / 252;
    const cagr = Math.pow(1 + totalReturn, 1 / years) - 1;

    const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(252); // Annualize

    const excessReturns = returns.map(r => r - riskFreeRate / 252);
    const meanExcess = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;
    const sharpeRatio = variance > 0 ? (meanExcess / Math.sqrt(variance)) * Math.sqrt(252) : 0;

    const maxDrawdown = Math.min(...result.drawdowns);
    const calmarRatio = maxDrawdown !== 0 ? cagr / Math.abs(maxDrawdown) : 0;

    // Sortino (downside deviation)
    const downsideReturns = returns.filter(r => r < 0);
    const downsideVariance = downsideReturns.length > 0
      ? downsideReturns.reduce((sum, r) => sum + r * r, 0) / downsideReturns.length
      : 0;
    const sortinoRatio = downsideVariance > 0
      ? (cagr - riskFreeRate) / (Math.sqrt(downsideVariance) * Math.sqrt(252))
      : 0;

    const positiveMonths = returns.filter(r => r > 0).length;

    return {
      cagr,
      volatility,
      sharpeRatio,
      maxDrawdown,
      calmarRatio,
      sortinoRatio,
      positiveMonths,
      totalReturn,
    };
  }
}

// Predefined portfolios based on research
export const PORTFOLIOS: PortfolioConfig[] = [
  {
    name: 'SPY (S&P 500)',
    allocation: { 'SPY': 1 },
    rebalanceFrequency: 'none',
  },
  {
    name: 'QQQ (Nasdaq-100)',
    allocation: { 'QQQ': 1 },
    rebalanceFrequency: 'none',
  },
  {
    name: '60/40 Portfolio',
    allocation: { 'SPY': 0.6, 'AGG': 0.4 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'All Weather (Dalio)',
    allocation: {
      'VTI': 0.30,
      'TLT': 0.40,
      'IEF': 0.15,
      'GLD': 0.075,
      'DBC': 0.075,
    },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'Golden Butterfly',
    allocation: {
      'VTI': 0.20,
      'VBR': 0.20,
      'TLT': 0.20,
      'SHY': 0.20,
      'GLD': 0.20,
    },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'Golden Butterfly + Trend',
    allocation: {
      'VTI': 0.20,
      'VBR': 0.20,
      'TLT': 0.20,
      'SHY': 0.20,
      'GLD': 0.20,
    },
    rebalanceFrequency: 'monthly',
    trendFollowing: {
      enabled: true,
      lookbackMonths: 12,
      movingAverageMonths: 10,
    },
  },
  {
    name: 'SPY/GLD 55/45',
    allocation: { 'SPY': 0.55, 'GLD': 0.45 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/TLT 58/32/10',
    allocation: { 'SPY': 0.58, 'GLD': 0.32, 'TLT': 0.10 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/TLT 50/35/15 ★',
    allocation: { 'SPY': 0.50, 'GLD': 0.35, 'TLT': 0.15 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/TLT 50/40/10',
    allocation: { 'SPY': 0.50, 'GLD': 0.40, 'TLT': 0.10 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD/IEF 50/35/15',
    allocation: { 'SPY': 0.50, 'GLD': 0.35, 'IEF': 0.15 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/GLD 55/45 +Trend',
    allocation: { 'SPY': 0.55, 'GLD': 0.45 },
    rebalanceFrequency: 'monthly',
    trendFollowing: {
      enabled: true,
      lookbackMonths: 12,
      movingAverageMonths: 10,
    },
  },
  {
    name: 'SPY/GLD/TLT 50/35/15 +Trend',
    allocation: { 'SPY': 0.50, 'GLD': 0.35, 'TLT': 0.15 },
    rebalanceFrequency: 'monthly',
    trendFollowing: {
      enabled: true,
      lookbackMonths: 12,
      movingAverageMonths: 10,
    },
  },
  {
    name: 'SPY/GLD/TLT 50/35/15 +VolTarget',
    allocation: { 'SPY': 0.50, 'GLD': 0.35, 'TLT': 0.15 },
    rebalanceFrequency: 'quarterly',
    volatilityTarget: { enabled: true, targetVol: 0.12 },
  },
  {
    name: 'SPY/GLD/TLT 46/38/16 ★★',
    allocation: { 'SPY': 0.46, 'GLD': 0.38, 'TLT': 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'SPY/EFA/GLD/TLT 36/10/38/16',
    allocation: { 'SPY': 0.36, 'EFA': 0.10, 'GLD': 0.38, 'TLT': 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'Momentum Tilt SPY/MTUM/GLD/TLT 30/20/34/16',
    allocation: { 'SPY': 0.30, 'MTUM': 0.20, 'GLD': 0.34, 'TLT': 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'Value Tilt SPY/VLUE/GLD/TLT 30/20/34/16',
    allocation: { 'SPY': 0.30, 'VLUE': 0.20, 'GLD': 0.34, 'TLT': 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'Min Vol Tilt SPY/USMV/GLD/TLT 30/20/34/16',
    allocation: { 'SPY': 0.30, 'USMV': 0.20, 'GLD': 0.34, 'TLT': 0.16 },
    rebalanceFrequency: 'annual',
  },
  {
    name: 'Factor Blend SPY/MTUM/VLUE/USMV/GLD/TLT 20/10/10/10/34/16',
    allocation: { 'SPY': 0.20, 'MTUM': 0.10, 'VLUE': 0.10, 'USMV': 0.10, 'GLD': 0.34, 'TLT': 0.16 },
    rebalanceFrequency: 'annual',
  },
];
