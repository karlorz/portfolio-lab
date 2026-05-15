"""
Factor Correlation Matrix Calculator (v4.10 Phase 1)

Computes correlation matrices for factor ETFs to verify diversification
benefits and identify redundancy for risk premia harvesting.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Factor ETFs
FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'USMV']


def load_factor_prices(db_path: Path) -> Dict[str, List[float]]:
    """Load price data for factor ETFs from database."""
    closes = {sym: [] for sym in FACTOR_ETFS}
    
    with sqlite3.connect(db_path) as conn:
        for symbol in FACTOR_ETFS:
            cursor = conn.execute("""
                SELECT close FROM factor_prices 
                WHERE symbol = ? 
                ORDER BY date ASC
            """, (symbol,))
            closes[symbol] = [row[0] for row in cursor.fetchall()]
    
    return closes


def calculate_returns(closes: List[float]) -> List[float]:
    """Calculate daily returns from price series."""
    return [(closes[i] / closes[i-1]) - 1 for i in range(1, len(closes))]


def calculate_correlation(returns1: List[float], returns2: List[float]) -> float:
    """Calculate Pearson correlation between two return series."""
    n = min(len(returns1), len(returns2))
    if n < 30:
        return 0.0
    
    r1 = returns1[:n]
    r2 = returns2[:n]
    
    mean1 = sum(r1) / n
    mean2 = sum(r2) / n
    
    variance1 = sum((x - mean1) ** 2 for x in r1) / n
    variance2 = sum((x - mean2) ** 2 for x in r2) / n
    
    if variance1 == 0 or variance2 == 0:
        return 0.0
    
    std1 = variance1 ** 0.5
    std2 = variance2 ** 0.5
    
    covariance = sum((r1[i] - mean1) * (r2[i] - mean2) for i in range(n)) / n
    correlation = covariance / (std1 * std2)
    
    return correlation


def build_correlation_matrix(prices: Dict[str, List[float]]) -> Dict:
    """Build correlation matrix for factor ETFs."""
    # Calculate returns
    returns = {sym: calculate_returns(closes) for sym, closes in prices.items()}
    
    # Build matrix
    matrix = {}
    for sym1 in FACTOR_ETFS:
        matrix[sym1] = {}
        for sym2 in FACTOR_ETFS:
            if sym1 == sym2:
                matrix[sym1][sym2] = 1.0
            else:
                corr = calculate_correlation(returns[sym1], returns[sym2])
                matrix[sym1][sym2] = round(corr, 4)
    
    return matrix


def analyze_redundancy(matrix: Dict) -> List[Tuple[str, str, float]]:
    """Identify highly correlated factor pairs (>0.8)."""
    redundant = []
    for i, sym1 in enumerate(FACTOR_ETFS):
        for sym2 in FACTOR_ETFS[i+1:]:
            corr = matrix[sym1][sym2]
            if abs(corr) > 0.8:
                redundant.append((sym1, sym2, corr))
    return redundant


def generate_report(matrix: Dict, redundant: List[Tuple]) -> str:
    """Generate correlation matrix report."""
    lines = [
        "# Factor ETF Correlation Matrix Report (v4.10)",
        "",
        "Generated: " + str(Path(__file__).stat().st_mtime),
        "",
        "## Correlation Matrix",
        ""
    ]
    
    # Header
    header = "| ETF | " + " | ".join(FACTOR_ETFS) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["-----"] * (len(FACTOR_ETFS) + 1)) + "|")
    
    # Rows
    for sym in FACTOR_ETFS:
        row = f"| {sym} |"
        for sym2 in FACTOR_ETFS:
            corr = matrix[sym][sym2]
            row += f" {corr:+.3f} |"
        lines.append(row)
    
    lines.append("")
    lines.append("## Redundancy Analysis")
    lines.append("")
    
    if redundant:
        lines.append("**WARNING: High correlation pairs detected:**")
        for sym1, sym2, corr in redundant:
            lines.append(f"- {sym1}-{sym2}: {corr:.3f} (consider capping combined allocation)")
    else:
        lines.append("No high correlation pairs detected (all < 0.80).")
        lines.append("")
        lines.append("Factor diversification benefits confirmed.")
    
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Correlation < 0.5: Good diversification")
    lines.append("- Correlation 0.5-0.8: Moderate overlap")
    lines.append("- Correlation > 0.8: High redundancy, consider capping")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """CLI for correlation matrix generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Factor ETF Correlation Matrix v4.10")
    parser.add_argument("--output", type=str, default="data/factor_correlation_matrix.json")
    parser.add_argument("--report", type=str, default="data/factor_correlation_report.md")
    
    args = parser.parse_args()
    
    # Load data
    db_path = Path("data/factors/factor_data.db")
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.info("Run factor data population first")
        return
    
    logger.info("Loading factor price data...")
    prices = load_factor_prices(db_path)
    
    # Check data availability
    for sym, closes in prices.items():
        logger.info(f"  {sym}: {len(closes)} records")
    
    # Build matrix
    logger.info("Building correlation matrix...")
    matrix = build_correlation_matrix(prices)
    
    # Analyze redundancy
    redundant = analyze_redundancy(matrix)
    
    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'factors': FACTOR_ETFS,
            'matrix': matrix,
            'redundant_pairs': [{'pair': f"{p[0]}-{p[1]}", 'correlation': p[2]} for p in redundant],
            'diversification_score': 1.0 - (len(redundant) / 6)  # 6 possible pairs
        }, f, indent=2)
    logger.info(f"Saved correlation matrix to {output_path}")
    
    # Generate report
    report = generate_report(matrix, redundant)
    report_path = Path(args.report)
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved report to {report_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("Factor Correlation Matrix Summary")
    print("="*50)
    for sym1 in FACTOR_ETFS:
        for sym2 in FACTOR_ETFS:
            if sym1 < sym2:
                corr = matrix[sym1][sym2]
                status = "OK" if abs(corr) < 0.8 else "WARN"
                print(f"  {sym1}-{sym2}: {corr:+.3f} [{status}]")
    
    if redundant:
        print(f"\nWARNING: {len(redundant)} redundant pairs detected")
    else:
        print("\nAll factor correlations acceptable (< 0.80)")


if __name__ == "__main__":
    main()
