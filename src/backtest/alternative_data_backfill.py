"""
Alternative Data Historical Backfill Generator
v2.60 Phase 4 Implementation - Portfolio-Lab

Generates synthetic historical signals for 2020-2023 period for backtesting
alternative data NLP integration before live deployment.
"""

import json
import random
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import os


@dataclass
class DailyAlternativeSignal:
    """Daily composite signal from alternative data sources."""
    date: str
    earnings_sentiment: float  # -1.0 to 1.0
    news_sentiment: float      # -1.0 to 1.0
    jobs_growth: float         # -1.0 to 1.0 (hiring velocity)
    social_sentiment: float    # -1.0 to 1.0
    composite_score: float     # -1.0 to 1.0 (weighted)
    regime: str                # 'risk_on', 'risk_off', 'neutral'
    confidence: float          # 0.0 to 1.0
    z_score: float            # normalized composite
    
    # Component availability flags
    has_earnings: bool
    has_news: bool
    has_jobs: bool
    has_social: bool


class AlternativeDataBackfill:
    """Generate realistic historical alternative data signals."""
    
    WEIGHTS = {
        'earnings': 0.40,
        'news': 0.30,
        'jobs': 0.20,
        'social': 0.10
    }
    
    # Known crisis periods for stress testing
    COVID_START = datetime(2020, 2, 20)
    COVID_BOTTOM = datetime(2020, 3, 23)
    COVID_RECOVERY = datetime(2020, 8, 1)
    
    INFLATION_PEAK = datetime(2022, 6, 1)
    BEAR_MARKET_2022 = datetime(2022, 1, 1)
    BEAR_BOTTOM = datetime(2022, 10, 12)
    
    RATE_HIKES_START = datetime(2022, 3, 1)
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.signals: List[DailyAlternativeSignal] = []
        
    def _is_crisis_period(self, date: datetime) -> tuple[bool, str]:
        """Determine if date falls within known crisis period."""
        # COVID crash
        if self.COVID_START <= date <= self.COVID_BOTTOM:
            return True, 'covid_crash'
        elif self.COVID_BOTTOM < date <= self.COVID_RECOVERY:
            return True, 'covid_recovery'
        
        # 2022 bear market
        elif self.BEAR_MARKET_2022 <= date <= self.BEAR_BOTTOM:
            return True, 'bear_2022'
        
        return False, 'normal'
    
    def _generate_earnings_sentiment(self, date: datetime, crisis: bool, crisis_type: str) -> float:
        """Generate earnings sentiment with crisis adjustments."""
        base = random.gauss(0.05, 0.25)  # Slightly positive bias normally
        
        if crisis:
            if crisis_type == 'covid_crash':
                base = random.gauss(-0.65, 0.30)  # Severe negative
            elif crisis_type == 'covid_recovery':
                base = random.gauss(0.15, 0.35)  # Volatile recovery
            elif crisis_type == 'bear_2022':
                base = random.gauss(-0.25, 0.20)  # Moderate negative (guidance cuts)
        
        # Earnings season clustering (Jan, Apr, Jul, Oct)
        month = date.month
        if month in [1, 4, 7, 10]:
            base *= 1.3  # Stronger signal during earnings
        
        return max(-1.0, min(1.0, base))
    
    def _generate_news_sentiment(self, date: datetime, crisis: bool, crisis_type: str) -> float:
        """Generate news sentiment with crisis adjustments."""
        base = random.gauss(0.02, 0.15)  # Near neutral normally
        
        if crisis:
            if crisis_type == 'covid_crash':
                base = random.gauss(-0.80, 0.20)  # Extreme negative news
            elif crisis_type == 'covid_recovery':
                base = random.gauss(0.10, 0.40)  # Mixed, volatile
            elif crisis_type == 'bear_2022':
                base = random.gauss(-0.35, 0.25)  # Fed policy uncertainty
        
        return max(-1.0, min(1.0, base))
    
    def _generate_jobs_growth(self, date: datetime, crisis: bool, crisis_type: str) -> float:
        """Generate job posting growth signal."""
        base = random.gauss(0.08, 0.20)  # Positive growth normally
        
        if crisis:
            if crisis_type == 'covid_crash':
                base = random.gauss(-0.90, 0.15)  # Hiring freeze
            elif crisis_type == 'covid_recovery':
                base = random.gauss(0.35, 0.30)  # Rapid rehiring
            elif crisis_type == 'bear_2022':
                base = random.gauss(-0.15, 0.20)  # Tech layoffs
        
        return max(-1.0, min(1.0, base))
    
    def _generate_social_sentiment(self, date: datetime, crisis: bool, crisis_type: str) -> float:
        """Generate social media sentiment (noisiest signal)."""
        base = random.gauss(0.0, 0.30)  # Very noisy normally
        
        if crisis:
            if crisis_type == 'covid_crash':
                base = random.gauss(-0.40, 0.40)  # Panic
            elif crisis_type == 'covid_recovery':
                base = random.gauss(0.25, 0.45)  # Optimistic but noisy
            elif crisis_type == 'bear_2022':
                base = random.gauss(-0.20, 0.35)  # Bearish sentiment
        
        return max(-1.0, min(1.0, base))
    
    def _calculate_regime(self, composite: float, confidence: float) -> str:
        """Map composite score to regime signal."""
        if confidence < 0.3:
            return 'neutral'
        
        if composite > 0.25:
            return 'risk_on'
        elif composite < -0.25:
            return 'risk_off'
        return 'neutral'
    
    def generate_daily_signal(self, date: datetime) -> DailyAlternativeSignal:
        """Generate a single day's alternative data signal."""
        crisis, crisis_type = self._is_crisis_period(date)
        
        # Generate component signals
        earnings = self._generate_earnings_sentiment(date, crisis, crisis_type)
        news = self._generate_news_sentiment(date, crisis, crisis_type)
        jobs = self._generate_jobs_growth(date, crisis, crisis_type)
        social = self._generate_social_sentiment(date, crisis, crisis_type)
        
        # Component availability (simulating data gaps)
        has_earnings = date.month in [1, 4, 7, 10] or random.random() > 0.7
        has_news = True  # News always available
        has_jobs = random.random() > 0.1  # Jobs data mostly available
        has_social = random.random() > 0.3  # Social somewhat available
        
        # Adjust weights based on availability
        active_weights = {}
        total_weight = 0.0
        
        if has_earnings:
            active_weights['earnings'] = self.WEIGHTS['earnings']
            total_weight += self.WEIGHTS['earnings']
        if has_news:
            active_weights['news'] = self.WEIGHTS['news']
            total_weight += self.WEIGHTS['news']
        if has_jobs:
            active_weights['jobs'] = self.WEIGHTS['jobs']
            total_weight += self.WEIGHTS['jobs']
        if has_social:
            active_weights['social'] = self.WEIGHTS['social']
            total_weight += self.WEIGHTS['social']
        
        # Normalize weights
        if total_weight > 0:
            for key in active_weights:
                active_weights[key] /= total_weight
        
        # Calculate composite
        composite = 0.0
        if has_earnings and 'earnings' in active_weights:
            composite += earnings * active_weights['earnings']
        if has_news and 'news' in active_weights:
            composite += news * active_weights['news']
        if has_jobs and 'jobs' in active_weights:
            composite += jobs * active_weights['jobs']
        if has_social and 'social' in active_weights:
            composite += social * active_weights['social']
        
        # Confidence based on data availability and crisis period
        base_confidence = 0.4 + 0.4 * (sum([has_earnings, has_news, has_jobs, has_social]) / 4)
        if crisis:
            base_confidence *= 1.15  # Higher confidence during clear signals
        confidence = min(1.0, base_confidence + random.gauss(0, 0.1))
        
        # Z-score (simplified - assumes historical mean ~0, std ~0.3)
        z_score = composite / 0.3
        
        # Determine regime
        regime = self._calculate_regime(composite, confidence)
        
        return DailyAlternativeSignal(
            date=date.strftime('%Y-%m-%d'),
            earnings_sentiment=round(earnings, 4),
            news_sentiment=round(news, 4),
            jobs_growth=round(jobs, 4),
            social_sentiment=round(social, 4),
            composite_score=round(composite, 4),
            regime=regime,
            confidence=round(confidence, 4),
            z_score=round(z_score, 4),
            has_earnings=has_earnings,
            has_news=has_news,
            has_jobs=has_jobs,
            has_social=has_social
        )
    
    def generate_backfill(self, start_date: str = '2020-01-01', 
                         end_date: str = '2026-05-13') -> List[DailyAlternativeSignal]:
        """Generate complete historical backfill."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current <= end:
            signal = self.generate_daily_signal(current)
            self.signals.append(signal)
            current += timedelta(days=1)
        
        return self.signals
    
    def save_to_json(self, filepath: str):
        """Save signals to JSON file."""
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '2.60',
                'phase': '4.1',
                'start_date': self.signals[0].date if self.signals else None,
                'end_date': self.signals[-1].date if self.signals else None,
                'total_days': len(self.signals),
                'weights': self.WEIGHTS,
                'description': 'Synthetic alternative data signals for backtesting'
            },
            'signals': [asdict(s) for s in self.signals]
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.signals)} days of alternative data signals to {filepath}")
    
    def generate_metadata(self) -> Dict:
        """Generate summary statistics for validation."""
        if not self.signals:
            return {}
        
        regimes = {'risk_on': 0, 'risk_off': 0, 'neutral': 0}
        crisis_periods = {
            'covid_crash': [],
            'covid_recovery': [],
            'bear_2022': [],
            'normal': []
        }
        
        for s in self.signals:
            regimes[s.regime] += 1
            
            date = datetime.strptime(s.date, '%Y-%m-%d')
            crisis, crisis_type = self._is_crisis_period(date)
            if crisis:
                crisis_periods[crisis_type].append(s.composite_score)
            else:
                crisis_periods['normal'].append(s.composite_score)
        
        return {
            'total_signals': len(self.signals),
            'regime_distribution': regimes,
            'regime_pct': {
                k: round(v / len(self.signals) * 100, 2) 
                for k, v in regimes.items()
            },
            'avg_confidence': round(
                sum(s.confidence for s in self.signals) / len(self.signals), 4
            ),
            'component_availability': {
                'earnings': round(
                    sum(1 for s in self.signals if s.has_earnings) / len(self.signals) * 100, 2
                ),
                'news': round(
                    sum(1 for s in self.signals if s.has_news) / len(self.signals) * 100, 2
                ),
                'jobs': round(
                    sum(1 for s in self.signals if s.has_jobs) / len(self.signals) * 100, 2
                ),
                'social': round(
                    sum(1 for s in self.signals if s.has_social) / len(self.signals) * 100, 2
                )
            },
            'crisis_period_analysis': {
                period: {
                    'count': len(scores),
                    'avg_sentiment': round(sum(scores) / len(scores), 4) if scores else 0
                }
                for period, scores in crisis_periods.items()
            }
        }


def main():
    """Generate historical backfill for v2.60 Phase 4."""
    print("=" * 60)
    print("v2.60 Alternative Data - Phase 4.1 Historical Backfill")
    print("=" * 60)
    
    backfill = AlternativeDataBackfill(seed=42)
    
    # Generate 2020-01-01 to 2026-05-13 (2,324 trading days)
    signals = backfill.generate_backfill('2020-01-01', '2026-05-13')
    
    # Save to JSON
    output_path = '/root/projects/portfolio-lab/data/signals/alternative_data_historical_2020_2026.json'
    backfill.save_to_json(output_path)
    
    # Generate metadata
    metadata = backfill.generate_metadata()
    
    metadata_path = '/root/projects/portfolio-lab/data/signals/alternative_data_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_path}")
    print("\nBackfill Summary:")
    print(json.dumps(metadata, indent=2))
    
    # Validation report
    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)
    
    checks = [
        ("Total signals >= 2300", len(signals) >= 2300),
        ("Regime distribution reasonable (neutral dominant)", 
         metadata['regime_pct']['neutral'] >= 60 and metadata['regime_pct']['neutral'] <= 90),
        ("Risk extremes detected appropriately", 
         metadata['regime_pct']['risk_on'] + metadata['regime_pct']['risk_off'] >= 10),
        ("Crisis periods detected", 
         metadata['crisis_period_analysis']['covid_crash']['avg_sentiment'] < -0.3),
        ("Component availability > 50%", 
         all(v > 50 for v in metadata['component_availability'].values())),
        ("COVID crash negative sentiment", 
         metadata['crisis_period_analysis']['covid_crash']['avg_sentiment'] < 0),
        ("2022 bear negative sentiment", 
         metadata['crisis_period_analysis']['bear_2022']['avg_sentiment'] < 0),
        ("Avg confidence > 0.5", metadata['avg_confidence'] > 0.5),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All validation checks passed!")
        print("\nPhase 4.1 Historical Backfill COMPLETE")
        print("Next: Phase 4.2 Walk-Forward Testing (scheduled 2026-05-15)")
    else:
        print("\n⚠️  Some checks failed - review recommended")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
