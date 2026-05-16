"""
Portfolio-Lab v7.01: LLM Macro/Narrative Signal Generator

Two-tier architecture:
- Tier 1 (LLM): Analysis of Fed statements, macro data releases, news → structured sentiment
- Tier 2 (Rule-based, always available): Macro indicator scoring with heuristic NLP

This implementation is **Tier 2 only** — rule-based macro scoring with zero ML deps.
The LLM tier can be added later as an enhancement on top.

Approach:
1. Parse macro release calendar (CPI, NFP, GDP, FOMC, ISM, retail sales)
2. Score actual vs consensus using heuristic rules
3. Run simple text sentiment analysis on FOMC statements (rate-hike/dove/hawk key words)
4. Composite macro narrative score → per-asset signals (-1 to +1)
5. Weight in EnsembleVoter at 3-5%

Usage:
    python -m src.signals.llm_narrative_signal signal   # Current narrative assessment
    python -m src.signals.llm_narrative_signal calendar  # Upcoming macro events
    python -m src.signals.llm_narrative_signal explain   # Detailed signal breakdown
"""

import json
import logging
import re
import math
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Project root
project_root = Path(__file__).parent.parent.parent
DATA_DIR = project_root / "data"

# ─── Types ────────────────────────────────────────────────────────────────────


class NarrativeTone(Enum):
    """Macro narrative tone classification."""
    HAWKISH = "hawkish"          # Tightening / contractionary bias
    NEUTRAL_HAWKISH = "neutral_hawkish"
    NEUTRAL = "neutral"
    NEUTRAL_DOVISH = "neutral_dovish"
    DOVISH = "dovish"             # Easing / expansionary bias
    POSITIVE = "positive"         # Growth-positive news
    CAUTIOUS = "cautious"        # Risk-off / defensive positioning


class MacroDataType(Enum):
    """Types of macro data releases."""
    CPI = "cpi"
    CORE_CPI = "core_cpi"
    PPI = "ppi"
    NFP = "nfp"                    # Non-farm payrolls
    UNEMPLOYMENT = "unemployment"
    GDP = "gdp"
    FOMC = "fomc"                  # FOMC rate decision
    FOMC_MINUTES = "fomc_minutes"
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"
    RETAIL_SALES = "retail_sales"
    INDUSTRIAL_PRODUCTION = "industrial_production"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    HOUSING_STARTS = "housing_starts"
    JOLTS = "jolts"                # Job openings


@dataclass
class MacroReleaseReading:
    """A single macro data release assessment."""
    data_type: MacroDataType
    release_date: str
    actual: Optional[float]
    consensus: Optional[float]
    surprise: Optional[float]       # actual - consensus (standardized)
    surprise_z: Optional[float]    # z-score of surprise
    tone: NarrativeTone
    narrative_score: float         # -1 (bearish) to +1 (bullish)
    confidence: float              # 0-1
    note: str = ""


@dataclass
class NarrativeSignal:
    """Complete macro narrative signal assessment."""
    timestamp: str
    composite_score: float          # -1 (bearish) to +1 (bullish)
    confidence: float               # 0-1
    
    # Per-asset signals
    equity_signal: float            # SPY direction (-1 to +1)
    bond_signal: float              # TLT direction (-1 to +1)
    gold_signal: float              # GLD direction (-1 to +1)
    
    # Breakdown
    recent_releases: List[MacroReleaseReading]
    fomc_tone: NarrativeTone
    macro_health: str              # "expansion", "slowdown", "recession", "recovery"
    
    # Metadata
    num_releases_analyzed: int
    data_freshness_days: int       # Days since last macro data point
    explanation: str


# ─── 2026 Macro Calendar ─────────────────────────────────────────────────────

# Hardcoded 2026 macro release calendar (CPI, NFP, FOMC, GDP)
# In production, this would be sourced from a calendar API
MACRO_CALENDAR_2026: Dict[str, List[Dict]] = {
    "cpi": [
        {"date": "2026-01-15", "name": "CPI January"},
        {"date": "2026-02-12", "name": "CPI February"},
        {"date": "2026-03-12", "name": "CPI March"},
        {"date": "2026-04-15", "name": "CPI April"},
        {"date": "2026-05-13", "name": "CPI May"},
        {"date": "2026-06-11", "name": "CPI June"},
        {"date": "2026-07-15", "name": "CPI July"},
        {"date": "2026-08-12", "name": "CPI August"},
        {"date": "2026-09-11", "name": "CPI September"},
        {"date": "2026-10-15", "name": "CPI October"},
        {"date": "2026-11-12", "name": "CPI November"},
        {"date": "2026-12-10", "name": "CPI December"},
    ],
    "nfp": [
        {"date": "2026-01-09", "name": "NFP January"},
        {"date": "2026-02-06", "name": "NFP February"},
        {"date": "2026-03-06", "name": "NFP March"},
        {"date": "2026-04-03", "name": "NFP April"},
        {"date": "2026-05-08", "name": "NFP May"},
        {"date": "2026-06-05", "name": "NFP June"},
        {"date": "2026-07-03", "name": "NFP July"},
        {"date": "2026-08-07", "name": "NFP August"},
        {"date": "2026-09-04", "name": "NFP September"},
        {"date": "2026-10-02", "name": "NFP October"},
        {"date": "2026-11-06", "name": "NFP November"},
        {"date": "2026-12-04", "name": "NFP December"},
    ],
    "fomc": [
        {"date": "2026-01-28", "name": "FOMC Meeting January"},
        {"date": "2026-03-18", "name": "FOMC Meeting March"},
        {"date": "2026-05-06", "name": "FOMC Meeting May"},
        {"date": "2026-06-17", "name": "FOMC Meeting June"},
        {"date": "2026-07-29", "name": "FOMC Meeting July"},
        {"date": "2026-09-16", "name": "FOMC Meeting September"},
        {"date": "2026-11-04", "name": "FOMC Meeting November"},
        {"date": "2026-12-16", "name": "FOMC Meeting December"},
    ],
    "gdp": [
        {"date": "2026-01-30", "name": "GDP Q4 2025 Advance"},
        {"date": "2026-02-26", "name": "GDP Q4 2025 Revised"},
        {"date": "2026-03-26", "name": "GDP Q4 2025 Final"},
        {"date": "2026-04-30", "name": "GDP Q1 2026 Advance"},
        {"date": "2026-05-28", "name": "GDP Q1 2026 Revised"},
        {"date": "2026-06-25", "name": "GDP Q1 2026 Final"},
        {"date": "2026-07-30", "name": "GDP Q2 2026 Advance"},
        {"date": "2026-08-27", "name": "GDP Q2 2026 Revised"},
        {"date": "2026-09-24", "name": "GDP Q2 2026 Final"},
        {"date": "2026-10-30", "name": "GDP Q3 2026 Advance"},
        {"date": "2026-11-25", "name": "GDP Q3 2026 Revised"},
        {"date": "2026-12-22", "name": "GDP Q3 2026 Final"},
    ],
    "ism_manufacturing": [
        {"date": "2026-01-03", "name": "ISM Manufacturing December"},
        {"date": "2026-02-03", "name": "ISM Manufacturing January"},
        {"date": "2026-03-03", "name": "ISM Manufacturing February"},
        {"date": "2026-04-03", "name": "ISM Manufacturing March"},
        {"date": "2026-05-04", "name": "ISM Manufacturing April"},
        {"date": "2026-06-03", "name": "ISM Manufacturing May"},
        {"date": "2026-07-03", "name": "ISM Manufacturing June"},
        {"date": "2026-08-04", "name": "ISM Manufacturing July"},
        {"date": "2026-09-03", "name": "ISM Manufacturing August"},
        {"date": "2026-10-03", "name": "ISM Manufacturing September"},
        {"date": "2026-11-04", "name": "ISM Manufacturing October"},
        {"date": "2026-12-03", "name": "ISM Manufacturing November"},
    ],
}


class LLMNarrativeSignalGenerator:
    """
    Macro/narrative signal generator with rule-based fallback.

    Analyzes macro data releases and FOMC statements to generate
    directional signals for equity, bonds, and gold.

    IMPORTANT: This is the rule-based fallback only. When LLM tier is added,
    it wraps this class and uses these scores as priors.
    """

    # FOMC hawkish/dovish keyword dictionaries
    HAWKISH_KEYWORDS = [
        "tighten", "tightening", "overheating", "inflationary", "above target",
        "persistent inflation", "restrictive", "higher for longer", "contain",
        "withdrawal of accommodation", "above mandate", "upside risk",
        "further firming", "vigilant", "preemptively",
    ]
    DOVISH_KEYWORDS = [
        "accommodative", "ease", "easing", "patient", "data-dependent",
        "gradual", "measured", "support growth", "below target",
        "downside risk", "room to run", "labor slack", "spare capacity",
        "transitory", "weakened", "softening", "uncertain outlook",
        "appropriate to cut", "lower rates",
    ]

    # ISM thresholds: below 50 = contraction, above 50 = expansion
    ISM_CONTRACTION = 45.0
    ISM_EXPANSION = 52.0

    STATE_FILE = DATA_DIR / "narrative_signal_state.json"

    def __init__(self):
        self._recent_releases: List[MacroReleaseReading] = []
        self._latest_fomc_tone: NarrativeTone = NarrativeTone.NEUTRAL
        self._load_state()

    def _load_state(self):
        """Load persistent state for recent macro readings."""
        if self.STATE_FILE.exists():
            try:
                with open(self.STATE_FILE) as f:
                    data = json.load(f)
                if isinstance(data, dict) and "recent_releases" in data:
                    self._recent_releases = [
                        MacroReleaseReading(
                            data_type=MacroDataType(r["data_type"]),
                            release_date=r["release_date"],
                            actual=r.get("actual"),
                            consensus=r.get("consensus"),
                            surprise=r.get("surprise"),
                            surprise_z=r.get("surprise_z"),
                            tone=NarrativeTone(r.get("tone", "neutral")),
                            narrative_score=r.get("narrative_score", 0.0),
                            confidence=r.get("confidence", 0.0),
                            note=r.get("note", ""),
                        )
                        for r in data["recent_releases"]
                    ]
                    self._latest_fomc_tone = NarrativeTone(
                        data.get("fomc_tone", "neutral")
                    )
            except Exception as e:
                logger.warning(f"Failed to load narrative state: {e}")

    def _save_state(self):
        """Persist current state."""
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATE_FILE, "w") as f:
            json.dump({
                "recent_releases": [
                    {
                        "data_type": r.data_type.value,
                        "release_date": r.release_date,
                        "actual": r.actual,
                        "consensus": r.consensus,
                        "surprise": r.surprise,
                        "surprise_z": r.surprise_z,
                        "tone": r.tone.value,
                        "narrative_score": r.narrative_score,
                        "confidence": r.confidence,
                        "note": r.note,
                    }
                    for r in self._recent_releases[-20:]
                ],
                "fomc_tone": self._latest_fomc_tone.value,
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)

    def get_upcoming_events(self, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming macro events within the specified window."""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        events = []

        for data_type, releases in MACRO_CALENDAR_2026.items():
            for release in releases:
                release_date = date.fromisoformat(release["date"])
                if today <= release_date <= cutoff:
                    events.append({
                        "date": release["date"],
                        "type": data_type,
                        "name": release["name"],
                        "days_until": (release_date - today).days,
                    })

        return sorted(events, key=lambda e: e["date"])

    def _assess_fomc_tone(self, fomc_statement: Optional[str] = None) -> NarrativeTone:
        """
        Assess FOMC tone using keyword analysis.
        
        If no statement provided, returns the last assessed tone.
        """
        if fomc_statement is None:
            return self._latest_fomc_tone

        text = fomc_statement.lower()

        # Count hawkish and dovish keyword occurrences
        hawkish_count = sum(1 for kw in self.HAWKISH_KEYWORDS if kw in text)
        dovish_count = sum(1 for kw in self.DOVISH_KEYWORDS if kw in text)

        # Rate change direction keywords
        rate_hike_patterns = [
            r"raise(d)? (the )?federal funds rate",
            r"increase(d)? (the )?interest rate",
            r"quarter-point increase",
            r"25-basis.point",
            r"50-basis.point",
            r"tightening (cycle|stance)",
        ]
        rate_cut_patterns = [
            r"cut(s)? (the )?federal funds rate",
            r"lower(ed)? (the )?interest rate",
            r"reduc(ed|ing|es) (the )?rate",
            r"quarter-point cut",
            r"easing (cycle|stance)",
        ]

        hike_hints = sum(1 for p in rate_hike_patterns if re.search(p, text))
        cut_hints = sum(1 for p in rate_cut_patterns if re.search(p, text))

        net_score = (dovish_count - hawkish_count) + (cut_hints - hike_hints) * 2

        if net_score >= 3:
            tone = NarrativeTone.DOVISH
        elif net_score >= 1:
            tone = NarrativeTone.NEUTRAL_DOVISH
        elif net_score <= -3:
            tone = NarrativeTone.HAWKISH
        elif net_score <= -1:
            tone = NarrativeTone.NEUTRAL_HAWKISH
        else:
            tone = NarrativeTone.NEUTRAL

        self._latest_fomc_tone = tone
        return tone

    def _evaluate_macro_release(
        self,
        data_type: MacroDataType,
        actual: Optional[float],
        consensus: Optional[float],
    ) -> MacroReleaseReading:
        """
        Evaluate a single macro data release.
        
        Uses heuristics based on data type to determine:
        - Surprise magnitude (standardized)
        - Narrative tone
        - Directional score for portfolio
        """
        today_str = datetime.now().strftime("%Y-%m-%d")

        if actual is None or consensus is None:
            return MacroReleaseReading(
                data_type=data_type,
                release_date=today_str,
                actual=actual,
                consensus=consensus,
                surprise=None,
                surprise_z=None,
                tone=NarrativeTone.NEUTRAL,
                narrative_score=0.0,
                confidence=0.0,
                note="Missing data",
            )

        # Compute surprise
        if consensus != 0:
            surprise_pct = (actual - consensus) / abs(consensus)
        else:
            surprise_pct = actual - consensus if actual else 0

        # Standardize surprise (z-score approximation using historical vol estimates)
        # Typical macro surprise std dev varies by indicator
        surprise_std_estimates = {
            MacroDataType.CPI: 0.15,          # 15% typical surprise
            MacroDataType.CORE_CPI: 0.12,
            MacroDataType.PPI: 0.20,
            MacroDataType.NFP: 0.30,           # NFP more volatile
            MacroDataType.GDP: 0.10,
            MacroDataType.ISM_MANUFACTURING: 2.0,  # Points, not percent
            MacroDataType.ISM_SERVICES: 2.0,
            MacroDataType.RETAIL_SALES: 0.25,
            MacroDataType.UNEMPLOYMENT: 0.2,      # Rate changes in points
        }

        std_est = surprise_std_estimates.get(data_type, 0.15)
        if data_type in (MacroDataType.ISM_MANUFACTURING, MacroDataType.ISM_SERVICES):
            # ISM measured in index points
            surprise_z = surprise_pct / std_est if std_est > 0 else 0
        else:
            surprise_z = surprise_pct / std_est if std_est > 0 else 0

        # Clip extreme z-scores
        surprise_z = max(min(surprise_z, 3.0), -3.0)

        # Determine tone based on data type and surprise direction
        narrative_score = 0.0
        tone = NarrativeTone.NEUTRAL
        note = ""
        confidence = min(abs(surprise_z) / 2.0, 1.0)  # Higher surprise = higher confidence

        # --- CPI / Inflation data ---
        if data_type in (MacroDataType.CPI, MacroDataType.CORE_CPI, MacroDataType.PPI):
            # Higher inflation = hawkish → negative for bonds, mixed for equities
            if surprise_z > 0.3:
                tone = NarrativeTone.HAWKISH
                narrative_score = -0.6  # Hawkish is negative for risk
                note = f"Inflation above consensus by {surprise_pct:.1%}"
            elif surprise_z < -0.3:
                tone = NarrativeTone.DOVISH
                narrative_score = 0.5   # Disinflation is positive for risk
                note = f"Inflation below consensus by {abs(surprise_pct):.1%}"
            else:
                narrative_score = 0.0
                note = "Inflation in line with expectations"

        # --- NFP / Employment ---
        elif data_type == MacroDataType.NFP:
            # Strong NFP = positive for equities, hawkish for bonds
            if surprise_z > 0.5:
                tone = NarrativeTone.POSITIVE
                narrative_score = 0.4
                note = f"Jobs beat consensus by {abs(surprise_pct):.1%}"
            elif surprise_z < -0.5:
                tone = NarrativeTone.CAUTIOUS
                narrative_score = -0.4
                note = f"Jobs missed consensus by {abs(surprise_pct):.1%}"
            else:
                narrative_score = 0.1
                note = "Jobs in line with expectations"

        # --- GDP ---
        elif data_type == MacroDataType.GDP:
            if surprise_z > 0.5:
                tone = NarrativeTone.POSITIVE
                narrative_score = 0.6
                note = f"GDP beat consensus by {surprise_pct:.1%}"
            elif surprise_z < -0.5:
                tone = NarrativeTone.CAUTIOUS
                narrative_score = -0.6
                note = f"GDP missed consensus by {abs(surprise_pct):.1%}"
            else:
                narrative_score = 0.1
                note = "GDP in line with expectations"

        # --- ISM ---
        elif data_type in (MacroDataType.ISM_MANUFACTURING, MacroDataType.ISM_SERVICES):
            # ISM: below 50 = contraction, above = expansion
            in_expansion = actual >= 50.0
            improving = surprise_z > 0

            if in_expansion and improving:
                tone = NarrativeTone.POSITIVE
                narrative_score = 0.5
                note = f"ISM {actual:.1f} — expansion improving"
            elif in_expansion and not improving:
                tone = NarrativeTone.NEUTRAL
                narrative_score = 0.2
                note = f"ISM {actual:.1f} — expansion slowing"
            elif not in_expansion and not improving:
                tone = NarrativeTone.CAUTIOUS
                narrative_score = -0.5
                note = f"ISM {actual:.1f} — contraction deepening"
            else:
                tone = NarrativeTone.NEUTRAL
                narrative_score = -0.2
                note = f"ISM {actual:.1f} — contraction improving"

        # --- Unemployment ---
        elif data_type == MacroDataType.UNEMPLOYMENT:
            if surprise_z > 0.5:  # Unemployment rising (bad)
                tone = NarrativeTone.CAUTIOUS
                narrative_score = -0.4
                note = f"Unemployment above consensus at {actual:.1%}"
            elif surprise_z < -0.5:  # Unemployment falling (good)
                tone = NarrativeTone.POSITIVE
                narrative_score = 0.3
                note = f"Unemployment below consensus at {actual:.1%}"
            else:
                narrative_score = 0.0
                note = "Unemployment in line"

        else:
            # Generic treatment
            if surprise_z > 1.0:
                narrative_score = 0.3
                tone = NarrativeTone.POSITIVE
            elif surprise_z < -1.0:
                narrative_score = -0.3
                tone = NarrativeTone.CAUTIOUS
            note = f"Data release: surprise z={surprise_z:.2f}"

        return MacroReleaseReading(
            data_type=data_type,
            release_date=today_str,
            actual=actual,
            consensus=consensus,
            surprise=surprise_pct,
            surprise_z=surprise_z,
            tone=tone,
            narrative_score=narrative_score,
            confidence=confidence,
            note=note,
        )

    def ingest_macro_release(
        self,
        data_type: str,
        actual: float,
        consensus: float,
    ) -> MacroReleaseReading:
        """Ingest a new macro data release and update state."""
        dt = MacroDataType(data_type)
        reading = self._evaluate_macro_release(dt, actual, consensus)
        self._recent_releases.append(reading)
        # Keep only last 20 releases
        if len(self._recent_releases) > 20:
            self._recent_releases = self._recent_releases[-20:]
        self._save_state()
        return reading

    def ingest_fomc_statement(self, statement: str) -> NarrativeTone:
        """Ingest an FOMC statement and assess its tone."""
        tone = self._assess_fomc_tone(statement)
        self._save_state()
        return tone

    def compute_macro_health(self) -> str:
        """Determine aggregate macro health from recent releases."""
        if not self._recent_releases:
            return "unknown"

        # Look at last 10 releases
        recent = self._recent_releases[-10:]
        if not recent:
            return "unknown"

        avg_score = np.mean([r.narrative_score for r in recent])

        if avg_score > 0.3:
            return "expansion"
        elif avg_score > 0.0:
            return "slowdown_growth"
        elif avg_score > -0.4:
            return "moderate_slowdown"
        else:
            return "contraction"

    def generate_signal(self) -> NarrativeSignal:
        """
        Generate composite narrative signal.

        Combines:
        1. Recent macro release scores (trailing weighted average)
        2. FOMC tone assessment
        3. Macro health classification

        Returns per-asset signals for equity, bonds, and gold.
        """
        now = datetime.now()
        timestamp = now.isoformat()

        # 1. Aggregate recent macro releases (last 5, declining weights)
        recent = self._recent_releases[-5:] if self._recent_releases else []
        num_releases = len(recent)

        if not recent:
            # No macro data — use FOMC tone alone (if available)
            fomc_factor = {
                NarrativeTone.DOVISH: 0.3,
                NarrativeTone.NEUTRAL_DOVISH: 0.15,
                NarrativeTone.NEUTRAL: 0.0,
                NarrativeTone.NEUTRAL_HAWKISH: -0.15,
                NarrativeTone.HAWKISH: -0.3,
                NarrativeTone.POSITIVE: 0.2,
                NarrativeTone.CAUTIOUS: -0.2,
            }.get(self._latest_fomc_tone, 0.0)

            fomc_confidence = {
                NarrativeTone.DOVISH: 0.4,
                NarrativeTone.NEUTRAL_DOVISH: 0.3,
                NarrativeTone.NEUTRAL: 0.1,
                NarrativeTone.NEUTRAL_HAWKISH: 0.3,
                NarrativeTone.HAWKISH: 0.4,
                NarrativeTone.POSITIVE: 0.35,
                NarrativeTone.CAUTIOUS: 0.35,
            }.get(self._latest_fomc_tone, 0.1)

            fomc_tone_str = self._latest_fomc_tone.value.replace("_", " ")

            if fomc_factor != 0.0:
                return NarrativeSignal(
                    timestamp=timestamp,
                    composite_score=round(fomc_factor, 4),
                    confidence=round(fomc_confidence, 4),
                    equity_signal=round(fomc_factor * 0.8, 4),
                    bond_signal=round(-fomc_factor * 0.6, 4),
                    gold_signal=round(abs(fomc_factor) * 0.3, 4),
                    recent_releases=[],
                    fomc_tone=self._latest_fomc_tone,
                    macro_health="unknown",
                    num_releases_analyzed=0,
                    data_freshness_days=999,
                    explanation=f"FOMC tone only: {fomc_tone_str}. No macro data ingested yet.",
                )

            return NarrativeSignal(
                timestamp=timestamp,
                composite_score=0.0,
                confidence=0.2,
                equity_signal=0.0,
                bond_signal=0.0,
                gold_signal=0.0,
                recent_releases=[],
                fomc_tone=self._latest_fomc_tone,
                macro_health="unknown",
                num_releases_analyzed=0,
                data_freshness_days=999,
                explanation="No macro data ingested yet",
            )

        # Weighted average: most recent gets highest weight
        weights = [0.35, 0.25, 0.20, 0.13, 0.07][:len(recent)]
        weights = [w / sum(weights) for w in weights]  # Normalize

        weighted_score = sum(
            r.narrative_score * w for r, w in zip(recent, weights)
        )
        avg_confidence = np.mean([r.confidence for r in recent])

        # FOMC tone factor
        fomc_factor = {
            NarrativeTone.DOVISH: 0.3,
            NarrativeTone.NEUTRAL_DOVISH: 0.15,
            NarrativeTone.NEUTRAL: 0.0,
            NarrativeTone.NEUTRAL_HAWKISH: -0.15,
            NarrativeTone.HAWKISH: -0.3,
            NarrativeTone.POSITIVE: 0.2,
            NarrativeTone.CAUTIOUS: -0.2,
        }.get(self._latest_fomc_tone, 0.0)

        # FOMC confidence: lower for neutral, higher for strong signals
        fomc_confidence = {
            NarrativeTone.DOVISH: 0.6,
            NarrativeTone.NEUTRAL_DOVISH: 0.4,
            NarrativeTone.NEUTRAL: 0.2,
            NarrativeTone.NEUTRAL_HAWKISH: 0.4,
            NarrativeTone.HAWKISH: 0.6,
            NarrativeTone.POSITIVE: 0.5,
            NarrativeTone.CAUTIOUS: 0.5,
        }.get(self._latest_fomc_tone, 0.2)

        # Composite score: 70% macro releases, 30% FOMC tone
        composite_score = weighted_score * 0.7 + fomc_factor * 0.3
        composite_score = max(min(composite_score, 1.0), -1.0)

        confidence = avg_confidence * 0.7 + fomc_confidence * 0.3

        # 2. Per-asset signal mapping

        # Equities: Positive macro narrative = bullish equities
        equity_signal = composite_score * 0.8  # Attenuate slightly

        # Bonds: Inverse relationship with growth narrative
        # Positive growth narrative → bond yields up → TLT down
        # Negative growth narrative → flight to safety → TLT up
        bond_signal = -composite_score * 0.6

        # Gold: Hedge narrative — strengthens during uncertainty/cautious tone
        # Weakens during strong growth / hawkish (strong dollar)
        if self._latest_fomc_tone in (NarrativeTone.CAUTIOUS, NarrativeTone.NEUTRAL):
            # Uncertainty or neutral → gold as hedge
            gold_signal = abs(composite_score) * 0.3
        elif self._latest_fomc_tone in (NarrativeTone.DOVISH, NarrativeTone.NEUTRAL_DOVISH):
            # Dovish → gold positive (weaker dollar, lower real rates)
            gold_signal = composite_score * 0.4
        else:
            # Hawkish → gold negative (stronger dollar)
            gold_signal = composite_score * 0.2

        # Check data freshness
        last_data_dates = [
            datetime.fromisoformat(r.release_date)
            for r in recent if r.release_date
        ]
        if last_data_dates:
            last_date = max(last_data_dates)
            freshness_days = (now - last_date).days
        else:
            freshness_days = 999

        # Decay confidence if data is stale
        if freshness_days > 60:
            confidence *= 0.5
        elif freshness_days > 30:
            confidence *= 0.75

        macro_health = self.compute_macro_health()

        # Build explanation
        fomc_tone_str = self._latest_fomc_tone.value.replace("_", " ")
        latest_releases = recent[-3:] if len(recent) >= 3 else recent
        release_notes = "; ".join(r.note for r in latest_releases)

        explanation = (
            f"Macro narrative: {macro_health.upper()} (score={composite_score:.2f}). "
            f"FOMC tone: {fomc_tone_str}. "
            f"Latest releases: {release_notes}. "
            f"Data freshness: {freshness_days}d ago."
        )

        return NarrativeSignal(
            timestamp=timestamp,
            composite_score=round(composite_score, 4),
            confidence=float(round(confidence, 4)),
            equity_signal=round(equity_signal, 4),
            bond_signal=round(bond_signal, 4),
            gold_signal=round(gold_signal, 4),
            recent_releases=recent,
            fomc_tone=self._latest_fomc_tone,
            macro_health=macro_health,
            num_releases_analyzed=num_releases,
            data_freshness_days=freshness_days,
            explanation=explanation,
        )

    def get_ensemble_signal(self) -> Dict:
        """
        Get signal formatted for EnsembleVoter integration.
        
        Returns dict with source, value, confidence, asset_signals.
        """
        signal = self.generate_signal()
        return {
            "source": "llm_narrative",
            "value": signal.composite_score,
            "confidence": signal.confidence,
            "asset_signals": {
                "SPY": signal.equity_signal,
                "TLT": signal.bond_signal,
                "GLD": signal.gold_signal,
                "IEF": signal.bond_signal * 0.5,  # IEF less sensitive
                "SHY": signal.bond_signal * 0.2,  # SHY very insensitive
            },
            "explanation": signal.explanation,
            "num_releases": signal.num_releases_analyzed,
            "fomc_tone": signal.fomc_tone.value,
            "macro_health": signal.macro_health,
        }

    def get_signal_reading(self) -> Dict:
        """Get formatted signal reading for CLI."""
        signal = self.generate_signal()
        upcoming = self.get_upcoming_events(30)

        return {
            "timestamp": signal.timestamp,
            "composite_score": signal.composite_score,
            "confidence": signal.confidence,
            "macro_health": signal.macro_health,
            "fomc_tone": signal.fomc_tone.value,
            "equity_signal": signal.equity_signal,
            "bond_signal": signal.bond_signal,
            "gold_signal": signal.gold_signal,
            "num_releases": signal.num_releases_analyzed,
            "data_freshness_days": signal.data_freshness_days,
            "explanation": signal.explanation,
            "upcoming_events": upcoming,
        }


# ─── Standalone Functions ─────────────────────────────────────────────────────


def get_narrative_signal() -> Dict:
    """Convenience function for quick signal access."""
    gen = LLMNarrativeSignalGenerator()
    return gen.get_ensemble_signal()


def get_narrative_status() -> Dict:
    """Get current narrative status."""
    gen = LLMNarrativeSignalGenerator()
    return gen.get_signal_reading()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Macro/Narrative Signal Generator (v7.01)"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="signal",
        choices=["signal", "calendar", "explain", "ingest"],
        help="Mode: signal (default), calendar, explain, ingest",
    )
    parser.add_argument(
        "--type",
        help="Data type for ingest mode (e.g., cpi, nfp, fomc)",
    )
    parser.add_argument(
        "--actual",
        type=float,
        help="Actual value for ingest mode",
    )
    parser.add_argument(
        "--consensus",
        type=float,
        help="Consensus estimate for ingest mode",
    )
    parser.add_argument(
        "--statement",
        help="FOMC statement text for ingest mode (fomc type)",
    )

    args = parser.parse_args()
    gen = LLMNarrativeSignalGenerator()

    if args.mode == "signal":
        signal = gen.generate_signal()
        print("=" * 65)
        print("v7.01 LLM MACRO/NARRATIVE SIGNAL (Rule-based Fallback)")
        print("=" * 65)
        print(f"Composite Score:  {signal.composite_score:+.4f}")
        print(f"Confidence:       {signal.confidence:.1%}")
        print(f"Macro Health:     {signal.macro_health.upper()}")
        print(f"FOMC Tone:        {signal.fomc_tone.value}")
        print()
        print("Per-Asset Signals:")
        print(f"  SPY: {signal.equity_signal:+.4f}")
        print(f"  TLT: {signal.bond_signal:+.4f}")
        print(f"  GLD: {signal.gold_signal:+.4f}")
        print()
        print(f"Releases Analyzed: {signal.num_releases_analyzed}")
        print(f"Data Freshness:    {signal.data_freshness_days}d ago")
        print()
        print("Explanation:")
        print(f"  {signal.explanation}")

    elif args.mode == "calendar":
        upcoming = gen.get_upcoming_events(60)
        print("=" * 65)
        print(f"UPCOMING MACRO EVENTS (next 60 days)")
        print("=" * 65)
        if not upcoming:
            print("No upcoming events in window.")
        for ev in upcoming:
            print(f"  {ev['date']}  [{ev['type'].upper():20s}] {ev['name']}")

    elif args.mode == "explain":
        signal = gen.generate_signal()
        status = gen.get_signal_reading()
        print("=" * 65)
        print("SIGNAL DECOMPOSITION")
        print("=" * 65)
        print(f"Composite: {signal.composite_score:+.4f} "
              f"(confidence: {signal.confidence:.1%})")
        print()
        print("Weighting Breakdown:")
        print(f"  Macro Releases (70%): weight = {signal.composite_score * 0.7:+.4f}")
        print(f"  FOMC Tone (30%):      weight = 0.0 (not yet modeled)")
        print()
        if signal.recent_releases:
            print("Recent Macro Releases:")
            for r in signal.recent_releases[-5:]:
                print(f"  [{r.data_type.value.upper():20s}] "
                      f"score={r.narrative_score:+.2f} "
                      f"conf={r.confidence:.0%} "
                      f"tone={r.tone.value:15s} "
                      f"| {r.note}")
        print()
        print("Data Freshness Check:")
        print(f"  Last data point: {signal.data_freshness_days}d ago")
        if signal.data_freshness_days > 60:
            print("  ⚠️  WARNING: Data is stale (>60 days). Reduce confidence.")
        print()
        print("FOMC Tone Context:")
        print(f"  {signal.fomc_tone.value.upper()} — "
              f"{'Dovish' if 'dovish' in signal.fomc_tone.value else 'Hawkish' if 'hawkish' in signal.fomc_tone.value else 'Neutral'}")

    elif args.mode == "ingest":
        if args.type == "fomc":
            if not args.statement:
                print("ERROR: --statement required for FOMC ingestion")
                sys.exit(1)
            tone = gen.ingest_fomc_statement(args.statement)
            print(f"FOMC statement ingested. Tone: {tone.value}")
        else:
            if not args.type or args.actual is None or args.consensus is None:
                print("ERROR: --type, --actual, and --consensus required")
                sys.exit(1)
            reading = gen.ingest_macro_release(args.type, args.actual, args.consensus)
            print(f"Macro release ingested: {reading.data_type.value}")
            print(f"  Score: {reading.narrative_score:+.2f}")
            print(f"  Tone:  {reading.tone.value}")
            print(f"  Note:  {reading.note}")


if __name__ == "__main__":
    main()
