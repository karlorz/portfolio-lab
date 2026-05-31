## [2026-05-29] retro | loop cycle: regime-transition-forecaster
- Friction:       Test ordering flakiness (2 tests pass in isolation but fail in full suite — pre-existing)
- Miss:           Case mismatch caught by simplify-worker, not by initial TDD design
- Improve:        Always normalize enum-like string inputs at API boundaries
- Generalize?:    yes (case normalization pattern applies to all signal modules)
- ClaudeMd?:      yes (regime transition forecaster is a new P1 feature)
- WorkflowShift?: no
