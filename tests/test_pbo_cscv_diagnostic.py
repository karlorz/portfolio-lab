import numpy as np

from scripts.pbo_cscv_diagnostic import cscv_pbo


def test_cscv_pbo_is_zero_when_in_sample_winner_always_wins_oos():
    """A persistent winner should rank above the OOS median in every CSCV split."""
    alternating = np.tile([0.01, -0.01], 40)
    base = np.column_stack(
        [
            alternating + 0.003,
            alternating + 0.001,
            alternating - 0.001,
            alternating - 0.003,
        ]
    )

    pbo, logits = cscv_pbo(base, S=4)

    assert pbo == 0.0
    assert all(logit > 0 for logit in logits)
