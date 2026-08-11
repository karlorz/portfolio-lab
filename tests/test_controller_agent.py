#!/usr/bin/env python3
"""Tests for src/agents/controller_agent.py — champion allocation SoT (Item 3).

ControllerAgent.DEFAULT_ALLOCATION must derive from src.paths.BASE_ALLOCATION
(single source of truth for the 46/38/16 champion) rather than a hardcoded
literal. Module import is numpy-only under PORTFOLIO_LAB_ENABLE_ML=0 (torch
gated at module level); instantiation requires ML and is out of scope here.
"""

import os

os.environ["PORTFOLIO_LAB_ENABLE_ML"] = "0"

import numpy as np
import pytest

from src.paths import BASE_ALLOCATION
from src.agents.controller_agent import ControllerAgent


def test_default_allocation_derives_from_base_allocation():
    expected = np.array([0.46, 0.38, 0.16, 0.0])
    assert np.allclose(ControllerAgent.DEFAULT_ALLOCATION, expected, atol=1e-9)
    assert ControllerAgent.DEFAULT_ALLOCATION[0] == pytest.approx(BASE_ALLOCATION["SPY"], abs=1e-9)
    assert ControllerAgent.DEFAULT_ALLOCATION[1] == pytest.approx(BASE_ALLOCATION["GLD"], abs=1e-9)
    assert ControllerAgent.DEFAULT_ALLOCATION[2] == pytest.approx(BASE_ALLOCATION["TLT"], abs=1e-9)
    assert ControllerAgent.DEFAULT_ALLOCATION[3] == 0.0
    assert ControllerAgent.N_ASSETS == 4
