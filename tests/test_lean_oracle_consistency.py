"""
Consistency checks between physics/known_answers.py and the Lean oracle in
lean/Oracle/Potential.lean.

Two independent-lifecycle groups of tests live here:

- Group A (below, always runs): pins the numbers physics/known_answers.py
  computes against values recomputed independently in this test file, using
  the SAME formulas the Lean theorems state. This does not require Lean or
  ax-prover to have run at all -- it already catches a change to one of
  known_answers.py's hardcoded constants (8.0, 4.0, etc.) today. Comparing a
  literal to the same literal restated a second time would prove nothing, so
  every assertion here extracts its "actual" value from a real function call.

- Group B (skips until the Lean side has actually been run): checks that
  every theorem in lean/Oracle/Potential.lean has been discharged by
  ax-prover (see PROJECT_STATE.md for the exact command), by reading the
  archived results/lean_oracle_prove_output.json, AND independently that no
  `sorry` remains in the .lean source -- a JSON can report one theorem as
  proven while a different theorem in the same file is still unproven.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from physics.known_answers import (
    eyring_kramers_rate_0d,
    find_well_positions,
    free_energy_difference,
)
from physics.potential import potential_derivative

LEAN_SOURCE = Path("lean/Oracle/Potential.lean")
PROVE_OUTPUT = Path("results/lean_oracle_prove_output.json")

EXPECTED_THEOREMS = [
    "Oracle.Potential:V_hasDerivAt",
    "Oracle.Potential:V'_hasDerivAt",
    "Oracle.Potential:critical_points_b0",
    "Oracle.Potential:curvature_at_wells",
    "Oracle.Potential:curvature_at_saddle",
    "Oracle.Potential:barrier_height_eq",
    "Oracle.Potential:potential_even_at_b0",
]

TOLERANCE = 1e-10


# --- Group A: runs today, never skips -------------------------------------


def test_rate_prefactor_matches_curvature_formula():
    """
    eyring_kramers_rate_0d(beta, A) internally hardcodes
    curvature_at_well=8*A and curvature_at_barrier=4*A (known_answers.py
    lines 93-94) inside its prefactor sqrt(8A*4A)/(2*pi). We recover that
    prefactor from the function's OUTPUT (by dividing out the exp(-beta*A)
    exponential factor) and compare it against the same closed-form
    sqrt(8A*4A)/(2*pi) computed independently here. This is exactly what
    Lean's curvature_at_wells/curvature_at_saddle theorems state as
    consequences of V'_hasDerivAt -- if someone edits the 8.0 or 4.0 inside
    known_answers.py, this recovered prefactor changes and the test fails,
    unlike a literal-vs-literal comparison which would not notice.
    """
    for beta, barrier_height_value in [(5.0, 1.0), (3.0, 2.0)]:
        rate = eyring_kramers_rate_0d(beta=beta, A=barrier_height_value)
        recovered_prefactor = rate * np.exp(beta * barrier_height_value)

        curvature_at_well = 8.0 * barrier_height_value
        curvature_at_barrier = 4.0 * barrier_height_value
        expected_prefactor = np.sqrt(curvature_at_well * curvature_at_barrier) / (2.0 * np.pi)

        assert abs(recovered_prefactor - expected_prefactor) < TOLERANCE


def test_potential_derivative_matches_formula_at_sample_points():
    """
    Recomputes 4*A*x*(x**2-1)+b independently at several (A, b, x) sample
    points and compares against potential_derivative()'s actual output --
    this is the same formula Lean's V_hasDerivAt fixes as dV/dx.
    """
    sample_points = [
        (1.0, 0.0, 0.5),
        (1.0, 0.0, -1.3),
        (2.0, 0.1, 0.0),
        (0.5, -0.2, 2.0),
    ]
    for barrier_height_value, tilt, x in sample_points:
        expected_slope = 4 * barrier_height_value * x * (x**2 - 1) + tilt
        actual_slope = potential_derivative(x, A=barrier_height_value, b=tilt)

        assert abs(actual_slope - expected_slope) < TOLERANCE


def test_symmetric_wells_and_zero_free_energy():
    """
    At b=0, find_well_positions must recover exactly (-1, 1) and
    free_energy_difference must be exactly 0 -- the same claims Lean's
    critical_points_b0 and potential_even_at_b0 theorems state.
    """
    x_minus, x_plus = find_well_positions(A=1.0, b=0.0)
    assert abs(x_minus - (-1.0)) < TOLERANCE
    assert abs(x_plus - 1.0) < TOLERANCE

    delta_F = free_energy_difference(A=1.0, b=0.0)
    assert abs(delta_F) < TOLERANCE


# --- Group B: skips until ax-prover has actually discharged the proofs ----


def test_all_lean_theorems_proven():
    """
    Checks results/lean_oracle_prove_output.json (the archived ax-prover CLI
    output) reports success for every theorem in EXPECTED_THEOREMS. Skips
    with the exact command to run if that file doesn't exist yet -- this is
    a deliberate pending step (see PROJECT_STATE.md), not a silently passing
    check.
    """
    if not PROVE_OUTPUT.exists():
        pytest.skip(
            f"Lean oracle not proven yet -- run: cd lean && lake exe cache get "
            f"&& lake build && ax-prover prove Oracle.Potential --folder . "
            f"-o ../{PROVE_OUTPUT}"
        )

    prove_results = json.loads(PROVE_OUTPUT.read_text(encoding="utf-8"))

    for theorem_key in EXPECTED_THEOREMS:
        assert theorem_key in prove_results, f"missing prover output for {theorem_key}"
        assert prove_results[theorem_key]["success"] is True, (
            f"{theorem_key} was not proven: {prove_results[theorem_key].get('error')}"
        )
        assert prove_results[theorem_key]["error"] is None


def test_lean_source_has_no_remaining_sorry():
    """
    Independent check from test_all_lean_theorems_proven: a JSON can report
    one theorem as successfully proven while a DIFFERENT theorem in the same
    file is still unproven, so we also directly scan the .lean source for
    any remaining `sorry`. Gated on the SAME "has ax-prover actually been
    run" signal as test_all_lean_theorems_proven (PROVE_OUTPUT existing),
    not on LEAN_SOURCE existing -- the scaffolded file with its `sorry`
    placeholders exists from the moment this module is written, long before
    anyone has attempted a proof.
    """
    if not PROVE_OUTPUT.exists():
        pytest.skip(
            f"Lean oracle not proven yet -- run: cd lean && lake exe cache get "
            f"&& lake build && ax-prover prove Oracle.Potential --folder . "
            f"-o ../{PROVE_OUTPUT}"
        )

    lean_source_text = LEAN_SOURCE.read_text(encoding="utf-8")
    assert "sorry" not in lean_source_text, (
        f"{LEAN_SOURCE} still contains 'sorry' -- proofs incomplete"
    )
