"""
Consistency checks between the Lean oracle (lean/Oracle/Potential.lean) and
the Python code. These tests read the .lean file itself:
- the Lean definitions of V and V' evaluate to the same numbers as
  physics/potential.py;
- the 7 theorem statements are exactly the expected ones, so a statement
  cannot be weakened unnoticed;
- the constants the theorems state (wells at +-1, curvatures 8A and -4A,
  barrier height A) agree with the Python potential.

Whether the theorems are proved is answered by Lean itself (`#print axioms`,
results/lean_v_hasDerivAt_check.log), not by these tests.
"""

import re
from pathlib import Path

from physics.known_answers import barrier_height, find_well_positions
from physics.potential import potential, potential_derivative

LEAN_SOURCE = Path("lean/Oracle/Potential.lean")
SAMPLE_POINTS = [(1.0, 0.0, 0.5), (1.0, 0.0, -1.3), (2.0, 0.1, 0.0), (0.5, -0.2, 2.0)]
EXPECTED_STATEMENTS = [
    "theorem V_hasDerivAt (A b x : ℝ) : HasDerivAt (V A b) (V' A b x) x",
    "theorem V'_hasDerivAt (A b x : ℝ) : HasDerivAt (V' A b) (12 * A * x ^ 2 - 4 * A) x",
    "theorem critical_points_b0 (A x : ℝ) (hA : A ≠ 0) :\n"
    "    deriv (V A 0) x = 0 ↔ x = -1 ∨ x = 0 ∨ x = 1",
    "theorem curvature_at_wells (A : ℝ) :\n"
    "    deriv (V' A 0) 1 = 8 * A ∧ deriv (V' A 0) (-1) = 8 * A",
    "theorem curvature_at_saddle (A : ℝ) : deriv (V' A 0) 0 = -4 * A",
    "theorem barrier_height_eq (A : ℝ) : V A 0 0 - V A 0 1 = A",
    "theorem potential_even_at_b0 (A x : ℝ) : V A 0 x = V A 0 (-x)",
]


def _lean_text():
    """The .lean source with normalized line endings."""
    return LEAN_SOURCE.read_text(encoding="utf-8").replace("\r\n", "\n")


def _lean_definition_as_python(name):
    """
    The right-hand side of `noncomputable def <name> (A b x : ℝ) : ℝ := ...`
    as a Python expression (Lean's ^ becomes **).
    """
    pattern = rf"noncomputable def {re.escape(name)} \(A b x : ℝ\) : ℝ := (.+)"
    match = re.search(pattern, _lean_text())
    assert match, f"definition of {name} not found in {LEAN_SOURCE}"
    return match.group(1).strip().replace("^", "**")


def test_lean_definitions_match_python_potential():
    """Lean's V and V' must give the same numbers as physics/potential.py."""
    v_expression = _lean_definition_as_python("V")
    v_prime_expression = _lean_definition_as_python("V'")

    for barrier, tilt, x in SAMPLE_POINTS:
        variables = {"A": barrier, "b": tilt, "x": x}
        lean_v = eval(v_expression, {}, variables)
        lean_v_prime = eval(v_prime_expression, {}, variables)

        assert abs(lean_v - potential(x, A=barrier, b=tilt)) < 1e-10
        assert abs(lean_v_prime - potential_derivative(x, A=barrier, b=tilt)) < 1e-10


def test_lean_theorem_statements_are_unchanged():
    """The file must contain exactly the 7 expected statements, word for word."""
    text = _lean_text()

    for statement in EXPECTED_STATEMENTS:
        assert statement + " := by" in text, f"changed or missing: {statement.splitlines()[0]}"
    assert text.count("\ntheorem ") == len(EXPECTED_STATEMENTS)


def test_stated_constants_match_python_potential():
    """
    The values the theorems state, checked on the Python side: wells at
    +-1, V'' = 8A at the wells and -4A at the saddle (by central finite
    difference of physics.potential), and barrier height A.
    """
    barrier = 1.0
    step = 1e-4

    def second_derivative(x):
        return (potential(x + step, A=barrier) - 2.0 * potential(x, A=barrier)
                + potential(x - step, A=barrier)) / step**2

    x_minus, x_plus = find_well_positions(A=barrier, b=0.0)
    assert abs(x_minus + 1.0) < 1e-10 and abs(x_plus - 1.0) < 1e-10
    assert abs(second_derivative(1.0) - 8.0 * barrier) < 1e-5
    assert abs(second_derivative(-1.0) - 8.0 * barrier) < 1e-5
    assert abs(second_derivative(0.0) - (-4.0 * barrier)) < 1e-5
    assert abs(potential(0.0, A=barrier) - potential(1.0, A=barrier) - barrier_height(A=barrier)) < 1e-12
