import Mathlib

/-!
# Lean oracle for `physics/known_answers.py`

This file formally states -- but does not prove -- the algebraic landscape
facts that `physics/known_answers.py` currently asserts only in Python
docstrings and hardcodes as float literals: the second-derivative curvature
at the wells (8A) and saddle (-4A), the b=0 critical-point locations
(x = -1, 0, 1), the barrier height (A), and the b=0 well symmetry (giving
ΔF=0). Every theorem below ends in `sorry`; `ax-prover` discharges the
proofs, never Claude Code (see CLAUDE.md's "Lean / ax-prover scope
boundaries" section).

Out of scope (see CLAUDE.md): the Eyring-Kramers rate formula itself
(Fokker-Planck asymptotics, not algebra), any MSM statistical-convergence
claim, 2D Allen-Cahn, and exact tilted (b != 0) well positions (irrational
cubic roots -- `scipy.optimize.brentq` is the only honest answer there).

`V_hasDerivAt` and `V'_hasDerivAt` are the two foundational theorems: they
tie the literal formulas already written in `known_answers.py`'s docstrings
to the REAL first and second derivatives of the potential, via Mathlib's
`HasDerivAt`. Every theorem after them is a corollary that must chain
through one of these two facts -- so a wrong differentiation cannot be
papered over by a correct-looking numeric substitution.
-/

namespace Oracle.Potential

/-- The (optionally tilted) double-well potential, matching
`physics/potential.py::potential`: `V(x) = A*(x^2-1)^2 + b*x`. -/
noncomputable def V (A b x : ℝ) : ℝ := A * (x ^ 2 - 1) ^ 2 + b * x

/-- `dV/dx`, matching `physics/potential.py::potential_derivative` and
`known_answers.py`'s reliance on it (e.g. `find_well_positions`, line 130):
`V'(x) = 4*A*x*(x^2-1) + b`. -/
noncomputable def V' (A b x : ℝ) : ℝ := 4 * A * x * (x ^ 2 - 1) + b

/-- `V'` really is the derivative of `V`, for any tilt `b`.
Anchors `physics/potential.py:70-101` (`potential_derivative`) as a whole. -/
theorem V_hasDerivAt (A b x : ℝ) : HasDerivAt (V A b) (V' A b x) x := by
  sorry

/-- `12*A*x^2 - 4*A` really is the derivative of `V'`, i.e. the second
derivative of `V`. Anchors `known_answers.py:58-59`, the docstring's stated
formula `V''(x) = 12*A*x**2 - 4*A`. -/
theorem V'_hasDerivAt (A b x : ℝ) : HasDerivAt (V' A b) (12 * A * x ^ 2 - 4 * A) x := by
  sorry

/-- At b=0 and A != 0, the stationary points of `V` are exactly the two
wells and the saddle: x = -1, 0, 1. A corollary of `V_hasDerivAt`, not an
independently restated fact. Anchors `known_answers.py:102-134`
(`find_well_positions`'s b=0 case: wells at exactly +-1, barrier at 0). -/
theorem critical_points_b0 (A x : ℝ) (hA : A ≠ 0) :
    deriv (V A 0) x = 0 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

/-- Curvature at both wells is exactly 8A. A corollary of `V'_hasDerivAt`.
Anchors `known_answers.py:60,93` (`curvature_at_well = 8.0 * A`). -/
theorem curvature_at_wells (A : ℝ) :
    deriv (V' A 0) 1 = 8 * A ∧ deriv (V' A 0) (-1) = 8 * A := by
  sorry

/-- Curvature at the saddle is exactly -4A. A corollary of `V'_hasDerivAt`.
Anchors `known_answers.py:61,94` (`curvature_at_barrier = 4.0 * A`, used as
`|V''(0)|`). -/
theorem curvature_at_saddle (A : ℝ) : deriv (V' A 0) 0 = -4 * A := by
  sorry

/-- The barrier (at x=0) sits exactly A above the well at x=1. Pure algebra
on `V`'s definition, no derivative involved. Anchors `known_answers.py:41-47,
64,95` (`barrier_height()`, `delta_V = A`). -/
theorem barrier_height_eq (A : ℝ) : V A 0 0 - V A 0 1 = A := by
  sorry

/-- At b=0, `V` is an even function of x. Pure algebra on `V`'s definition.
Anchors `known_answers.py:137-160` (`free_energy_difference`'s docstring:
"for b=0, deltaF is exactly zero by the x -> -x symmetry of V"). -/
theorem potential_even_at_b0 (A x : ℝ) : V A 0 x = V A 0 (-x) := by
  sorry

end Oracle.Potential
