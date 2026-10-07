import Mathlib

/-!
# Lean oracle for `physics/known_answers.py`

This file formally states the algebraic landscape facts that
`physics/known_answers.py` currently asserts only in Python docstrings and
hardcodes as float literals: the second-derivative curvature at the wells
(8A) and saddle (-4A), the b=0 critical-point locations (x = -1, 0, 1), the
barrier height (A), and the b=0 well symmetry (giving ΔF=0). Five of the
seven theorems below are still unproved placeholders. Claude Code writes
statements only, never proof bodies (see CLAUDE.md's "Lean / ax-prover
scope boundaries" section); the intended prover is `ax-prover`. Two
theorems are fully proved:
- `potential_even_at_b0` (pure algebra, no derivative), by `ax-prover`,
  2026-07-30.
- `V_hasDerivAt`, by the external tool Lemma (output dated 2026-09-15), at
  the human's direction and outside the ax-prover workflow, after
  ax-prover failed on it 11 times and its run ran out of API credits. The
  statement is unchanged; only the proof body was added. Checked
  2026-09-25 on this project's pinned toolchain (v4.33.0-rc1, Mathlib
  9d302fc per `lake-manifest.json`): `#print axioms` lists only the three
  standard axioms [propext, Classical.choice, Quot.sound]. Log:
  `results/lean_v_hasDerivAt_check.log`; details in PROJECT_STATE.md §9
  (2026-09-25 entry).

Note for editors: this module docstring deliberately never spells out
Lean's placeholder keyword. ax-prover 0.1.1 treats the `import Mathlib`
block, which includes this docstring, as a declaration, and queues it as
an unproven theorem named `Mathlib` if that keyword appears here.

Out of scope (see CLAUDE.md): the Eyring-Kramers rate formula itself
(Fokker-Planck asymptotics, not algebra), any MSM statistical-convergence
claim, 2D Allen-Cahn, and exact tilted (b != 0) well positions (irrational
cubic roots -- `scipy.optimize.brentq` is the only honest answer there).

`V_hasDerivAt` and `V'_hasDerivAt` are the two foundational theorems: they
tie the literal formulas already written in `known_answers.py`'s docstrings
to the REAL first and second derivatives of the potential, via Mathlib's
`HasDerivAt`. The derivative facts below (`critical_points_b0`,
`curvature_at_wells`, `curvature_at_saddle`) are stated with Mathlib's
`deriv`, not with the hand-written formulas, so a wrong hand
differentiation cannot be papered over by a correct-looking numeric
substitution. Their intended proofs go through `V_hasDerivAt` (critical
points) and `V'_hasDerivAt` (curvatures); Lean does not force that route.
`barrier_height_eq` and `potential_even_at_b0` are pure algebra on `V`'s
definition and use neither.
-/

namespace Oracle.Potential

/-- The (optionally tilted) double-well potential, matching
`physics/potential.py::potential`: `V(x) = A*(x^2-1)^2 + b*x`. -/
noncomputable def V (A b x : ℝ) : ℝ := A * (x ^ 2 - 1) ^ 2 + b * x

/-- `dV/dx`, matching `physics/potential.py::potential_derivative`:
`V'(x) = 4*A*x*(x^2-1) + b`. -/
noncomputable def V' (A b x : ℝ) : ℝ := 4 * A * x * (x ^ 2 - 1) + b

/-- `V'` really is the derivative of `V`, for any tilt `b`. Anchors
`physics/potential.py::potential_derivative`. -/
theorem V_hasDerivAt (A b x : ℝ) : HasDerivAt (V A b) (V' A b x) x := by
  -- Restate the goal with `V A b` unfolded to its lambda. `show` accepts
  -- any definitionally equal goal, and `V` is an ordinary `def`, so this
  -- is just delta-unfolding for readability: the final `exact` would also
  -- accept the folded goal, since everything below matches it up to defeq.
  show HasDerivAt (fun y : ℝ => A * (y ^ 2 - 1) ^ 2 + b * y) (V' A b x) x
  -- Differentiate `y ^ 2 - 1` first. `hasDerivAt_pow 2 x` gives the value
  -- `↑2 * x ^ (2 - 1)`, which `simpa` normalizes to `2 * x`; subtracting
  -- the constant `1` (`.sub_const 1`) leaves the derivative unchanged.
  have hsq : HasDerivAt (fun y : ℝ => y ^ 2 - 1) (2 * x) x := by
    simpa using (hasDerivAt_pow 2 x).sub_const 1
  -- Power rule composed with `hsq`: `HasDerivAt.pow` gives the value
  -- `↑n * f x ^ (n - 1) * f'` for the pointwise power `f ^ n`. The type
  -- ascription below is accepted up to defeq (`((2 : ℕ) : ℝ)` is `2`, and
  -- `2 - 1` reduces to `1` in ℕ); `pow_one` removes the leftover `^ 1`
  -- in `hval` below.
  have hsq2 : HasDerivAt (fun y : ℝ => (y ^ 2 - 1) ^ 2)
      (2 * (x ^ 2 - 1) ^ 1 * (2 * x)) x := hsq.pow 2
  -- Scale by the constant `A`.
  have hAterm : HasDerivAt (fun y : ℝ => A * (y ^ 2 - 1) ^ 2)
      (A * (2 * (x ^ 2 - 1) ^ 1 * (2 * x))) x := hsq2.const_mul A
  -- The linear tilt term `b * y` has constant derivative `b`.
  have hbterm : HasDerivAt (fun y : ℝ => b * y) b x := by
    simpa using (hasDerivAt_id x).const_mul b
  -- Sum the two pieces. `HasDerivAt.add` is stated for the pointwise sum
  -- `f + g`, so `hsum`'s function is definitionally (not syntactically)
  -- the unfolded `V A b`; the final `exact` closes the goal by that defeq.
  have hsum := hAterm.add hbterm
  -- Massage the combinator-derived derivative value into the literal
  -- form `V' A b x = 4*A*x*(x^2-1) + b` by pure ring arithmetic.
  have hval : A * (2 * (x ^ 2 - 1) ^ 1 * (2 * x)) + b = V' A b x := by
    simp only [V', pow_one]
    ring
  rw [hval] at hsum
  exact hsum

/-- `12*A*x^2 - 4*A` really is the derivative of `V'`, i.e. the second
derivative of `V`. Anchors the formula `V''(x) = 12*A*x**2 - 4*A` in
`known_answers.eyring_kramers_rate_0d`'s docstring. -/
theorem V'_hasDerivAt (A b x : ℝ) : HasDerivAt (V' A b) (12 * A * x ^ 2 - 4 * A) x := by
  sorry

/-- At b=0 and A != 0, the stationary points of `V` are exactly x = -1, 0, 1
(two wells and the saddle when A > 0). Intended as a corollary of
`V_hasDerivAt`. Anchors `known_answers.find_well_positions` (wells at +-1
for b=0). -/
theorem critical_points_b0 (A x : ℝ) (hA : A ≠ 0) :
    deriv (V A 0) x = 0 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by
  sorry

/-- The derivative of `V'` at both wells is 8A. Together with `V_hasDerivAt`
this is V''(+-1) = 8A. Anchors `curvature_at_well = 8.0 * A` in
`known_answers.eyring_kramers_rate_0d`. -/
theorem curvature_at_wells (A : ℝ) :
    deriv (V' A 0) 1 = 8 * A ∧ deriv (V' A 0) (-1) = 8 * A := by
  sorry

/-- The derivative of `V'` at the saddle is -4A, i.e. V''(0) = -4A.
Anchors `curvature_at_barrier = 4.0 * A` (used as |V''(0)|) in
`known_answers.eyring_kramers_rate_0d`. -/
theorem curvature_at_saddle (A : ℝ) : deriv (V' A 0) 0 = -4 * A := by
  sorry

/-- The barrier (at x=0) sits exactly A above the well at x=1. Pure algebra
on `V`'s definition. Anchors `known_answers.barrier_height`. -/
theorem barrier_height_eq (A : ℝ) : V A 0 0 - V A 0 1 = A := by
  sorry

/-- At b=0, `V` is an even function of x. Pure algebra on `V`'s definition.
Anchors `known_answers.energy_difference_of_minima` (exactly 0 for b=0). -/
theorem potential_even_at_b0 (A x : ℝ) : V A 0 x = V A 0 (-x) := by
  rw [V, V, neg_sq, zero_mul, zero_mul]

end Oracle.Potential
