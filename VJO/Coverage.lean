import VJO.Soundness

/-!
# VJO: Coverage Bounds (F3)

Formalizes coverage bounds for BOA's search:
- F3.1: totalAboveThreshold is finite
- F3.2: coverage lower bound (BOA explores min(budget, N) nodes)
- F3.3: unsat confidence (missed fraction <= delta)
-/

open BigOperators Finset

/-! ## Definitions -/

/-- The set of responses of length exactly n above the jailbreak threshold. -/
def aboveThreshold {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ) (n : ℕ) : Set (List V) :=
  {r | r.length = n ∧ responseProb M D p r ≥ jailbreakThreshold M p D ε n}

/-- Total above-threshold nodes up to depth n_max. -/
def totalAboveThreshold {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ) (n_max : ℕ) : Set (List V) :=
  {r | r.length ≤ n_max ∧ responseProb M D p r ≥ jailbreakThreshold M p D ε r.length}

/-! ## F3.1: totalAboveThreshold is finite -/

theorem totalAboveThreshold_finite {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ) (n_max : ℕ) :
    (totalAboveThreshold M D p ε n_max).Finite :=
  above_threshold_finite M D p ε n_max

/-! ## F3.2: Coverage lower bound -/

/-- Coverage lower bound: given budget T, BOA can explore at least
    min(T, N) nodes where N = |totalAboveThreshold M D p ε n_max|.
    The explored set is a subset of totalAboveThreshold. -/
theorem coverage_lower_bound {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ)
    (n_max budget : ℕ) :
    let hfin := totalAboveThreshold_finite M D p ε n_max
    ∃ explored : Finset (List V),
      (∀ r ∈ explored, r ∈ totalAboveThreshold M D p ε n_max) ∧
      explored.card = min budget hfin.toFinset.card := by
  intro hfin
  obtain ⟨t, ht_sub, ht_card⟩ :=
    Finset.exists_subset_card_eq (Nat.min_le_right budget hfin.toFinset.card)
  exact ⟨t, fun r hr => hfin.mem_toFinset.mp (ht_sub hr), by omega⟩

/-! ## F3.3: Unsat confidence -/

/-- If explored covers fraction (1-delta) of above-threshold nodes,
    then at most delta fraction were missed. -/
theorem unsat_confidence
    (N : ℕ) (δ : ℝ) (explored_card : ℕ)
    (hcoverage : (1 - δ) * N ≤ explored_card) :
    (N : ℝ) - explored_card ≤ δ * N := by
  nlinarith

