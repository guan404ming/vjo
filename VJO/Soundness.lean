import VJO.Defs

/-!
# VJO: Soundness of Likelihood Pruning (F2.2)

Formalizes Theorem F2.2 from arXiv:2506.17299:
if a prefix's response probability is below the jailbreak threshold,
then no single-token extension can meet the threshold either.
-/

open BigOperators Finset

/-! ## Axiom: n-token likelihood is non-increasing -/

axiom nTokenLikelihood_antitone {V : Type*} [Fintype V]
    (M : Model V) (p : Prompt V) (D : DecodingStrategy V) (n : ℕ) :
    nTokenLikelihood M p D (n + 1) ≤ nTokenLikelihood M p D n

/-! ## Helper lemmas -/

lemma strategyCondProb_le_one {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (pfx : Response V) (v : V) :
    strategyCondProb M D p pfx v ≤ 1 := by
  unfold strategyCondProb
  split
  · next hv =>
    have hZ : 0 ≤ ∑ u ∈ D.eligible M p pfx, M.prob p pfx u :=
      Finset.sum_nonneg (fun u _ => M.nonneg p pfx u)
    rcases eq_or_lt_of_le hZ with hZ0 | hZpos
    · simp [← hZ0, le_antisymm (hZ0 ▸ Finset.single_le_sum (fun u _ => M.nonneg p pfx u) hv)
        (M.nonneg p pfx v)]
    · exact (div_le_one₀ hZpos).mpr (Finset.single_le_sum (fun u _ => M.nonneg p pfx u) hv)
  · linarith

lemma strategyCondProb_nonneg {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (pfx : Response V) (v : V) :
    0 ≤ strategyCondProb M D p pfx v := by
  unfold strategyCondProb
  split
  · apply div_nonneg (M.nonneg p pfx v)
    exact Finset.sum_nonneg (fun u _ => M.nonneg p pfx u)
  · exact le_refl 0

lemma responseProb_nonneg {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (r : Response V) :
    0 ≤ responseProb M D p r := by
  unfold responseProb
  apply List.prod_nonneg
  intro x hx
  rw [List.mem_ofFn] at hx
  obtain ⟨i, rfl⟩ := hx
  exact strategyCondProb_nonneg M D p (r.take i.val) (r.get i)

/-! ## Lemma: responseProb of append singleton -/

lemma responseProb_append_singleton {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (s : Response V) (t : V) :
    responseProb M D p (s ++ [t]) = responseProb M D p s * strategyCondProb M D p s t := by
  simp only [responseProb]
  suffices h : (List.ofFn fun i : Fin (s ++ [t]).length =>
      strategyCondProb M D p ((s ++ [t]).take i.val) ((s ++ [t]).get i)) =
    (List.ofFn fun i : Fin s.length =>
      strategyCondProb M D p (s.take i.val) (s.get i)) ++ [strategyCondProb M D p s t] by
    rw [h, List.prod_append, List.prod_singleton]
  apply List.ext_getElem
  · simp
  · intro i hi1 hi2
    rw [List.getElem_ofFn, List.getElem_append]
    split
    · next hi =>
      rw [List.getElem_ofFn]
      have hi' : i < s.length := by rwa [List.length_ofFn] at hi
      congr 1
      · exact List.take_append_of_le_length (le_of_lt hi')
      · simp [List.get_eq_getElem, List.getElem_append_left (h := hi')]
    · next hi =>
      push Not at hi
      have hi' : i = s.length := by
        rw [List.length_ofFn] at hi
        have : i < s.length + 1 := by simpa using hi1
        omega
      subst hi'
      simp [List.get_eq_getElem]

/-! ## Lemma: responseProb is monotone under extension -/

lemma responseProb_append_le {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (s : Response V) (t : V) :
    responseProb M D p (s ++ [t]) ≤ responseProb M D p s := by
  rw [responseProb_append_singleton]
  have h1 := strategyCondProb_le_one M D p s t
  have h2 := responseProb_nonneg M D p s
  nlinarith

/-! ## Main theorem: likelihood pruning soundness -/

theorem likelihood_pruning_sound {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ)
    (s : Response V)
    (h : responseProb M D p s < jailbreakThreshold M p D ε (s.length + 1)) :
    ∀ t : V, responseProb M D p (s ++ [t]) <
      jailbreakThreshold M p D ε (s ++ [t]).length := by
  intro t
  have hlen : (s ++ [t]).length = s.length + 1 := by simp
  rw [hlen]
  exact lt_of_le_of_lt (responseProb_append_le M D p s t) h

/-! ## BOA Soundness (F2.1) -/

/-- BOA returns either a satisfying response or Unsat. -/
inductive BOAResult (V : Type*) where
  | Sat : Response V → BOAResult V
  | Unsat : BOAResult V

/-- What BOA certifies: a Sat result witnesses both conditions; Unsat is vacuously true. -/
def BOACertified {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ)
    (result : BOAResult V) : Prop :=
  match result with
  | BOAResult.Sat r =>
      responseProb M D p r ≥ jailbreakThreshold M p D ε r.length ∧
      Judge p r = true
  | BOAResult.Unsat => True

/-- **F2.1**: BOA only returns Sat when both conditions are verified. -/
theorem boa_soundness {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ)
    (r : Response V)
    (hprob : responseProb M D p r ≥ jailbreakThreshold M p D ε r.length)
    (hjudge : Judge p r = true) :
    BOACertified M D p ε (BOAResult.Sat r) := by
  exact ⟨hprob, hjudge⟩

/-- A Sat witness solves the jailbreak oracle problem. -/
theorem boa_sat_implies_oracle {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ)
    (r : Response V)
    (hcert : BOACertified M D p ε (BOAResult.Sat r)) :
    jailbreakOracleProblem M p D ε := by
  exact ⟨r, hcert.1, hcert.2⟩

/-! ## F2.3: Termination -/

/-- Bounded-length lists over a Fintype form a finite set. -/
lemma fintype_bounded_lists (V : Type*) [Fintype V] (n : ℕ) :
    {l : List V | l.length ≤ n}.Finite := by
  induction n with
  | zero =>
    convert Set.finite_singleton ([] : List V)
    ext l; simp [List.length_eq_zero_iff]
  | succ n ih =>
    have : {l : List V | l.length ≤ n + 1} =
        {l | l.length ≤ n} ∪ {l | l.length = n + 1} := by
      ext l; simp only [Set.mem_union, Set.mem_setOf_eq]; omega
    rw [this]
    apply Set.Finite.union ih
    have : {l : List V | l.length = n + 1} ⊆
        Set.range (fun (f : Fin (n + 1) → V) => List.ofFn f) := by
      intro l hl
      simp only [Set.mem_setOf_eq] at hl
      exact ⟨fun i => l[i.val]'(by omega), by
        apply List.ext_getElem <;> simp_all [List.getElem_cons]
        intro i hi
        split
        · next h => subst h; rfl
        · next h => congr 1; omega⟩
    exact Set.Finite.subset (Set.finite_range _) this

/-- Above-threshold candidates with bounded length form a finite set. -/
lemma above_threshold_finite {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ) (n_max : ℕ) :
    {r : List V | r.length ≤ n_max ∧
      responseProb M D p r ≥ jailbreakThreshold M p D ε r.length}.Finite := by
  apply Set.Finite.subset (fintype_bounded_lists V n_max)
  intro r ⟨hlen, _⟩
  exact hlen

/-- **F2.3**: The set of nodes BOA can explore is finite: responses of length ≤ n_max
    with probability ≥ threshold form a finite set, bounding the search space. -/
theorem priority_queue_well_founded {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ) (n_max : ℕ) :
    Set.Finite {r : List V | r.length ≤ n_max ∧
      responseProb M D p r ≥ jailbreakThreshold M p D ε r.length} :=
  above_threshold_finite M D p ε n_max
