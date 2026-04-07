import VJO.Soundness

/-!
# VJO: Block List Analysis (F4)

Formalizes block list pruning in BOA. Shows that BOA-with-block-list has a
completeness gap: any jailbreak response containing a blocked token is missed.
-/

open Finset BigOperators

/-! ## Block list definitions -/

/-- A response is blocked if it contains any token from the block list. -/
def isBlocked {V : Type*} [DecidableEq V] (BL : Finset V) (r : List V) : Prop :=
  ∃ t ∈ r, t ∈ BL

/-- BOA with block list only explores responses with no blocked tokens. -/
def exploredByBOA {V : Type*} [DecidableEq V] (BL : Finset V) (r : List V) : Prop :=
  ∀ t ∈ r, t ∉ BL

/-! ## Theorem 1: Completeness gap -/

/-- If a response is blocked, BOA-with-BL will never return it as a witness,
    even if it is a valid jailbreak above threshold. -/
theorem blocklist_completeness_gap {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (ε : ℝ)
    (BL : Finset V) (r : Response V)
    (hblocked : isBlocked BL r)
    (hthresh : responseProb M D p r ≥ jailbreakThreshold M p D ε r.length)
    (hjudge : Judge p r = true) :
    ¬ exploredByBOA BL r := by
  intro hexplored
  obtain ⟨t, ht_mem, ht_bl⟩ := hblocked
  exact hexplored t ht_mem ht_bl

/-! ## Theorem 2: Count of unblocked responses -/

/-- The number of length-n responses with all tokens outside BL is
    (|V| - |BL|)^n. -/
theorem unblocked_responses_count {V : Type*} [Fintype V] [DecidableEq V]
    (BL : Finset V) (n : ℕ) :
    {r : List V | r.length = n ∧ exploredByBOA BL r}.Finite ∧
    (∃ (hfin : {r : List V | r.length = n ∧ exploredByBOA BL r}.Finite),
      hfin.toFinset.card = (Fintype.card V - BL.card) ^ n) := by
  set S := {r : List V | r.length = n ∧ exploredByBOA BL r}
  have hfin : S.Finite := by
    apply Set.Finite.subset (fintype_bounded_lists V n)
    intro r ⟨hlen, _⟩; simp only [Set.mem_setOf_eq]; omega
  refine ⟨hfin, hfin, ?_⟩
  -- Bijection: S ≃ (Fin n → ↥(univ \ BL))
  set C := (Finset.univ \ BL : Finset V)
  let toFun : S → (Fin n → ↥C) := fun ⟨r, hlen, hmem⟩ i =>
    ⟨r.get (i.cast hlen.symm), by
      simp only [C, Finset.mem_sdiff, Finset.mem_univ, true_and]
      exact hmem _ (List.get_mem r _)⟩
  let invFun : (Fin n → ↥C) → S := fun f =>
    ⟨List.ofFn (fun i => (f i).val), List.length_ofFn .., fun t ht => by
      rw [List.mem_ofFn] at ht
      obtain ⟨i, rfl⟩ := ht
      have := (f i).prop
      simp only [C, Finset.mem_sdiff, Finset.mem_univ, true_and] at this
      exact this⟩
  have hlr : Function.LeftInverse invFun toFun := by
    intro ⟨r, hlen, hmem⟩
    simp only [invFun, toFun]; congr 1
    apply List.ext_getElem
    · simp [hlen]
    · intro i hi1 hi2; simp [List.getElem_ofFn]
  have hrl : Function.RightInverse invFun toFun := by
    intro f; simp only [toFun, invFun]; funext i; simp [List.getElem_ofFn]
  have hequiv : S ≃ (Fin n → ↥C) := ⟨toFun, invFun, hlr, hrl⟩
  haveI : Fintype S := hfin.fintype
  rw [hfin.card_toFinset, Fintype.card_congr hequiv, Fintype.card_fun, Fintype.card_fin]
  congr 1
  simp [Fintype.card_subtype, C]
  rw [Finset.filter_not]; simp [Finset.card_sdiff]

/-! ## Theorem 3: Missed fraction bound -/

/-- If BL covers fraction p of vocabulary (|BL|/|V| = p),
    the fraction of length-n responses that avoid BL is (1-p)^n. -/
theorem blocklist_missed_fraction {V : Type*} [Fintype V] [DecidableEq V]
    (BL : Finset V) (p : ℝ) (n_align : ℕ)
    (hV : 0 < Fintype.card V)
    (hp : p = BL.card / Fintype.card V) :
    (Fintype.card V - BL.card : ℝ) ^ n_align /
      (Fintype.card V : ℝ) ^ n_align = (1 - p) ^ n_align := by
  have hV' : (Fintype.card V : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  rw [hp, ← div_pow, sub_div, div_self hV']
