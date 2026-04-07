import Mathlib

/-!
# VJO: Verified Jailbreak Oracle — Core Definitions (F1)

Formalizes core types and Definitions 1–3 from:
  "LLM Jailbreak Oracle" (arXiv:2506.17299, Lin et al. 2025)

Work Package F1: all definitions type-checked, no sorry.
-/

open BigOperators

/-! ## Basic types -/

/-- A prompt is a sequence of tokens from vocabulary V. -/
abbrev Prompt (V : Type*) := List V

/-- A response is a sequence of tokens from vocabulary V. -/
abbrev Response (V : Type*) := List V

/-! ## Language model -/

/-- A language model assigns a conditional next-token probability given a prompt and a
    response pfx. Non-negativity and normalisation are enforced structurally. -/
structure Model (V : Type*) [Fintype V] where
  /-- P_M(v | p, pfx) : probability of token v given prompt p and response pfx. -/
  prob     : Prompt V → Response V → V → ℝ
  nonneg   : ∀ p pfx v, 0 ≤ prob p pfx v
  sum_one  : ∀ p pfx, ∑ v : V, prob p pfx v = 1

/-! ## Decoding strategy -/

/-- A decoding strategy selects a nonempty set of eligible tokens at each step.
    - Top-k: the k tokens with highest probability under P_M.
    - Top-p (nucleus): the minimal set of tokens with cumulative mass ≥ p. -/
structure DecodingStrategy (V : Type*) [Fintype V] where
  eligible : Model V → Prompt V → Response V → Finset V
  nonempty : ∀ M p pfx, (eligible M p pfx).Nonempty

/-! ## Response probability -/

/-- P_D(v | M, p, pfx): strategy-conditional probability of token v, renormalized
    over the eligible set D.eligible M p pfx.
    Tokens outside the eligible set have probability 0. -/
noncomputable def strategyCondProb {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (pfx : Response V) (v : V) : ℝ :=
  let Z := ∑ u ∈ D.eligible M p pfx, M.prob p pfx u
  if v ∈ D.eligible M p pfx then M.prob p pfx v / Z else 0

/-- Pr_D(r | M, p): probability of generating response r under decoding strategy D.
    Computed as ∏_{t=0}^{|r|-1} P_D(r_t | M, p, r_{<t}), where P_D is the
    strategy-conditional probability renormalized over the eligible token set.
    Corresponds to Pr_D(r|M,p) in arXiv:2506.17299. -/
noncomputable def responseProb {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (D : DecodingStrategy V) (p : Prompt V) (r : Response V) : ℝ :=
  (List.ofFn fun i : Fin r.length =>
    strategyCondProb M D p (r.take i.val) (r.get i)).prod

/-! ## Judge (axiomatized oracle) -/

/-- J(p, r): returns true iff response r constitutes a jailbreak for prompt p.
    Axiomatized as an opaque oracle; not further specified. -/
axiom Judge {V : Type*} : Prompt V → Response V → Bool

/-! ## Definition 1: n-Token Response Likelihood -/

/-- L_n(M, p, D): the expected probability of the first n tokens of any response
    of length ≥ n, sampled from decoding strategy D given model M and prompt p.

    Formally (arXiv:2506.17299, Def. 1):
      L_n(M, p, D) = 𝔼_{r ~ D(M,p,≥n)} [Pr_D(r_{1:n} | M, p)]

    Axiomatized for F1; a measure-theoretic formulation is deferred to the full paper. -/
axiom nTokenLikelihood {V : Type*} [Fintype V]
    (M : Model V) (p : Prompt V) (D : DecodingStrategy V) (n : ℕ) : ℝ

axiom nTokenLikelihood_nonneg {V : Type*} [Fintype V]
    (M : Model V) (p : Prompt V) (D : DecodingStrategy V) (n : ℕ) :
    0 ≤ nTokenLikelihood M p D n

axiom nTokenLikelihood_le_one {V : Type*} [Fintype V]
    (M : Model V) (p : Prompt V) (D : DecodingStrategy V) (n : ℕ) :
    nTokenLikelihood M p D n ≤ 1

/-! ## Definition 2: Jailbreak Likelihood Threshold -/

/-- τ(n) = ε · L_n(M, p, D)

    (arXiv:2506.17299, Def. 2): the jailbreak likelihood threshold for response length n,
    where ε ∈ (0, 1] is a user-provided sensitivity parameter. -/
noncomputable def jailbreakThreshold {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (p : Prompt V) (D : DecodingStrategy V) (ε : ℝ) (n : ℕ) : ℝ :=
  ε * nTokenLikelihood M p D n

/-! ## Definition 3: Jailbreak Oracle Problem -/

/-- The Jailbreak Oracle Problem (arXiv:2506.17299, Def. 3).

    Asks whether there exists a response r̂ such that:
      (1) Pr(r̂ | M, p) ≥ τ(|r̂|)   [sufficient generation likelihood], and
      (2) J(p, r̂) = true             [judged as a jailbreak].

    BOA returns Sat(r̂) when such an r̂ is found; Unsat otherwise. -/
def jailbreakOracleProblem {V : Type*} [Fintype V] [DecidableEq V]
    (M : Model V) (p : Prompt V) (D : DecodingStrategy V) (ε : ℝ) : Prop :=
  ∃ r : Response V,
    responseProb M D p r ≥ jailbreakThreshold M p D ε r.length ∧
    Judge p r = true
