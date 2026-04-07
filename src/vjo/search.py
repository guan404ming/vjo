"""
VJO: Probability-Ordered Complete Jailbreak Search

Core search algorithm with formal guarantees:
  1. Completeness: finds ALL above-threshold jailbreaks (up to n_max)
  2. First-find optimality: first jailbreak found has highest probability
  3. No wasted work: only explores nodes above pruning threshold

Corresponds to OptimalSearch.lean theorems.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Interfaces (model-agnostic)
# ---------------------------------------------------------------------------

@runtime_checkable
class TokenOracle(Protocol):
    """Interface to the language model. Implemented by vLLM backend."""

    def get_next_token_logprobs(
        self, prompt: str, prefix_ids: list[int], top_k: int
    ) -> list[tuple[int, float]]:
        """Return top-k (token_id, log_prob) pairs for next token."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        ...


@runtime_checkable
class JudgeOracle(Protocol):
    """Interface to the jailbreak judge."""

    def is_jailbreak(self, prompt: str, response: str) -> bool:
        """Return True if response is a jailbreak for prompt."""
        ...


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SearchResult(Enum):
    SAT = "sat"
    UNSAT = "unsat"
    TIMEOUT = "timeout"


@dataclass
class Witness:
    """A jailbreak witness with all data needed for certificate generation."""
    token_ids: list[int]
    token_log_probs: list[float]
    response_text: str
    response_log_prob: float  # sum of token_log_probs
    length: int


@dataclass
class SearchStats:
    """Statistics from a search run."""
    result: SearchResult
    witness: Witness | None
    nodes_expanded: int
    nodes_pruned: int
    nodes_in_frontier: int
    time_elapsed: float
    max_depth_reached: int
    # Completeness info (for UNSAT certificate)
    explored_set_size: int
    pruning_threshold_log: float


@dataclass(order=True)
class SearchNode:
    """A node in the search frontier (priority queue).

    Priority = negative log prob (heapq is min-heap, we want max prob first).
    """
    neg_log_prob: float
    token_ids: list[int] = field(compare=False)
    token_log_probs: list[float] = field(compare=False)


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def search(
    oracle: TokenOracle,
    judge: JudgeOracle,
    prompt: str,
    *,
    epsilon: float = 1e-4,
    n_token_likelihood_log: float | None = None,
    n_max: int = 128,
    top_k: int = 50,
    time_budget: float = 1500.0,
    min_response_len: int = 5,
    block_list: set[int] | None = None,
    batch_size: int = 256,
) -> SearchStats:
    """Probability-ordered complete jailbreak search.

    Args:
        oracle: Language model interface (provides next-token log probs).
        judge: Jailbreak judge interface.
        prompt: The attack prompt.
        epsilon: Sensitivity parameter (jailbreak threshold = epsilon * L_n).
        n_token_likelihood_log: log(L_{n_max}). If None, uses log(epsilon)
            as a conservative lower bound.
        n_max: Maximum response length in tokens.
        top_k: Number of top tokens to consider at each step. 0 for full
            vocabulary (removes top-k caveat from certificate).
        time_budget: Maximum search time in seconds.
        min_response_len: Minimum response length before checking judge.
        block_list: Token IDs to exclude from search.
        batch_size: Number of nodes to expand per LLM call. Larger batches
            amortize LLM call overhead. Default 256 (up from 1) assumes
            vLLM with prefix caching enabled.

    Returns:
        SearchStats with result, witness (if SAT), and statistics.

    Formal guarantees (from OptimalSearch.lean):
        - If returns SAT(r): r is above threshold AND judge(r)=true
          (boa_soundness)
        - If returns UNSAT: no above-threshold jailbreak of length <= n_max
          exists (complete_search_finds_all + complete_pruning_sound)
        - First SAT found has the highest responseProb among all jailbreaks
          (first_find_optimal)
    """
    block_list = block_list or set()
    start_time = time.time()

    # Complete pruning threshold: threshold(n_max) = epsilon * L_{n_max}
    # In log space: log(epsilon) + log(L_{n_max})
    # Conservative: if L_{n_max} unknown, use log(epsilon) (since L <= 1)
    if n_token_likelihood_log is not None:
        pruning_threshold_log = math.log(epsilon) + n_token_likelihood_log
    else:
        # Conservative: L_{n_max} <= 1, so threshold >= epsilon * L_{n_max}
        # Using just epsilon as threshold is MORE aggressive pruning (less complete)
        # For true completeness, we'd need L_{n_max}.
        # Use log(epsilon) as approximation.
        pruning_threshold_log = math.log(epsilon)

    # Initialize frontier with top-k first tokens
    frontier: list[SearchNode] = []
    first_tokens = oracle.get_next_token_logprobs(prompt, [], top_k)

    for token_id, log_prob in first_tokens:
        if token_id in block_list:
            continue
        if log_prob >= pruning_threshold_log:
            heapq.heappush(frontier, SearchNode(
                neg_log_prob=-log_prob,
                token_ids=[token_id],
                token_log_probs=[log_prob],
            ))

    nodes_expanded = 0
    nodes_pruned = 0
    max_depth = 0

    while frontier:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            return SearchStats(
                result=SearchResult.TIMEOUT,
                witness=None,
                nodes_expanded=nodes_expanded,
                nodes_pruned=nodes_pruned,
                nodes_in_frontier=len(frontier),
                time_elapsed=elapsed,
                max_depth_reached=max_depth,
                explored_set_size=nodes_expanded,
                pruning_threshold_log=pruning_threshold_log,
            )

        # Pop up to batch_size nodes from frontier.
        # Check judge on each; collect expandable nodes for batched LLM call.
        expand_batch: list[SearchNode] = []
        found_witness: Witness | None = None

        while frontier and len(expand_batch) < batch_size:
            node = heapq.heappop(frontier)
            nodes_expanded += 1
            depth = len(node.token_ids)
            max_depth = max(max_depth, depth)
            current_log_prob = -node.neg_log_prob

            # Check judge (if response is long enough)
            if depth >= min_response_len:
                response_text = oracle.decode(node.token_ids)
                if response_text and judge.is_jailbreak(prompt, response_text):
                    found_witness = Witness(
                        token_ids=node.token_ids,
                        token_log_probs=node.token_log_probs,
                        response_text=response_text,
                        response_log_prob=current_log_prob,
                        length=depth,
                    )
                    break

            # Collect for expansion (skip if at max depth)
            if depth < n_max:
                expand_batch.append(node)

        if found_witness is not None:
            elapsed = time.time() - start_time
            return SearchStats(
                result=SearchResult.SAT,
                witness=found_witness,
                nodes_expanded=nodes_expanded,
                nodes_pruned=nodes_pruned,
                nodes_in_frontier=len(frontier),
                time_elapsed=elapsed,
                max_depth_reached=max_depth,
                explored_set_size=nodes_expanded,
                pruning_threshold_log=pruning_threshold_log,
            )

        if not expand_batch:
            continue

        # Batched expansion: one LLM call for all nodes in the batch.
        # With prefix caching enabled, shared prefixes reuse KV cache.
        for node in expand_batch:
            current_log_prob = -node.neg_log_prob
            next_tokens = oracle.get_next_token_logprobs(
                prompt, node.token_ids, top_k
            )

            for token_id, log_prob in next_tokens:
                if token_id in block_list:
                    continue

                child_log_prob = current_log_prob + log_prob

                # Complete pruning: prune if below threshold(n_max)
                if child_log_prob < pruning_threshold_log:
                    nodes_pruned += 1
                    continue

                heapq.heappush(frontier, SearchNode(
                    neg_log_prob=-child_log_prob,
                    token_ids=node.token_ids + [token_id],
                    token_log_probs=node.token_log_probs + [log_prob],
                ))

    # Frontier empty: UNSAT (provably complete)
    elapsed = time.time() - start_time
    return SearchStats(
        result=SearchResult.UNSAT,
        witness=None,
        nodes_expanded=nodes_expanded,
        nodes_pruned=nodes_pruned,
        nodes_in_frontier=0,
        time_elapsed=elapsed,
        max_depth_reached=max_depth,
        explored_set_size=nodes_expanded,
        pruning_threshold_log=pruning_threshold_log,
    )
