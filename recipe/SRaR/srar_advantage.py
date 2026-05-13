# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SRaR advantage estimator: decoupled advantage = outcome advantage + step-wise rubric offset.

Implements Algorithm 1 from "Step-wise Rubric Rewards for LLM Reasoning":
- Outcome advantage: standard GRPO normalization of r_base across rollouts of same prompt
- Rubric offset: per-step cross-rollout normalized rubric deltas, broadcast to step tokens

For RaR: uses standard GRPO advantage (rubric reward is already in the total reward scalar).
"""

import json
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("srar")
def compute_srar_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    non_tensor_batch: dict = None,
    batch: dict = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SRaR decoupled advantage: outcome + step-wise rubric offset.

    The advantage for token t in rollout i is:
        A^(t)_i = (r_base,i - μ_base) / (σ_base + ε) + r̃^(t)_i

    where r̃^(t)_i is the cross-rollout normalized rubric delta for the step containing token t.

    Args:
        token_level_rewards: shape (bs, response_length), contains r_base at last token
        response_mask: shape (bs, response_length)
        index: uid array for grouping rollouts by prompt
        non_tensor_batch: dict with "step_rubric_deltas" and "step_token_spans"
        batch: dict with batch tensors
    """
    bsz, seq_len = token_level_rewards.shape

    # Part 1: Compute outcome advantage (standard GRPO on r_base)
    scores = token_level_rewards.sum(dim=-1)  # r_base for each rollout

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    for i in range(bsz):
        id2scores[index[i]].append(scores[i].item())
        id2indices[index[i]].append(i)

    outcome_advantages = torch.zeros(bsz, dtype=torch.float32)
    with torch.no_grad():
        for uid, score_list in id2scores.items():
            scores_t = torch.tensor(score_list)
            mu = scores_t.mean()
            std = scores_t.std() if len(score_list) > 1 else torch.tensor(1.0)
            for local_idx, global_idx in enumerate(id2indices[uid]):
                if norm_adv_by_std_in_grpo:
                    outcome_advantages[global_idx] = (scores_t[local_idx] - mu) / (std + epsilon)
                else:
                    outcome_advantages[global_idx] = scores_t[local_idx] - mu

    # Part 2: Compute step-wise rubric offset with cross-rollout normalization
    rubric_offset = torch.zeros(bsz, seq_len, dtype=torch.float32)

    if non_tensor_batch is not None and "step_rubric_deltas" in non_tensor_batch:
        step_deltas_strs = non_tensor_batch["step_rubric_deltas"]
        step_spans_strs = non_tensor_batch["step_token_spans"]

        # Parse all step deltas and spans
        all_step_deltas = []
        all_step_spans = []
        for i in range(bsz):
            try:
                deltas = json.loads(step_deltas_strs[i]) if step_deltas_strs[i] else {}
            except (json.JSONDecodeError, TypeError):
                deltas = {}
            try:
                spans = json.loads(step_spans_strs[i]) if step_spans_strs[i] else []
            except (json.JSONDecodeError, TypeError):
                spans = []
            all_step_deltas.append(deltas)
            all_step_spans.append(spans)

        # Cross-rollout normalization per step (Eq. 4 in paper)
        # Group by prompt (uid), then for each step k, normalize d_{k,i} across rollouts
        for uid, indices_in_group in id2indices.items():
            if len(indices_in_group) <= 1:
                continue

            # Collect per-step deltas across rollouts of this prompt
            # step_key -> list of (rollout_index, delta_value)
            step_to_rollout_deltas = defaultdict(list)
            for global_idx in indices_in_group:
                deltas = all_step_deltas[global_idx]
                for step_key, delta_val in deltas.items():
                    step_to_rollout_deltas[step_key].append((global_idx, delta_val))

            # Normalize each step's deltas across rollouts
            for step_key, rollout_deltas in step_to_rollout_deltas.items():
                if len(rollout_deltas) <= 1:
                    # Single sample provides no relative signal -> set to 0
                    continue

                values = torch.tensor([d for _, d in rollout_deltas])
                mu_k = values.mean()
                sigma_k = values.std()

                for global_idx, delta_val in rollout_deltas:
                    if sigma_k > epsilon:
                        normalized = (delta_val - mu_k.item()) / (sigma_k.item() + epsilon)
                    else:
                        normalized = 0.0

                    # Assign normalized delta to the tokens of the corresponding step
                    step_num = int(step_key)
                    spans = all_step_spans[global_idx]

                    # Find the span for this step
                    # step_key from judge is 1-indexed, spans list is 0-indexed
                    # step 0 means whole solution, step -1 means no relevant step
                    if step_num == 0:
                        # Spans the whole solution
                        rubric_offset[global_idx, :] += normalized * response_mask[global_idx]
                    elif step_num == -1:
                        # No relevant step, skip
                        continue
                    elif 1 <= step_num <= len(spans):
                        span = spans[step_num - 1]
                        start_tok = int(span[0])
                        end_tok = int(span[1])
                        rubric_offset[global_idx, start_tok:end_tok] += normalized
                    else:
                        # Step number out of range, assign to whole solution
                        rubric_offset[global_idx, :] += normalized * response_mask[global_idx]

    # Combine: A^(t)_i = outcome_advantage_i + rubric_offset^(t)_i
    # Broadcast outcome advantage to all tokens
    advantages = outcome_advantages.unsqueeze(-1) * response_mask + rubric_offset * response_mask

    return advantages, advantages


@register_adv_est("rar")
def compute_rar_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RaR advantage: standard GRPO on combined (r_base + rubric_reward).

    For RaR, the rubric reward is already aggregated into the token_level_rewards
    as a single trajectory-level scalar. So we just use standard GRPO normalization.
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
