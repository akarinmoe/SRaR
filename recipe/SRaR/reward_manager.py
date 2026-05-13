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
RaR and SRaR Reward Managers conforming to the verl experimental reward loop API.
Implements LLM judge-based rubric evaluation following the paper
"Step-wise Rubric Rewards for LLM Reasoning".
"""

import inspect
import json
import os
import re
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

JUDGE_PROMPT_TEMPLATE = """Problem
{problem}

Rubric Items
{rubric_items}

Student Solution
{response}

Student's Final Answer
{extracted_answer}

Task
For every rubric item listed above, evaluate the student's solution.
Rules:
• SUGGEST → satisfied: true if the student correctly performed that reasoning step.
• PITFALL → satisfied: true if the student made that mistake (fell into the pitfall).
• BONUS → satisfied: true if the student used that approach.
For step: the 1-indexed step number (### Step N) most closely associated with this item. Use 0 if it spans the whole solution; use -1 if there is no relevant step.
Return ONLY a valid JSON array:
[
  {{"id": <int>, "satisfied": <bool>, "step": <int>}},
  ...
]"""


def parse_steps(response: str) -> list[tuple[int, int]]:
    """Parse step boundaries from a response with ### Step N: headers."""
    pattern = r"###\s*Step\s*\d+\s*:"
    matches = list(re.finditer(pattern, response))
    if not matches:
        return [(0, len(response))]
    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
        spans.append((start, end))
    return spans


def parse_rubric_items(rubric_text: str) -> list[dict]:
    """Parse rubric items from text format."""
    items = []
    pattern = r"<(SUGGEST|PITFALL|BONUS|ANSWER)>\s*(.+)"
    for idx, line in enumerate(rubric_text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        m = re.match(pattern, line)
        if m:
            items.append({"id": idx + 1, "type": m.group(1), "text": m.group(2).strip()})
    return items


def extract_boxed_answer(response: str) -> str:
    """Extract answer from \\boxed{} in response. Handles nested braces."""
    idx = response.rfind("\\boxed{")
    if idx < 0:
        return ""
    i = idx + len("\\boxed{")
    depth = 1
    while i < len(response) and depth > 0:
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return response[idx + len("\\boxed{"):i - 1]
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace(" ", "")
    answer = answer.replace("$", "")
    answer = re.sub(r"(\\text\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*?)(\})", r"\2", answer)
    answer = re.sub(r"(\\frac)([^{])(.)", r"frac{\2}{\3}", answer)
    return answer.strip()


def compute_score_boxed(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs):
    """Compute score by extracting answer from \\boxed{} and comparing to ground truth."""
    pred = extract_boxed_answer(solution_str)
    pred_normalized = normalize_answer(pred)
    gt_normalized = normalize_answer(str(ground_truth))

    correct = pred_normalized == gt_normalized if pred_normalized else False

    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred,
    }


def check_format(response: str) -> bool:
    """Check if response contains step headers and \\boxed{} answer."""
    has_steps = bool(re.search(r"###\s*Step\s*\d+\s*:", response))
    has_boxed = bool(re.search(r"\\boxed\{", response))
    return has_steps and has_boxed


async def call_judge_async(loop, problem, response, extracted_answer, rubric_items, judge_url, judge_model):
    """Call LLM judge asynchronously."""
    import openai

    rubric_text = ""
    for item in rubric_items:
        rubric_text += f"  {item['id']}. <{item['type']}> {item['text']}\n"

    # Truncate response to avoid exceeding judge model context
    truncated_response = response[:12000] if len(response) > 12000 else response

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        rubric_items=rubric_text,
        response=truncated_response,
        extracted_answer=extracted_answer,
    )

    def _call():
        import httpx
        http_client = httpx.Client(proxy=None, verify=False)
        client = openai.OpenAI(
            base_url=judge_url,
            api_key=os.environ.get("JUDGE_API_KEY", "EMPTY"),
            http_client=http_client,
        )
        try:
            completion = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8192,
            )
            content = completion.choices[0].message.content
            if content is None:
                # Print more debug info
                finish_reason = completion.choices[0].finish_reason if completion.choices else "no_choices"
                print(f"[RubricJudge] None content, finish_reason={finish_reason}, prompt_len={len(prompt)}")
                return [{"id": item["id"], "satisfied": False, "step": 0} for item in rubric_items]
            content = content.strip()
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and all(isinstance(r, dict) for r in parsed):
                    return parsed
        except Exception as e:
            print(f"[RubricJudge] Error: {e}")
        return [{"id": item["id"], "satisfied": False, "step": 0} for item in rubric_items]

    return await loop.run_in_executor(None, _call)


def compute_rubric_deltas(judge_results, rubric_items, r_sug=0.8, r_pit=-1.0, r_bon=1.0):
    """Compute per-step raw rubric deltas from judge results (Eq. 3).

    Returns (step_deltas, type_scores) where type_scores has suggest/pitfall/bonus sums.
    """
    from collections import defaultdict

    type_counts = defaultdict(int)
    for item in rubric_items:
        if item["type"] != "ANSWER":
            type_counts[item["type"]] += 1

    id2item = {item["id"]: item for item in rubric_items}
    step_deltas = defaultdict(float)
    type_scores = {"suggest_score": 0.0, "pitfall_score": 0.0, "bonus_score": 0.0}

    for result in judge_results:
        if not isinstance(result, dict):
            continue
        item_id = result.get("id")
        satisfied = result.get("satisfied", False)
        step = result.get("step", 0)

        if item_id not in id2item:
            continue
        item = id2item[item_id]
        item_type = item["type"]

        if item_type == "ANSWER":
            continue
        if not satisfied:
            continue

        if item_type == "SUGGEST":
            delta = r_sug / max(type_counts["SUGGEST"], 1)
            type_scores["suggest_score"] += delta
        elif item_type == "PITFALL":
            delta = -abs(r_pit) / max(type_counts["PITFALL"], 1)
            type_scores["pitfall_score"] += delta
        elif item_type == "BONUS":
            delta = r_bon / max(type_counts["BONUS"], 1)
            type_scores["bonus_score"] += delta
        else:
            delta = 0.0

        step_deltas[step] += delta

    return dict(step_deltas), type_scores


class RaRRewardManager(RewardManagerBase):
    """RaR (Rubrics as Rewards) Reward Manager.

    Computes a trajectory-level rubric reward aggregated into a single scalar.
    Total reward = (1-λ)*r_acc + λ*r_fmt + rubric_sum
    """

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score_boxed
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        reward_kwargs = config.reward.get("reward_kwargs", {})
        self.judge_url = reward_kwargs.get("judge_url", os.environ.get("JUDGE_URL", "http://localhost:8000/v1"))
        self.judge_model = reward_kwargs.get("judge_model", os.environ.get("JUDGE_MODEL", ""))
        self.r_sug = reward_kwargs.get("r_sug", 0.8)
        self.r_pit = reward_kwargs.get("r_pit", -1.0)
        self.r_bon = reward_kwargs.get("r_bon", 1.0)
        self.format_weight = reward_kwargs.get("format_weight", 0.1)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1
        data_item = data[0]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        prompt_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        )
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # Compute accuracy score
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source, solution_str=response_str,
                ground_truth=ground_truth, extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None, lambda: self.compute_score(
                    data_source=data_source, solution_str=response_str,
                    ground_truth=ground_truth, extra_info=extra_info,
                ),
            )

        if isinstance(result, dict):
            acc_score = result["score"]
        else:
            acc_score = result

        r_acc = 1.0 if acc_score > 0 else 0.0
        r_fmt = 1.0 if check_format(response_str) else 0.0
        lam = self.format_weight
        r_base = (1 - lam) * r_acc + lam * r_fmt

        # Compute rubric reward
        rubric_text = extra_info.get("rubric", "")
        rubric_reward = 0.0
        type_scores = {"suggest_score": 0.0, "pitfall_score": 0.0, "bonus_score": 0.0}

        if rubric_text:
            rubric_items = parse_rubric_items(rubric_text)
            if rubric_items:
                extracted_answer = extract_boxed_answer(response_str)
                judge_results = await call_judge_async(
                    self.loop, prompt_str, response_str, extracted_answer,
                    rubric_items, self.judge_url, self.judge_model,
                )
                step_deltas, type_scores = compute_rubric_deltas(
                    judge_results, rubric_items,
                    r_sug=self.r_sug, r_pit=self.r_pit, r_bon=self.r_bon,
                )
                rubric_reward = sum(step_deltas.values())

        total_reward = r_base + rubric_reward

        reward_extra_info = {
            "acc": r_acc,
            "format": r_fmt,
            "rubric_reward": rubric_reward,
            "suggest_score": type_scores.get("suggest_score", 0.0),
            "pitfall_score": type_scores.get("pitfall_score", 0.0),
            "bonus_score": type_scores.get("bonus_score", 0.0),
            "score": total_reward,
        }
        return {"reward_score": total_reward, "reward_extra_info": reward_extra_info}


class SRaRRewardManager(RewardManagerBase):
    """SRaR (Step-wise Rubrics as Rewards) Reward Manager.

    Computes base reward as the token-level reward score.
    Stores per-step rubric deltas and step token spans in reward_extra_info
    for the decoupled advantage estimator.
    """

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score_boxed
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        reward_kwargs = config.reward.get("reward_kwargs", {})
        self.judge_url = reward_kwargs.get("judge_url", os.environ.get("JUDGE_URL", "http://localhost:8000/v1"))
        self.judge_model = reward_kwargs.get("judge_model", os.environ.get("JUDGE_MODEL", ""))
        self.r_sug = reward_kwargs.get("r_sug", 0.8)
        self.r_pit = reward_kwargs.get("r_pit", -1.0)
        self.r_bon = reward_kwargs.get("r_bon", 1.0)
        self.format_weight = reward_kwargs.get("format_weight", 0.1)

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1
        data_item = data[0]
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum())
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        prompt_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        )
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # Compute accuracy score
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source, solution_str=response_str,
                ground_truth=ground_truth, extra_info=extra_info,
            )
        else:
            result = await self.loop.run_in_executor(
                None, lambda: self.compute_score(
                    data_source=data_source, solution_str=response_str,
                    ground_truth=ground_truth, extra_info=extra_info,
                ),
            )

        if isinstance(result, dict):
            acc_score = result["score"]
        else:
            acc_score = result

        r_acc = 1.0 if acc_score > 0 else 0.0
        r_fmt = 1.0 if check_format(response_str) else 0.0
        lam = self.format_weight
        r_base = (1 - lam) * r_acc + lam * r_fmt

        # Compute step-wise rubric deltas
        rubric_text = extra_info.get("rubric", "")
        step_deltas = {}
        step_token_spans = []
        type_scores = {"suggest_score": 0.0, "pitfall_score": 0.0, "bonus_score": 0.0}

        if rubric_text:
            rubric_items = parse_rubric_items(rubric_text)
            if rubric_items:
                extracted_answer = extract_boxed_answer(response_str)
                judge_results = await call_judge_async(
                    self.loop, prompt_str, response_str, extracted_answer,
                    rubric_items, self.judge_url, self.judge_model,
                )
                step_deltas, type_scores = compute_rubric_deltas(
                    judge_results, rubric_items,
                    r_sug=self.r_sug, r_pit=self.r_pit, r_bon=self.r_bon,
                )

        # Compute step token spans
        step_spans_char = parse_steps(response_str)
        total_chars = len(response_str)
        total_tokens = valid_response_length
        if total_chars > 0:
            for start_char, end_char in step_spans_char:
                start_token = int(start_char / total_chars * total_tokens)
                end_token = int(end_char / total_chars * total_tokens)
                start_token = max(0, min(start_token, total_tokens))
                end_token = max(start_token, min(end_token, total_tokens))
                step_token_spans.append([start_token, end_token])
        else:
            step_token_spans = [[0, total_tokens]]

        # Convert step_deltas keys to strings for JSON serialization
        step_deltas_str = {str(k): v for k, v in step_deltas.items()}

        reward_extra_info = {
            "acc": r_acc,
            "format": r_fmt,
            "score": r_base,
            "suggest_score": type_scores.get("suggest_score", 0.0),
            "pitfall_score": type_scores.get("pitfall_score", 0.0),
            "bonus_score": type_scores.get("bonus_score", 0.0),
            "step_rubric_deltas": json.dumps(step_deltas_str),
            "step_token_spans": json.dumps(step_token_spans),
        }
        return {"reward_score": r_base, "reward_extra_info": reward_extra_info}
