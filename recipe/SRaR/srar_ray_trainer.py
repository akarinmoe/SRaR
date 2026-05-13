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
SRaR Ray Trainer: extends RayDAPOTrainer to pass step-level rubric data
to the advantage estimator, and logs train generations to SwanLab.
"""

import numpy as np

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, compute_response_mask

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

# Register the custom advantage estimators
import recipe.SRaR.srar_advantage  # noqa: F401


def compute_srar_advantage_wrapper(
    data: DataProto,
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    """Wrapper for compute_advantage that passes non_tensor_batch for SRaR."""
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
    adv_kwargs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "config": config,
    }
    if "uid" in data.non_tensor_batch:
        adv_kwargs["index"] = data.non_tensor_batch["uid"]
    adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
    adv_kwargs["batch"] = data.batch

    advantages, returns = adv_estimator_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def log_train_generations(tokenizer, batch, step, loggers):
    """Log 10 train generation samples to SwanLab/wandb."""
    try:
        n_log = 10
        prompts = batch.batch["prompts"]
        responses = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        prompt_length = prompts.shape[-1]
        scores = batch.batch["token_level_rewards"].sum(dim=-1).tolist()
        bsz = len(prompts)

        indices = np.random.RandomState(step).choice(bsz, size=min(n_log, bsz), replace=False)

        rows = []
        for idx in indices:
            valid_prompt_len = int(attention_mask[idx][:prompt_length].sum())
            valid_prompt_ids = prompts[idx][-valid_prompt_len:]
            valid_resp_len = int(attention_mask[idx][prompt_length:].sum())
            valid_resp_ids = responses[idx][:valid_resp_len]

            prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = tokenizer.decode(valid_resp_ids, skip_special_tokens=True)
            score = scores[idx]
            rows.append([step, prompt_str, response_str, round(score, 3)])

        if "swanlab" in loggers:
            import swanlab
            table = swanlab.echarts.Table()
            table.add(headers=["step", "prompt", "response", "score"], rows=rows)
            swanlab.log({"train/generations": table}, step=step)

        if "wandb" in loggers:
            import wandb
            if wandb.run is not None:
                tbl = wandb.Table(columns=["step", "prompt", "response", "score"], data=rows)
                wandb.log({"train/generations": tbl}, step=step)

    except Exception as e:
        print(f"[TrainGenLog] Error: {e}")


def extract_rubric_metrics(batch):
    """Extract rubric score metrics from batch non_tensor_batch."""
    metrics = {}
    for key in ["suggest_score", "pitfall_score", "bonus_score"]:
        if key in batch.non_tensor_batch:
            vals = batch.non_tensor_batch[key]
            try:
                vals = [float(v) for v in vals]
                metrics[f"rubric/{key}/mean"] = np.mean(vals)
            except (ValueError, TypeError):
                pass
    return metrics


class RaySRaRTrainer(RayDAPOTrainer):
    """Trainer for SRaR with step-level rubric rewards and train generation logging."""

    def fit(self):
        import recipe.dapo.dapo_ray_trainer as dapo_trainer_module
        from verl.trainer.ppo import ray_trainer as ppo_trainer_module
        from verl.trainer.ppo import reward as reward_module
        from verl.utils.tracking import Tracking

        original_compute_adv = ppo_trainer_module.compute_advantage
        ppo_trainer_module.compute_advantage = compute_srar_advantage_wrapper
        dapo_trainer_module.compute_advantage = compute_srar_advantage_wrapper

        original_tracking_log = Tracking.log
        trainer_self = self
        self._current_train_batch = None
        self._prefilter_acc_values = []

        # Patch extract_reward to capture pre-filter accuracy
        original_extract_reward = dapo_trainer_module.extract_reward

        def extract_reward_with_acc_capture(batch):
            result = original_extract_reward(batch)
            reward_tensor, reward_extra_infos_dict = result
            if reward_extra_infos_dict and "acc" in reward_extra_infos_dict:
                acc_vals = reward_extra_infos_dict["acc"]
                trainer_self._prefilter_acc_values.extend(
                    [float(v) for v in acc_vals]
                )
            return result

        dapo_trainer_module.extract_reward = extract_reward_with_acc_capture

        def patched_tracking_log(tracking_self, data, step, **kwargs):
            if hasattr(trainer_self, '_current_train_batch') and trainer_self._current_train_batch is not None:
                rubric_metrics = extract_rubric_metrics(trainer_self._current_train_batch)
                data.update(rubric_metrics)
                log_train_generations(
                    trainer_self.tokenizer,
                    trainer_self._current_train_batch,
                    step,
                    trainer_self.config.trainer.logger,
                )
                trainer_self._current_train_batch = None
            # Add pre-filter accuracy
            if trainer_self._prefilter_acc_values:
                data["reward/accuracy"] = np.mean(trainer_self._prefilter_acc_values)
                trainer_self._prefilter_acc_values = []
            original_tracking_log(tracking_self, data, step, **kwargs)

        Tracking.log = patched_tracking_log

        original_compute_adv_wrapper = compute_srar_advantage_wrapper

        def compute_adv_with_capture(data, adv_estimator, **kwargs):
            trainer_self._current_train_batch = data
            return original_compute_adv_wrapper(data, adv_estimator, **kwargs)

        ppo_trainer_module.compute_advantage = compute_adv_with_capture
        dapo_trainer_module.compute_advantage = compute_adv_with_capture

        try:
            RayDAPOTrainer.fit(self)
        finally:
            ppo_trainer_module.compute_advantage = original_compute_adv
            dapo_trainer_module.compute_advantage = original_compute_adv
            dapo_trainer_module.extract_reward = original_extract_reward
            Tracking.log = original_tracking_log


class RayRaRTrainer(RayDAPOTrainer):
    """Trainer for RaR with train generation logging."""

    def fit(self):
        import recipe.dapo.dapo_ray_trainer as dapo_trainer_module
        from verl.trainer.ppo import ray_trainer as ppo_trainer_module
        from verl.utils.tracking import Tracking

        original_tracking_log = Tracking.log
        trainer_self = self
        trainer_self._current_train_batch = None
        trainer_self._prefilter_acc_values = []

        # Patch extract_reward to capture pre-filter accuracy
        original_extract_reward = dapo_trainer_module.extract_reward

        def extract_reward_with_acc_capture(batch):
            result = original_extract_reward(batch)
            reward_tensor, reward_extra_infos_dict = result
            if reward_extra_infos_dict and "acc" in reward_extra_infos_dict:
                acc_vals = reward_extra_infos_dict["acc"]
                trainer_self._prefilter_acc_values.extend(
                    [float(v) for v in acc_vals]
                )
            return result

        dapo_trainer_module.extract_reward = extract_reward_with_acc_capture

        def patched_tracking_log(tracking_self, data, step, **kwargs):
            if trainer_self._current_train_batch is not None:
                rubric_metrics = extract_rubric_metrics(trainer_self._current_train_batch)
                data.update(rubric_metrics)
                log_train_generations(
                    trainer_self.tokenizer,
                    trainer_self._current_train_batch,
                    step,
                    trainer_self.config.trainer.logger,
                )
                trainer_self._current_train_batch = None
            # Add pre-filter accuracy
            if trainer_self._prefilter_acc_values:
                data["reward/accuracy"] = np.mean(trainer_self._prefilter_acc_values)
                trainer_self._prefilter_acc_values = []
            original_tracking_log(tracking_self, data, step, **kwargs)

        Tracking.log = patched_tracking_log

        from verl.trainer.ppo.ray_trainer import compute_advantage as original_compute_adv

        def compute_adv_with_capture(data, adv_estimator, **kwargs):
            trainer_self._current_train_batch = data
            return original_compute_adv(data, adv_estimator, **kwargs)

        ppo_trainer_module.compute_advantage = compute_adv_with_capture
        dapo_trainer_module.compute_advantage = compute_adv_with_capture

        try:
            RayDAPOTrainer.fit(self)
        finally:
            ppo_trainer_module.compute_advantage = original_compute_adv
            dapo_trainer_module.compute_advantage = original_compute_adv
            dapo_trainer_module.extract_reward = original_extract_reward
            Tracking.log = original_tracking_log
