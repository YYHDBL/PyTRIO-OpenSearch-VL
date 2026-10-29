"""作者token级PPO裁剪目标，供PyTRIO custom loss使用。"""

import math

import torch


def make_ppo_loss(metadata, trajectories):
    if trajectories < 1 or not metadata:
        raise ValueError("损失需要原始轨迹分母和有效训练样本")
    frozen = tuple({**m, "old": tuple(m["old"])} for m in metadata)
    for m in frozen:
        if m["normalizer"] < 1 or m["start"] < 0 or not m["old"] or not math.isfinite(m["advantage"]):
            raise ValueError("PPO元数据无效")

    def loss_fn(data, logprobs):
        if not len(data) == len(logprobs) == len(frozen):
            raise ValueError("样本、logprob和PPO元数据数量不同")
        losses = []
        for current, m in zip(logprobs, frozen, strict=True):
            start, length = m["start"], len(m["old"])
            active = current.reshape(-1)[start:start + length]
            old = torch.tensor(m["old"], dtype=current.dtype, device=current.device)
            if active.numel() != length or not torch.isfinite(active).all() or not torch.isfinite(old).all():
                raise ValueError("PPO有效token区间或数值无效")
            ratio = (active - old).clamp(-20, 20).exp()
            advantage = m["advantage"]
            objective = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(0.8, 1.28))
            if advantage < 0:
                objective = torch.minimum(objective, torch.full_like(objective, -advantage * 3.0))
            losses.append(objective.sum() / (trajectories * m["normalizer"]))
        loss = torch.stack(losses).sum()
        if not torch.isfinite(loss):
            raise ValueError("PPO损失不是有限值")
        return loss, {"rl/ppo_loss": float(loss.detach()), "rl/trajectory_denominator": trajectories}

    return loss_fn
