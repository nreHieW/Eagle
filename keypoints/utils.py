from typing import Optional, Tuple
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from dataclasses import dataclass
from argparse import ArgumentParser
import math
from torch.optim.lr_scheduler import _LRScheduler


class RelativeEarlyStopping(EarlyStopping):
    """Slightly modified Early Stopping Callback that allows to set a relative threshold (loss * threshold)
    hope this will be integrated in Lightning one day: https://github.com/Lightning-AI/lightning/issues/12094


    """

    def __init__(
        self,
        monitor: Optional[str] = None,
        min_relative_delta: float = 0.01,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
    ):
        super().__init__(monitor, min_relative_delta, patience, verbose, mode, strict)

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = ""

        # the (1-delta)* current is the only change.
        if self.monitor_op(current * (1 - self.min_delta), self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records." f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."

        return should_stop, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = f"Metric {self.monitor} improved {abs(self.best_score - current)/self.best_score:.3f} times >=" f" min_delta = {abs(self.min_delta)}. New best score: {current:.7f}"
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.7f}"
        return msg


@dataclass
class Config:
    run_name: str
    project_name: str
    seed: int = 1337
    lr: float = 3e-4
    precision: str = "bf16-mixed"
    num_epochs: int = 200
    batch_size: int = 16
    heatmap_sigma: int = 3
    use_calibrated: bool = False


def get_config() -> Config:
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--heatmap_sigma", type=int, default=3)
    parser.add_argument("--use_calibrated", action="store_true")
    args = parser.parse_args()
    return Config(**vars(args))


class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, num_iterations, learning_rate, decay_frac, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.decay_frac = decay_frac
        self.min_lr = self.learning_rate * self.decay_frac
        super(WarmupCosineDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        if it < self.warmup_iters:
            return [self.learning_rate * (it + 1) / self.warmup_iters for _ in self.optimizer.param_groups]
        if it > self.num_iterations:
            return [self.min_lr for _ in self.optimizer.param_groups]
        decay_ratio = (it - self.warmup_iters) / (self.num_iterations - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return [self.min_lr + coeff * (self.learning_rate - self.min_lr) for _ in self.optimizer.param_groups]
