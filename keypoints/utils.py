from typing import Optional, Tuple
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from dataclasses import dataclass
from argparse import ArgumentParser


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


def get_config() -> Config:
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    return Config(**vars(args))
