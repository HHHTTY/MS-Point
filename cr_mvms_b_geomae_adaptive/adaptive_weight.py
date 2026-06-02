"""Adaptive GeoMAE Weight Controller.

Core innovation of CR-MVMS-B-GeoMAE-Adaptive experiment.

Problem (from CR-MVMS-B-GeoMAE):
  - Fixed MAE weight (0.50) gives only 3.5-7% loss ratio
  - As contrastive loss decreases during training, the ratio changes unpredictably
  - Need dynamic adjustment to maintain target gradient ratio (~10%)

Solution:
  - PID-like proportional controller using loss-value ratio as proxy
  - EMA smoothing for stable ratio estimation
  - Warmup phase follows original schedule, then hands off to controller
  - Multiplicative weight update for smooth transitions

Key design decisions:
  1. Loss ratio = recon_weighted / contrast_total (no extra forward pass needed)
  2. EMA alpha=0.3 for responsive yet stable tracking
  3. Adjustment every 100 steps (not every step, to avoid oscillation)
  4. Weight bounds [0.10, 3.0] prevent instability
  5. Controller learning rate=0.02 (conservative, proportional to error)
"""

import math
from typing import Dict, List, Optional


class AdaptiveGeoMAEWeight:
    """Adaptive weight controller for GeoMAE reconstruction loss.

    Usage:
        controller = AdaptiveGeoMAEWeight(target_ratio=0.10, schedule_cfg={...})

        # In training loop:
        for epoch in range(epochs):
            for step, batch in enumerate(loader):
                w = controller.get_weight(epoch)
                loss, logs = adaptive_loss(outputs, cfg, epoch, geo_mae_weight_override=w)
                loss.backward()

                if controller.should_update(step):
                    ratio = logs["loss_ratio_recon"]
                    new_w, adjusted = controller.update(ratio, epoch, step)
                    if adjusted:
                        print(f"Adaptive weight -> {new_w:.4f}")
    """

    def __init__(
        self,
        target_ratio: float = 0.10,
        adjustment_freq: int = 100,
        lr: float = 0.3,
        w_min: float = 0.10,
        w_max: float = 3.0,
        ema_alpha: float = 0.3,
        warmup_epochs: int = 40,
        initial_weight: float = 0.50,
        schedule_cfg: Optional[Dict] = None,
    ):
        self.target_ratio = target_ratio
        self.adjustment_freq = adjustment_freq
        self.lr = lr
        self.w_min = w_min
        self.w_max = w_max
        self.ema_alpha = ema_alpha
        self.warmup_epochs = warmup_epochs
        self.current_weight = initial_weight
        self.schedule_cfg = schedule_cfg or {}

        # State tracking
        self.ema_ratio: Optional[float] = None
        self.global_step = 0
        self.adjustment_history: List[Dict] = []
        self.ratio_history: List[float] = []

        # For detecting ratio trends
        self._consecutive_low = 0
        self._consecutive_high = 0

    def _scheduled_weight(self, epoch: int) -> float:
        """Compute scheduled weight during warmup phase.

        Uses the same cosine interpolation as the original CR-MVMS-B-GeoMAE.
        """
        cfg = self.schedule_cfg
        ramp_start = int(cfg.get("ramp_start", 0))
        ramp_end = int(cfg.get("ramp_end", 40))
        start_w = float(cfg.get("start_weight", 0.0))
        end_w = float(cfg.get("end_weight", 0.50))

        if epoch <= ramp_start:
            return start_w
        elif epoch >= ramp_end:
            return end_w
        else:
            p = (epoch - ramp_start) / max(1, ramp_end - ramp_start)
            return start_w + (end_w - start_w) * (1 - math.cos(p * math.pi)) / 2

    def is_warmup(self, epoch: int) -> bool:
        """Check if we're still in warmup phase."""
        return epoch < self.warmup_epochs

    def get_weight(self, epoch: int) -> float:
        """Get current GeoMAE weight for this epoch.

        During warmup: returns scheduled weight (same as original).
        After warmup: returns adaptive weight from controller.
        """
        if self.is_warmup(epoch):
            return self._scheduled_weight(epoch)
        return self.current_weight

    def should_update(self, step_in_epoch: int) -> bool:
        """Check if controller should update at this step.

        Only updates every adjustment_freq steps after warmup.
        """
        return self.global_step > 0 and self.global_step % self.adjustment_freq == 0

    def update(self, current_ratio: float, epoch: int, step: int) -> tuple:
        """Update controller based on current loss ratio.

        Args:
            current_ratio: Current loss ratio (recon_weighted / contrast_total)
            epoch: Current epoch
            step: Current step (within epoch)

        Returns:
            (new_weight, was_adjusted) tuple
        """
        self.global_step += 1

        # During warmup, don't adjust
        if self.is_warmup(epoch):
            return self.current_weight, False

        # Track ratio history (always, for monitoring)
        self.ratio_history.append(current_ratio)

        # Update EMA
        if self.ema_ratio is None:
            self.ema_ratio = current_ratio
        else:
            self.ema_ratio = (
                self.ema_alpha * current_ratio
                + (1 - self.ema_alpha) * self.ema_ratio
            )

        # Only adjust every adjustment_freq steps
        if self.global_step % self.adjustment_freq != 0:
            return self.current_weight, False

        # Direct target weight calculation:
        # If current ratio = ema_ratio with weight = current_weight,
        # then target weight = current_weight * (target_ratio / ema_ratio)
        if self.ema_ratio > 1e-6:
            target_weight = self.current_weight * (self.target_ratio / self.ema_ratio)
            target_weight = max(self.w_min, min(self.w_max, target_weight))
        else:
            # Ratio essentially zero, aggressively increase weight
            target_weight = min(self.current_weight * 2.0, self.w_max)

        # Blend towards target (smooth adjustment, lr controls convergence speed)
        # With lr=0.3, converges in ~5-8 adjustments
        new_weight = self.current_weight + self.lr * (target_weight - self.current_weight)

        # Clamp to bounds
        new_weight = max(self.w_min, min(self.w_max, new_weight))

        # Check if adjustment is meaningful
        old_weight = self.current_weight
        relative_change = abs(new_weight - old_weight) / max(old_weight, 1e-8)
        was_adjusted = relative_change > 0.001  # Log only if >0.1% change

        error = self.target_ratio - self.ema_ratio

        if was_adjusted:
            self.adjustment_history.append({
                "global_step": self.global_step,
                "epoch": epoch,
                "step": step,
                "old_weight": round(old_weight, 6),
                "new_weight": round(new_weight, 6),
                "target_weight": round(target_weight, 6),
                "ema_ratio": round(self.ema_ratio, 6),
                "current_ratio": round(current_ratio, 6),
                "error": round(error, 6),
            })

        self.current_weight = new_weight
        return new_weight, was_adjusted

    def get_state(self) -> Dict:
        """Get controller state for checkpoint saving."""
        return {
            "current_weight": self.current_weight,
            "ema_ratio": self.ema_ratio,
            "global_step": self.global_step,
            "adjustment_history": self.adjustment_history[-100:],  # Keep last 100
            "ratio_history": self.ratio_history[-500:],  # Keep last 500
            "_consecutive_low": self._consecutive_low,
            "_consecutive_high": self._consecutive_high,
        }

    def load_state(self, state: Dict):
        """Load controller state from checkpoint."""
        self.current_weight = state.get("current_weight", self.current_weight)
        self.ema_ratio = state.get("ema_ratio", None)
        self.global_step = state.get("global_step", 0)
        self.adjustment_history = state.get("adjustment_history", [])
        self.ratio_history = state.get("ratio_history", [])
        self._consecutive_low = state.get("_consecutive_low", 0)
        self._consecutive_high = state.get("_consecutive_high", 0)

    def summary(self) -> str:
        """Return a summary string of controller state."""
        recent_ratios = self.ratio_history[-20:] if self.ratio_history else [0]
        avg_ratio = sum(recent_ratios) / len(recent_ratios)
        n_adj = len(self.adjustment_history)
        ema_str = f"{self.ema_ratio:.4f}" if self.ema_ratio is not None else "N/A"
        return (
            f"AdaptiveGeoMAEWeight: "
            f"w={self.current_weight:.4f} "
            f"ema_ratio={ema_str} "
            f"avg20_ratio={avg_ratio:.4f} "
            f"target={self.target_ratio:.2f} "
            f"adjustments={n_adj} "
            f"step={self.global_step}"
        )
