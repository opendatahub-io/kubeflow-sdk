# Copyright 2024 The Kubeflow Authors.
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

"""Progress tracking callback for Transformers Trainer - Template for code injection.

Note: This template file references ProgressMetricsHandler which is provided at runtime.
Linting errors for undefined names are expected and suppressed.
"""

# ruff: noqa: F821

import json
import time
from typing import Optional

from transformers import TrainerCallback


class KubeflowProgressCallback(TrainerCallback):
    """Callback that tracks training progress and exposes via HTTP metrics."""

    def __init__(self) -> None:
        """Initialize progress callback."""
        self.start_time: Optional[float] = None

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> Optional[str]:
        """Format seconds into human-readable duration.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string like "2h 15m 30s" or None.
        """
        if seconds is None or seconds < 0:
            return None

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    @staticmethod
    def _format_timestamp(timestamp: Optional[float]) -> Optional[str]:
        """Format Unix timestamp to ISO 8601 string.

        Args:
            timestamp: Unix timestamp.

        Returns:
            ISO 8601 formatted string or None.
        """
        if timestamp is None:
            return None
        from datetime import datetime

        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """Called when training begins."""
        self.start_time = time.time()

        ProgressMetricsHandler.update_metrics(
            {
                "status": "training",
                "status_message": "Training started",
                "status_details": {
                    "last_event": "training_started",
                    "last_event_time": self._format_timestamp(self.start_time),
                },
                "progress": {
                    "step_current": state.global_step,
                    "step_total": state.max_steps,
                    "percent": 0.0,
                    "epoch": state.epoch if hasattr(state, "epoch") else 0,
                },
                "time": {
                    "started_sec": self.start_time,
                    "started_at": self._format_timestamp(self.start_time),
                    "updated_sec": self.start_time,
                    "updated_at": self._format_timestamp(self.start_time),
                },
            }
        )

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Called at the end of each training step."""
        current_time = time.time()
        current_step = state.global_step
        total_steps = state.max_steps

        # Calculate progression percentage
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0

        # Calculate time metrics
        elapsed_sec = current_time - self.start_time if self.start_time else 0
        remaining_sec = None
        if current_step > 0 and elapsed_sec > 0:
            estimated_total_time = elapsed_sec / (current_step / total_steps)
            remaining_sec = estimated_total_time - elapsed_sec

        # Generate meaningful status message
        remaining_str = self._format_duration(remaining_sec) if remaining_sec else "calculating..."
        status_msg = (
            f"Training in progress: {progress_pct:.1f}% complete, {remaining_str} remaining"
        )

        ProgressMetricsHandler.update_metrics(
            {
                "status_message": status_msg,
                "progress": {
                    "step_current": current_step,
                    "step_total": total_steps,
                    "percent": round(progress_pct, 2),
                    "epoch": state.epoch,
                },
                "time": {
                    "elapsed_sec": round(elapsed_sec, 2),
                    "elapsed": self._format_duration(elapsed_sec),
                    "remaining_sec": round(remaining_sec, 2) if remaining_sec else None,
                    "remaining": self._format_duration(remaining_sec) if remaining_sec else None,
                    "updated_sec": current_time,
                    "updated_at": self._format_timestamp(current_time),
                },
            }
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when Trainer logs metrics."""
        if logs:
            metrics_update = {}

            # Track standard metrics
            if "loss" in logs:
                metrics_update["loss"] = logs["loss"]
            if "learning_rate" in logs:
                metrics_update["learning_rate"] = logs["learning_rate"]
            if "train_samples_per_second" in logs:
                metrics_update["throughput_samples_sec"] = logs["train_samples_per_second"]

            # Track custom metrics
            custom_metrics_map = __CUSTOM_METRICS_PLACEHOLDER__  # noqa: F821
            if custom_metrics_map:
                for log_key, metric_name in custom_metrics_map.items():
                    if log_key in logs:
                        metrics_update[metric_name] = logs[log_key]

            if metrics_update:
                current_time = time.time()
                ProgressMetricsHandler.update_metrics(
                    {
                        "metrics": metrics_update,
                        "time": {
                            "updated_sec": current_time,
                            "updated_at": self._format_timestamp(current_time),
                        },
                    }
                )

    def on_train_end(self, args, state, control, **kwargs) -> None:
        """Called when training ends."""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        current_metrics = json.loads(ProgressMetricsHandler.get_metrics_json())
        final_step = current_metrics.get("progress", {}).get("step_current", 0)
        status_msg = (
            f"Training completed successfully at step {final_step} in "
            f"{self._format_duration(total_time)}"
        )

        ProgressMetricsHandler.update_metrics(
            {
                "status": "completed",
                "status_message": status_msg,
                "status_details": {
                    "last_event": "training_completed",
                    "last_event_time": self._format_timestamp(end_time),
                },
                "progress": {
                    "percent": 100.0,
                },
                "time": {
                    "elapsed_sec": round(total_time, 2),
                    "elapsed": self._format_duration(total_time),
                    "remaining_sec": 0,
                    "remaining": "0s",
                    "updated_sec": end_time,
                    "updated_at": self._format_timestamp(end_time),
                },
            }
        )
