from __future__ import annotations

import torch

from .robot_video_dataset_fast import RobotVideoDatasetFast


class RobotVideoDatasetFastRaw(RobotVideoDatasetFast):
    """Fast dataset variant that keeps composed video at raw spatial resolution.

    This preserves the output field structure of RobotVideoDatasetFast while skipping
    the final resize/crop/normalize path on CPU. The model is responsible for the
    GPU-side preprocessing.
    """

    def _finalize_video(self, video: torch.Tensor) -> torch.Tensor:
        return video.permute(1, 0, 2, 3).contiguous()

