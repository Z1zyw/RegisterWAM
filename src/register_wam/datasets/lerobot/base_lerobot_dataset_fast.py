from __future__ import annotations

from typing import Optional

import numpy as np

from .base_lerobot_dataset import BaseLerobotDataset, MAX_GETITEM_ATTEMPT, logger


class BaseLerobotDatasetFast(BaseLerobotDataset):
    def __init__(self, *args, log_retry_traceback: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_retry_traceback = bool(log_retry_traceback)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds {len(self)}.")

        sample_idx = idx
        attempt = 0
        last_exception: Optional[Exception] = None
        while attempt < MAX_GETITEM_ATTEMPT:
            try:
                lerobot_sample = self.multi_dataset[sample_idx]
                lerobot_sample = self._split_lerobot_sample(lerobot_sample)
                break
            except Exception as err:
                attempt += 1
                last_exception = err
                logger.warning(
                    "Error loading sample %s (attempt %s/%s). Retrying with a random index. Error: %s",
                    sample_idx,
                    attempt,
                    MAX_GETITEM_ATTEMPT,
                    err,
                )
                if self.log_retry_traceback:
                    logger.exception("Detailed dataset retry traceback")
                sample_idx = int(np.random.randint(len(self)))
        else:
            raise RuntimeError(
                f"Failed to load a valid sample after {MAX_GETITEM_ATTEMPT} attempts for index {idx}."
            ) from last_exception

        sample = {
            "idx": sample_idx,
            "task": lerobot_sample["task"],
            "action": {},
            "state": {},
            "images": {},
        }
        for meta in self.state_meta:
            sample["state"][meta["key"]] = self._get_state(meta, lerobot_sample)

        for meta in self.action_meta:
            sample["action"][meta["key"]] = self._get_action(meta, lerobot_sample)

        for meta in self.image_meta:
            sample["images"][meta["key"]] = self._get_image(meta, lerobot_sample)

        sample["action_is_pad"] = lerobot_sample[f"{self.action_meta[0]['lerobot_key']}_is_pad"]
        sample["state_is_pad"] = lerobot_sample[f"{self.state_meta[0]['lerobot_key']}_is_pad"]
        sample["image_is_pad"] = lerobot_sample[f"{self.image_meta[0]['lerobot_key']}_is_pad"]

        sample = self._get_additional_data(sample, lerobot_sample)

        for key in lerobot_sample:
            if key not in sample and "observation" not in key and "action" not in key:
                sample[key] = lerobot_sample[key]

        if self.processor is not None:
            sample = self.processor.preprocess(sample)

        return sample
