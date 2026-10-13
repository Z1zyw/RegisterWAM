from __future__ import annotations

from typing import Iterable

import torch


def _to_python_list(values) -> list:
    if isinstance(values, torch.Tensor):
        return values.tolist()
    if isinstance(values, list):
        out = []
        for value in values:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    out.append(value.item())
                else:
                    out.append(value.tolist())
            else:
                out.append(value)
        return out
    if isinstance(values, tuple):
        return _to_python_list(list(values))
    return list(values)


def _gather_hf_rows(hf_dataset, indices: Iterable[int]):
    return hf_dataset[list(indices)]


def _patched_get_query_timestamps(self, current_ts: float, query_indices: dict[str, list[int]] | None = None) -> dict[str, list[float]]:
    query_timestamps = {}
    sparse_offsets = getattr(self, "_register_wam_sparse_video_offsets", None)
    for key in self.meta.video_keys:
        if query_indices is not None and key in query_indices:
            absolute_indices = list(query_indices[key])
            if sparse_offsets is not None:
                absolute_indices = [absolute_indices[i] for i in sparse_offsets]
            timestamps = _gather_hf_rows(self.hf_dataset, absolute_indices)["timestamp"]
            query_timestamps[key] = _to_python_list(timestamps)
        else:
            query_timestamps[key] = [current_ts]
    return query_timestamps


def _patched_query_hf_dataset_fast(self, query_indices: dict[str, list[int]]) -> dict:
    result = {}
    processed_indices = set()
    index_to_selected = {}
    for key, q_idx in query_indices.items():
        if key in self.meta.video_keys:
            continue
        if "images" in key and not self.during_training:
            continue
        q_idx_tuple = tuple(q_idx)
        if q_idx_tuple not in processed_indices:
            selected_data = _gather_hf_rows(self.hf_dataset, q_idx)
            index_to_selected[q_idx_tuple] = selected_data
            processed_indices.add(q_idx_tuple)
        else:
            selected_data = index_to_selected[q_idx_tuple]
        result[key] = torch.stack(selected_data[key])
    return result


def apply_fast_lerobot_patches(multi_dataset, video_sample_indices: list[int] | None = None) -> None:
    sparse_offsets = None if video_sample_indices is None else tuple(int(i) for i in video_sample_indices)
    for dataset in multi_dataset._datasets:
        cls = dataset.__class__
        cls._get_query_timestamps = _patched_get_query_timestamps
        cls._query_hf_dataset_fast = _patched_query_hf_dataset_fast
        dataset._register_wam_sparse_video_offsets = sparse_offsets
