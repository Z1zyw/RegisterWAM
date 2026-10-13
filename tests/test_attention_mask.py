import torch

from register_wam.models.wan22.register_wam import RegisterWAM


def test_action_reads_only_registers():
    dummy = type("Dummy", (), {"register_num_tokens": 2, "visual_attention_mode": "bidirectional"})()
    mask = RegisterWAM._register_attention_mask(
        dummy,
        augmented_video_length=10,
        action_length=3,
        obs_length=4,
        device=torch.device("cpu"),
    )

    action_rows = mask[10:, :]
    assert action_rows[:, 4:6].all()
    assert not action_rows[:, :4].any()
    assert not action_rows[:, 6:10].any()
    assert action_rows[:, 10:].all()


def test_bidirectional_preserves_legacy_mask_and_auxiliary_isolation():
    dummy = type("Dummy", (), {"register_num_tokens": 2, "visual_attention_mode": "bidirectional"})()
    kwargs = dict(augmented_video_length=12, action_length=3, obs_length=4,
                  noised_obs_length=2, device=torch.device("cpu"))
    mask = RegisterWAM._register_attention_mask(dummy, **kwargs)
    dummy.visual_attention_mode = "std3"
    legacy = RegisterWAM._register_attention_mask(dummy, **kwargs)
    assert torch.equal(mask, legacy)
    assert mask[:10, :10].all()
    assert mask[10:12, 10:12].all()
    assert not mask[:10, 10:12].any()
    assert not mask[10:12, :10].any()
    assert not mask[:12, 12:].any()
