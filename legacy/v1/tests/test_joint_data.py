from collections import defaultdict
from types import SimpleNamespace

import torch

from tactilelatent.joint_data import SourceBatchSampler
from tactilelatent.joint_model import JointTactileVAE
from tactilelatent.prepare_joint import TARGETS
from tactilelatent.train_joint import joint_loss


def test_source_batch_sampler_never_mixes_target_shapes():
    dataset = SimpleNamespace(indices_by_source=defaultdict(list, {
        "egotouch": [0, 1, 2, 3, 4],
        "freetacman": [5, 6, 7, 8, 9],
    }))
    sampler = SourceBatchSampler(dataset, batch_size=2, seed=17,
                                 max_windows_per_source=4)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 4
    assert all(set(i < 5 for i in batch) in ({True}, {False}) for batch in batches)
    assert {i for batch in batches for i in batch} <= set(range(10))
    assert len({i for batch in batches for i in batch}) == 8
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_cycle_sampler_visits_large_source_before_repeating():
    dataset = SimpleNamespace(indices_by_source=defaultdict(list, {
        "large": list(range(12)), "small": [12, 13],
    }))
    sampler = SourceBatchSampler(dataset, batch_size=2, seed=7,
                                 max_windows_per_source=4, sampling_mode="cycle")
    seen = set()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        current = {i for batch in batches for i in batch if i < 12}
        assert len(current) == 4
        assert not current & seen
        seen |= current
        assert {12, 13} <= {i for batch in batches for i in batch}
        assert list(sampler) == batches
    assert seen == set(range(12))


def test_all_joint_decoders_have_native_shapes_and_gradients():
    model = JointTactileVAE(input_dim=12, width=16, heads=2,
                            temporal_layers=1, cross_view_layers=1,
                            latent_dim=7, dropout=0)
    features = torch.randn(1, 2, 3, 4, 12)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    for source, (_, shape, _) in TARGETS.items():
        output = model(features, mask, source)
        assert output["pred"].shape == (1, 2, *shape)
        target = torch.rand_like(output["pred"])
        valid = torch.ones_like(target, dtype=torch.bool)
        loss = joint_loss(output, target, valid, source, beta=.001)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.visual.prior_mu.weight.grad is not None
        model.zero_grad(set_to_none=True)
