import copy
import json

import numpy as np
import pytest
import torch

from tactilelatent.aligned_frames import file_hash, json_hash
from tactilelatent.bimanual_mirror import load_policy, require_spec, transform_targets, verify_evidence
from tactilelatent.gel_mirror_preparation import evaluate_marker_pair, gel_runtime, load_gel_model
from tactilelatent.mirror_joint import artifact_policy_hash


@pytest.mark.parametrize('source,permutation', [('ftp_unit_bimanual', [1, 0]), ('baihu', [3, 2, 1, 0])])
def test_confirmed_policy_reflects_native_targets_and_preserves_missing_points(source, permutation):
    policy = load_policy('configs/bimanual_mirror_policy_v2.json')
    assert verify_evidence(policy)[source]['verified']
    spec = require_spec(policy, source)
    assert spec['verification_basis'] == 'user_visual_review'
    assert spec['surface_permutation'] == permutation
    surface = torch.tensor([sid for sid in range(len(permutation)) for _ in range(sid+2)])
    n = len(surface)
    original = dict(marker_surface=surface, marker_reference=torch.linspace(.1, .9, 2*n).reshape(n, 2),
        marker_target=torch.arange(2*n, dtype=torch.float32).reshape(1, n, 2)/10,
        marker_valid=torch.ones(1, n, dtype=torch.bool), marker_spacing=torch.linspace(10, 20, n))
    original['marker_target'][0, 1] = float('nan'); original['marker_valid'][0, 1] = False
    result = transform_targets(original, spec)
    assert result['marker_surface'].tolist() == [permutation[i] for i in surface.tolist()]
    torch.testing.assert_close(result['marker_reference'][:, 0], 1-original['marker_reference'][:, 0])
    torch.testing.assert_close(result['marker_target'][..., 0], -original['marker_target'][..., 0], equal_nan=True)
    torch.testing.assert_close(result['marker_target'][..., 1], original['marker_target'][..., 1], equal_nan=True)
    for key, value in original.items():
        torch.testing.assert_close(transform_targets(result, spec)[key], value, equal_nan=True)
    assert not result['marker_valid'][0, 1]
    with pytest.raises(ValueError, match='blocked'):
        require_spec(policy, 'opengraph')


def test_inherited_source_receipt_retains_original_policy_and_rejects_mapping_changes():
    old = load_policy('configs/bimanual_mirror_policy_v1.json')
    current = load_policy('configs/bimanual_mirror_policy_v2.json')
    ready = dict(policy_sha256=json_hash(current))
    entry = dict(artifact_policy_path='configs/bimanual_mirror_policy_v1.json',
                 artifact_policy_sha256=json_hash(old))
    assert artifact_policy_hash(current, ready, entry, 'egotouch') == json_hash(old)
    changed = copy.deepcopy(current); changed['sources']['egotouch']['flip_width'] = False
    with pytest.raises(ValueError, match='changed source mapping'):
        artifact_policy_hash(changed, ready, entry, 'egotouch')
    with pytest.raises(ValueError, match='policy changed'):
        artifact_policy_hash(current, ready, dict(entry, artifact_policy_sha256='bad'), 'egotouch')


def test_normalized_baihu_teacher_loads_strictly_and_runtime_is_scoped(tmp_path):
    from tactilelatent.baihu_autoencoder import BaihuMarkerAutoencoder
    from tactilelatent import mirror_preparation as prep
    model = BaihuMarkerAutoencoder(width=16, heads=2, queries=2, latent_dim=32)
    path = tmp_path/'teacher.pt'
    torch.save(dict(model_type='expansion_marker_autoencoder_v2', model_config=model.config,
                    model=model.state_dict()), path)
    info = dict(checkpoint=str(path), sha256=file_hash(path))
    previous = prep.load_model, prep.evaluate, prep.verify_codes
    with gel_runtime():
        restored, _ = prep.load_model(info, 'cpu')
        assert isinstance(restored, BaihuMarkerAutoencoder)
        assert not any(p.requires_grad for p in restored.parameters())
        y = torch.randn(2, 7, 2); valid = torch.ones(2, 7, dtype=torch.bool)
        reference = torch.rand_like(y); surface = torch.arange(7).expand(2, -1) % 4
        model.eval()
        torch.testing.assert_close(restored(y, valid, reference, surface), model(y, valid, reference, surface))
    assert (prep.load_model, prep.evaluate, prep.verify_codes) == previous
    with pytest.raises(ValueError, match='teacher changed'):
        load_gel_model(dict(info, sha256='bad'), 'cpu')


def test_gel_audit_reports_signed_errors_in_native_pixels_and_each_platform(monkeypatch):
    from tactilelatent import mirror_preparation as prep
    rows = [dict(source='baihu', platform=p, episode_id=p) for p in ('D-WHEEL', 'PIKA', 'QINGLOONG')]
    monkeypatch.setattr(prep, 'sampled_batches', lambda *args: [(r, np.array([0])) for r in rows])

    class Reader:
        def read(self, row, indices):
            return dict(marker_target=torch.tensor([[[-.2, .4], [float('nan'), float('nan')]]]),
                marker_valid=torch.tensor([[True, False]]), marker_reference=torch.tensor([[.1, .2], [.7, .8]]),
                marker_surface=torch.tensor([0, 1]), marker_spacing=torch.tensor([10., 20.]))

    class HalfReconstruction(torch.nn.Module):
        displacement_scale = .1

        def forward(self, target, valid, reference, surface):
            return .5*torch.nan_to_num(target)

    policy = load_policy('configs/bimanual_mirror_policy_v2.json')
    result = evaluate_marker_pair(HalfReconstruction(), rows, require_spec(policy, 'baihu'), Reader(), 'cpu')
    for metrics in result.values():
        assert metrics['mae'] == pytest.approx(1.5)
        assert metrics['all_zero_baseline_mae'] == pytest.approx(3.)
        assert metrics['mae_spacing'] == pytest.approx(.15)
        assert metrics['units'] == 'native_pixels'
        assert metrics['valid_coordinates'] == 6
        assert set(metrics['by_platform']) == {'D-WHEEL', 'PIKA', 'QINGLOONG'}
