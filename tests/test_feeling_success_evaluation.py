import h5py
import numpy as np
import torch

from tactilelatent.evaluate_feeling_success import (INPUT_FIELDS, SuccessImages,
    binary_metrics, object_splits, predict, ranking_metrics)
from tactilelatent.frozen_probe import head


def test_object_splits_are_disjoint_and_label_independent():
    records = [{"object_name": f"object_{i // 3}", "success": bool(i % 2)} for i in range(105)]
    assignment, ids = object_splits(records)
    other, same_ids = object_splits([{**r, "success": not r["success"]} for r in records])
    assert assignment == other and ids == same_ids
    groups = [{records[i]["object_name"] for i in ids[p]} for p in ("fit", "tune", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert sorted(sum(ids.values(), [])) == list(range(len(records)))


def test_inputs_do_not_require_post_lift_or_labels(tmp_path):
    path = tmp_path / "trial.h5"
    with h5py.File(path, "w") as stream:
        group = stream.create_group("i0")
        for field in INPUT_FIELDS:
            group[field] = np.full((12, 20, 3), 128, dtype=np.uint8)
    batch = SuccessImages([{"path": str(path), "group": "i0"}])[0]
    assert set(batch) == {"rgb", "view_mask"}
    assert batch["rgb"].shape == (2, 3, 3, 224, 224)
    assert batch["view_mask"][:, 0].all()
    assert not batch["view_mask"][:, 1:].any()
    assert not batch["rgb"][:, 1:].any()


def test_binary_metrics_ties_and_coverage():
    truth = torch.tensor([0, 0, 1, 1])
    assert ranking_metrics(torch.tensor([.1, .2, .8, .9]), truth)["auroc"] == 1.
    assert ranking_metrics(torch.ones(4), truth) == {"auroc": .5, "average_precision": .5}
    p = torch.tensor([[.9, .1], [.8, .2], [.2, .8], [.1, .9]])
    result = binary_metrics(p, truth)
    assert result["accuracy"] == result["balanced_accuracy"] == 1.
    assert abs(result["brier_score"] - .025) < 1e-6
    assert result["confusion_matrix_true_rows_failure_success"] == [[2, 0], [0, 2]]


def test_fixed_sampling_and_head_shape():
    network = head(32, 2, classification=True)
    table = {"mu": torch.randn(5, 32), "logvar": torch.zeros(5, 32)}
    before = {k: v.clone() for k, v in network.state_dict().items()}
    a, b = predict(network, table, 64, 123), predict(network, table, 64, 123)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert a[0].shape == a[1].shape == (5, 2)
    assert all(torch.equal(before[k], v) for k, v in network.state_dict().items())
