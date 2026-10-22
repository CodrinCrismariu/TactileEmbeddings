import torch
from tactilelatent.evaluate_dataset_classifier import class_table


def test_balanced_source_table_preserves_latents_and_equal_counts():
    parts = [{"mu": torch.full((n, 2, 32), float(i)),
              "logvar": torch.full((n, 2, 32), -float(i))}
             for i, n in enumerate((2, 5, 1))]
    table = class_table(parts, torch.device('cpu'), balanced=True)
    assert torch.bincount(table['y']).tolist() == [10, 10, 10]
    assert table['mu'].shape == (30, 32)
    for i in range(3):
        assert (table['mu'][table['y'] == i] == i).all()
        assert (table['logvar'][table['y'] == i] == -i).all()
