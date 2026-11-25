import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.models.fenrec import VARIANTS, enduring_negatives, soft_label_weights, contrastive_loss, noise_negatives
from src.models.factory import build_model
from src.data.neural_dataset import TrainingDataset, verify_training_boundary
from src.utils.configuration import resolve_config
from test_neural import tiny_config


def source_methods():
    path = Path(__file__).resolve().parents[1] / "references/FENRec/src/trainers.py"
    if not path.exists():
        pytest.skip("Pinned reference checkout unavailable; standalone mathematical tests still run")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"_get_enduring_hard_negatives", "enduring_hard_negatives_contrastive_loss", "mask_correlated_samples_", "_softmax", "_cross_entropy_loss", "info_nce", "get_noise_negative_mask"}
    namespace = {"torch": torch, "nn": nn, "F": F}
    methods = {}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name in wanted:
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                    methods[node.name] = namespace[node.name]
    Reference = type("Reference", (), methods)
    obj = Reference()
    obj.warm_up_epoch = 20
    obj.args = SimpleNamespace(mix_ratio_rate=0., focal_m=.2, tau=8., f_neg=True)
    return obj


def test_soft_label_manual_and_boundary(parts):
    weights = soft_label_weights(torch.tensor([[2, 3, 4], [0, 0, 5], [0, 1, 2]]), .2)
    torch.testing.assert_close(weights[0], torch.tensor([1., .2, .04])/1.24)
    torch.testing.assert_close(weights[1], torch.tensor([0., 0., 1.]))
    torch.testing.assert_close(weights[2], torch.tensor([0., 1., .2])/1.2)
    dataset = TrainingDataset(parts["train"], 6, tiny_config("fenrec"))
    verify_training_boundary(dataset, parts)
    for sample in dataset.samples:
        heldout = set(parts["validation"].loc[parts["validation"].user_id == sample["user_id"], "item_id"]) | set(parts["test"].loc[parts["test"].user_id == sample["user_id"], "item_id"])
        assert not set(sample["future"]) & heldout


def test_enduring_negatives_formula_stop_gradient_and_norm():
    anchor = torch.tensor([[1., 0.]], requires_grad=True)
    neg = torch.tensor([[0., 2.]], requires_grad=True)
    result = enduring_negatives(neg, anchor, .4, .4)
    expected = F.normalize(torch.tensor([.4, .6]), dim=0)*2
    torch.testing.assert_close(result[0, 0], expected)
    assert not result.requires_grad
    torch.testing.assert_close(result.norm(dim=-1), torch.tensor([[2.]]))


def test_enduring_matches_pinned_source():
    source = source_methods()
    torch.manual_seed(7)
    z = torch.randn(8, 5)
    torch.manual_seed(42)
    expected = source._get_enduring_hard_negatives(z, z, .2, .5, 25)
    torch.manual_seed(42)
    actual = enduring_negatives(z, z, .2, .5)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("epoch", [0, 20, 25])
def test_contrastive_loss_and_gradients_match_pinned_source(epoch):
    source = source_methods()
    config = tiny_config("fenrec")
    torch.manual_seed(7)
    first, second = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([1, 2, 2, 3])
    noise = torch.randn(8, 8)
    torch.manual_seed(42)
    if epoch >= 20:
        logits, targets, lengths = source.enduring_hard_negatives_contrastive_loss(first, second, 1., epoch, 4, intent_id=labels, min_mix_ratio=.4, max_mix_ratio=.4, z_negative=noise)
        expected = source._cross_entropy_loss(logits, targets, [1, 1, .1], lengths)
    else:
        logits, targets = source.info_nce(first, second, 1., 4, intent_id=labels)
        expected = F.cross_entropy(logits, targets)
    expected_grad = torch.autograd.grad(expected, (first, second))
    torch.manual_seed(42)
    actual = contrastive_loss(first, second, labels, config, epoch, True, noise)
    actual_grad = torch.autograd.grad(actual, (first, second))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("variant", [f"F{i}" for i in range(6)])
def test_ablation_switches_and_same_backbone(variant, parts):
    config = tiny_config("fenrec", variant=variant)
    model = build_model("fenrec", 6, config)
    model.set_clusters(torch.randn(2, 8))
    assert (model.soft_labels, model.contrastive, model.enduring) == VARIANTS[variant]
    sizes = [sum(p.numel() for p in build_model("fenrec", 6, tiny_config("fenrec", variant=v)).parameters()) for v in VARIANTS]
    assert len(set(sizes)) == 1
    from torch.utils.data import DataLoader
    batch = next(iter(DataLoader(TrainingDataset(parts["train"], 6, config), batch_size=4)))
    loss, values = model.loss(batch, 21)
    loss.backward()
    assert torch.isfinite(loss)
    if variant in ("F0", "F1"):
        assert values["coarse_loss"] == values["fine_loss"] == 0


def test_ablation_contradiction_rejected():
    with pytest.raises(ValueError, match="contradicts"):
        tiny_config("fenrec", variant="F0", soft_labels=True)


def test_false_negative_mask_degenerate_finite():
    config = tiny_config("fenrec")
    z = torch.randn(3, 8, requires_grad=True)
    loss = contrastive_loss(z, z, torch.ones(3, dtype=torch.long), config, 0)
    assert loss == 0
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_noise_updates_are_detached_finite():
    config = tiny_config("fenrec")
    a, b = torch.randn(4, 8, requires_grad=True), torch.randn(4, 8, requires_grad=True)
    result = noise_negatives(a, b, torch.tensor([1, 2, 3, 4]), config)
    assert result.shape == (8, 8) and not result.requires_grad
    assert torch.isfinite(result).all()


def test_noise_updates_match_source_iteration_formula():
    config = tiny_config("fenrec")
    torch.manual_seed(7)
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    labels = torch.tensor([1, 2, 2, 3])
    torch.manual_seed(42)
    expected = torch.randn(8, 8, requires_grad=True)
    base = (a @ b.T).detach()
    keep = labels[:, None].ne(labels[None, :]) | torch.eye(4, dtype=torch.bool)
    base[~keep] = -torch.inf
    for _ in range(4):
        loss = F.cross_entropy(torch.cat([base, a.detach() @ expected.T], 1), torch.arange(4))
        gradient, = torch.autograd.grad(loss, expected, retain_graph=True)
        expected = expected + gradient/gradient.norm(dim=-1, keepdim=True)*1e-3
        expected = torch.where(torch.isnan(expected), torch.zeros_like(expected), expected)
    torch.manual_seed(42)
    actual = noise_negatives(a, b, labels, config)
    torch.testing.assert_close(actual, expected.detach())


def test_soft_labels_match_source_builder():
    import scipy.sparse as sp
    path = Path(__file__).resolve().parents[1] / "references/FENRec/src/datasets.py"
    if not path.exists():
        pytest.skip("Reference checkout unavailable")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(method for cls in tree.body if isinstance(cls, ast.ClassDef) for method in cls.body
                if isinstance(method, ast.FunctionDef) and method.name == "_get_soft_label")
    ns = {"np": np, "sp": sp, "tqdm": lambda iterator: iterator}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    obj = SimpleNamespace(id2seq={0: [1], 1: [1, 2], 2: [1, 2, 3]}, seq2user={0: 1, 1: 1, 2: 1},
                          id2tag={0: 2, 1: 3, 2: 4}, user2fullseq={1: [1, 2, 3, 4]})
    ns["_get_soft_label"](obj, gamma=.2)
    for index, future in enumerate([[2, 3, 4], [0, 3, 4], [0, 0, 4]]):
        weights = soft_label_weights(torch.tensor([future]), .2)[0]
        dense = obj.label_matrix[index].toarray()[0]
        for item, weight in zip(future, weights):
            if item:
                assert dense[item] == pytest.approx(float(weight))


def test_fine_clusters_query_raw_centers_then_normalize():
    model = build_model("fenrec", 6, tiny_config("fenrec"))
    centers = torch.eye(8)[:2]*3
    model.set_clusters(centers)
    labels, positives = model.query_clusters(torch.eye(8)[:2]*2)
    assert labels.tolist() == [0, 1]
    torch.testing.assert_close(positives.norm(dim=-1), torch.ones(2))
