"""Numerical source-backbone comparison without importing upstream GPU dependencies."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.models.factory import build_model
from test_neural import tiny_config


def test_fenrec_backbone_matches_reference_on_real_positions():
    root = Path(__file__).resolve().parents[1] / "references/FENRec/src"
    if not (root / "modules.py").exists():
        pytest.skip("Reference checkout unavailable")
    spec = importlib.util.spec_from_file_location("fenrec_reference_modules", root / "modules.py")
    modules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modules)
    tree = ast.parse((root / "models.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SASRecModel")
    ns = {"torch": torch, "nn": nn, "Encoder": modules.Encoder, "LayerNorm": modules.LayerNorm}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "reference_models.py", "exec"), ns)
    args = SimpleNamespace(item_size=12, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                           max_seq_length=5, hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
                           hidden_act="gelu", initializer_range=.02, cuda_condition=False)
    reference = ns["SASRecModel"](args).eval()
    ours = build_model("fenrec", 10, tiny_config("fenrec")).eval()
    def copy_linear(left, right):
        left.load_state_dict(right.state_dict())
    copy_linear(reference.item_embeddings, ours.embedding)
    copy_linear(reference.position_embeddings, ours.position)
    copy_linear(reference.LayerNorm, ours.input_norm)
    for old, new in zip(reference.item_encoder.layer, ours.blocks):
        for name in ("query", "key", "value"):
            copy_linear(getattr(old.attention, name), getattr(new.attention, name))
        copy_linear(old.attention.dense, new.attention.output)
        copy_linear(old.attention.LayerNorm, new.norm1)
        copy_linear(old.intermediate.dense_1, new.ffn[0])
        copy_linear(old.intermediate.dense_2, new.ffn[2])
        copy_linear(old.intermediate.LayerNorm, new.norm2)
    tokens = torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 5]])
    torch.testing.assert_close(ours(tokens)[tokens != 0], reference(tokens)[tokens != 0], rtol=2e-5, atol=2e-6)
