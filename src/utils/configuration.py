from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["gru4rec_ce", "sasrec_bce", "sasrec_ce", "bert4rec", "attention_ablation", "fenrec"]


def resolve_config(model, overrides=None):
    grid = yaml.safe_load((ROOT / "configs/experiments/experiment_grid.yaml").read_text())
    config = {**grid["defaults"], "model": model, "seed": 7, "attention_dropout": .2,
              "adam_beta2": .999, "weight_decay": 0., "amp": False, "gradient_clip": 5.,
              "loss_position_chunk": 128, "eval_batch_size": 128, "item_batch_size": 4096,
              "last_mask_probability": .1, "cpu_threads": 4, "pilot": False,
              "max_train_batches": None, "validation_user_limit": None,
              "implementation": "phase5a-v1", "ranking": "full_catalogue_history_mask_target_exempt_ties_id",
              "ranking_backend": "torch", "cache_preprocessing": True, "preprocessing_cache_mb": 512,
              "num_workers": 0, "pin_memory": True, "prefetch_factor": 2, "gradient_accumulation_steps": 1,
              "validation_every": 1, "checkpoint_every": 1, "min_epochs": 1, "min_post_warmup_epochs": 1,
              "common_warm_ratios": [.2, .4, .6, .8, 1.], "evaluate_test": True,
              "activation_checkpoint_ce": False, "cluster_gpu_buffer_mb": 128}
    model_path = ROOT / "configs/models" / f"{model}.yaml"
    if model_path.exists():
        config.update(yaml.safe_load(model_path.read_text()))
    if overrides:
        config.update(overrides)
    if model not in MODELS:
        raise ValueError(f"Unsupported neural model: {model}")
    if config["hidden_dim"] != config["embedding_dim"]:
        raise ValueError("Current backbones require embedding_dim == hidden_dim")
    for key in ("hidden_dim", "num_layers", "num_heads", "ffn_dim", "max_seq_length", "batch_size", "max_epochs", "eval_batch_size", "item_batch_size", "loss_position_chunk", "cpu_threads", "early_stopping_patience"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 < config["mask_ratio"] <= 1 or not 0 <= config["last_mask_probability"] <= 1:
        raise ValueError("Invalid masking protocol")
    for key in ("gradient_accumulation_steps", "validation_every", "checkpoint_every", "min_epochs", "prefetch_factor"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["num_workers"] < 0 or config["min_epochs"] > config["max_epochs"]:
        raise ValueError("Invalid worker count or minimum training budget")
    if config["ranking_backend"] not in ("torch", "reference"):
        raise ValueError("Unknown ranking backend")
    if config["max_train_batches"] is not None or config["validation_user_limit"] is not None:
        if not config["pilot"]:
            raise ValueError("Truncated training/validation requires --pilot")
    if model == "fenrec":
        from src.models.fenrec import VARIANTS
        variant = config.get("variant", "default")
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported FENRec variant {variant}")
        soft, contrastive, enduring = VARIANTS[variant]
        expected = {"soft_labels": soft, "contrastive": contrastive, "enduring": enduring}
        for key, value in expected.items():
            if key in (overrides or {}) and config[key] != value:
                raise ValueError(f"Ablation switch {key} contradicts {variant}")
        config.update(expected)
        if config["fenrec_loss_reference"] not in ("code", "paper_margin"):
            raise ValueError("Unknown FENRec weighting reference")
        if config["amp"]:
            raise ValueError("FENRec remains FP32: its nested noise-gradient objective is not AMP-certified")
        if contrastive != "none" and config["gradient_accumulation_steps"] != 1:
            raise ValueError("FENRec contrastive negatives depend on the complete batch; gradient accumulation would change the objective. Use accumulation=1 or F0/F1.")
    return config
