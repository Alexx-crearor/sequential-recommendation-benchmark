import copy
from pathlib import Path
import yaml
from src.utils.configuration import ROOT

PROFILES = ("pilot", "core_research", "full_research")


def load_profile(name="full_research", final_seeds=False):
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    directory = ROOT / "configs/experiments"
    profile = yaml.safe_load((directory / f"profile_{name}.yaml").read_text())
    grid = copy.deepcopy(yaml.safe_load((directory / "experiment_grid.yaml").read_text()))
    grid["seeds"] = profile["final_seeds" if final_seeds else "initial_seeds"]
    if "experiments" in profile:
        grid["experiments"] = {key: value for key, value in grid["experiments"].items() if key in profile["experiments"]}
    for experiment, spec in grid["experiments"].items():
        if "datasets" in profile and "datasets" in spec:
            spec["datasets"] = [key for key in spec["datasets"] if key in profile["datasets"]]
        if profile.get("implemented_only") and "models" in spec:
            spec["models"] = [name for name in spec["models"] if name != "tiger"]
        if experiment == "E3" and "retention" in profile:
            spec["retention"] = profile["retention"]
        if experiment == "E5" and "sequence_lengths" in profile:
            spec["max_seq_length"] = profile["sequence_lengths"]
    return grid, profile


def profile_overrides(profile, model):
    return {**profile.get("overrides", {}), **profile.get("model_overrides", {}).get(model, {}),
            "profile": profile["name"], "common_warm_ratios": profile.get("retention", [.2, .4, .6, .8, 1.])}
