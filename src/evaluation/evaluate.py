import pandas as pd
from src.data.splitting import evaluation_cases
from src.evaluation.ranking import rank_catalogue
from src.evaluation.metrics import contributions


def evaluate(model, parts, split="test", item_batch_size=4096):
    from src.models.popularity import MostPopular
    if isinstance(model, MostPopular):
        from src.evaluation.popularity_ranking import evaluate_popularity
        return evaluate_popularity(model, parts, split)
    if hasattr(model, "representations"):
        from src.evaluation.neural_ranking import evaluate_neural
        return evaluate_neural(model, parts, split, item_batch_size)
    warm = set(parts["train"].item_id)
    rows = []
    for case in evaluation_cases(parts, split):
        top, rank = rank_catalogue(model, case["history"], case["target"], model.num_items, item_batch_size=item_batch_size)
        rows.append({**case, "top_k": top, "target_rank": rank, "warm_target": case["target"] in warm,
                     **contributions(rank)})
    predictions = pd.DataFrame(rows)
    metrics = {}
    columns = list(contributions(1))
    for name, subset in (("full", predictions), ("warm", predictions.loc[predictions.warm_target])):
        metrics[name] = {"users": len(subset), **{key: float(subset[key].mean()) if len(subset) else None for key in columns}}
    metrics["warm_coverage"] = float(predictions.warm_target.mean())
    return metrics, predictions
