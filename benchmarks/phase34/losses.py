"""Frozen original loss for controlled before/after timing."""
import torch
from torch.nn import functional as F


def objective_loss(model, batch, epoch):
    if model.config["model"] == "fenrec":
        return model.loss(batch, epoch)
    hidden = model(batch["tokens"])
    valid = batch["targets"].ne(0)
    states, targets = hidden[valid], batch["targets"][valid]
    if model.config["objective"] == "next_bce":
        projected = model.project(states)
        positive = (projected * model.embedding(targets)).sum(-1)
        negative = (projected * model.embedding(batch["negatives"][valid])).sum(-1)
        if hasattr(model, "output_bias"):
            positive = positive + model.output_bias[targets-1]
            negative = negative + model.output_bias[batch["negatives"][valid]-1]
        loss = F.softplus(-positive.float()).mean() + F.softplus(negative.float()).mean()
    else:
        chunks = model.config["loss_position_chunk"]
        loss = states.new_zeros((), dtype=torch.float32)
        for start in range(0, len(states), chunks):
            logits = model.logits(states[start:start+chunks]).float()
            loss = loss + F.cross_entropy(logits, targets[start:start+chunks]-1, reduction="sum") / len(states)
    return loss, {"recommendation_loss": float(loss.detach()), "coarse_loss": 0., "fine_loss": 0.}
