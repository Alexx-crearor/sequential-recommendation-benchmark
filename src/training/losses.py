import torch
from torch.nn import functional as F


def objective_loss(model, batch, epoch):
    if model.config["model"] == "fenrec":
        return model.loss(batch, epoch)
    hidden = model(batch["tokens"])
    valid = batch["targets"].ne(0)
    if not valid.any():
        raise ValueError("Batch has no prediction targets")
    states, targets = hidden[valid], batch["targets"][valid]
    if model.config["objective"] == "next_bce":
        # Dot products for one positive and one train-history-excluded negative per position.
        projected = model.project(states)
        positive = (projected * model.embedding(targets)).sum(-1)
        negative = (projected * model.embedding(batch["negatives"][valid])).sum(-1)
        if hasattr(model, "output_bias"):
            positive = positive + model.output_bias[targets-1]
            negative = negative + model.output_bias[batch["negatives"][valid]-1]
        loss = F.softplus(-positive.float()).mean() + F.softplus(negative.float()).mean()
    else:
        # Chunk the selected sequence positions; never materialize B*L*catalogue at once.
        chunks = model.config["loss_position_chunk"]
        loss = states.new_zeros((), dtype=torch.float32)
        for start in range(0, len(states), chunks):
            labels = targets[start:start+chunks]-1
            def chunk_loss(hidden, labels):
                return F.cross_entropy(model.logits(hidden).float(), labels, reduction="sum")
            if model.config.get("activation_checkpoint_ce", False) and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                value = checkpoint(chunk_loss, states[start:start+chunks], labels, use_reentrant=False)
            else:
                value = chunk_loss(states[start:start+chunks], labels)
            loss = loss + value / len(states)
    return loss, {"recommendation_loss": float(loss.detach()), "coarse_loss": 0., "fine_loss": 0.}
