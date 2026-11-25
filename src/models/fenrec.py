"""FENRec: reference-code loss branch, source backbone, coarse + fine CL.

Source audit: uikdwnd/FENRec fd111ec src/trainers.py and src/modules.py.
CPU sklearn clustering replaces GPU FAISS, explicitly recorded in config.
"""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.base import SequentialModel, initialize
from src.models.bert4rec import PostNormBlock


VARIANTS = {
    "F0": (False, "none", False), "F1": (True, "none", False),
    "F2": (False, "standard", False), "F3": (False, "reference_hard", True),
    "F4": (True, "standard", False), "F5": (True, "reference_hard", True),
    "default": (True, "reference_hard", True),
}


def soft_label_weights(future, gamma):
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0,1]")
    valid = future.ne(0)
    if not valid.any(-1).all():
        raise ValueError("Empty soft-label window")
    positions = valid.cumsum(-1) - 1
    weights = (gamma ** positions.clamp_min(0).float()) * valid
    return weights / weights.sum(-1, keepdim=True)


def enduring_negatives(negatives, anchors, low, high):
    if not 0 <= low <= high <= 1:
        raise ValueError("Invalid mixing bounds")
    # Same shape and random draw order as pinned source. Eps handles zero vectors.
    neg = F.normalize(negatives.float(), dim=-1)[None].expand(len(anchors), -1, -1)
    anchor = F.normalize(anchors.float(), dim=-1)[:, None]
    lda = torch.rand(neg.shape[:-1], device=anchors.device).unsqueeze(-1) * (high-low) + low
    mixed = F.normalize((1-lda)*neg + lda*anchor, dim=-1)
    return (mixed * negatives.float().norm(dim=-1)[None, :, None]).detach()


def noise_negatives(first, second, labels, config):
    count = int(len(first) * config["noise_negative_portion"])
    if count == 0:
        return first.new_empty((0, first.shape[-1]))
    with torch.enable_grad():
        noise = torch.randn(count, first.shape[-1], device=first.device, requires_grad=True)
        base = (first.detach().float() @ second.detach().float().T)
        keep = labels[:, None].ne(labels[None, :]) | torch.eye(len(labels), device=labels.device, dtype=torch.bool)
        base = base.masked_fill(~keep, -torch.inf)
        for _ in range(config["noise_steps"]):
            scores = torch.cat([base, first.detach().float() @ noise.T], 1)
            loss = F.cross_entropy(scores, torch.arange(len(first), device=first.device))
            grad, = torch.autograd.grad(loss, noise)
            noise = (noise + F.normalize(grad, dim=-1) * config["noise_step_size"]).detach().requires_grad_(True)
    return noise.detach()


def contrastive_loss(first, second, labels, config, epoch, hard=False, noise=None):
    """Symmetric mean over 2B anchors; matches source positive and FNM ordering."""
    first, second = first.float(), second.float()
    z = torch.cat([first, second])
    batch = len(first)
    repeated = labels.repeat(2)
    keep = repeated[:, None].ne(repeated[None, :])
    raw = z @ z.T
    temperature = config["temperature"]
    positive_indices = (torch.arange(2*batch, device=z.device) + batch) % (2*batch)
    positive = raw[torch.arange(2*batch, device=z.device), positive_indices] / temperature
    active = hard and epoch >= config["warm_up_epochs"]
    if not active:
        negative = (raw / temperature).masked_fill(~keep, -torch.inf)
        logits = torch.cat([positive[:, None], negative], 1)
    else:
        tau, margin = config["tau"], config["focal_m"]
        similarity = torch.tanh(raw / tau)
        positive = positive * similarity[torch.arange(2*batch, device=z.device), positive_indices]
        # Primary branch reproduces source, not the paper's additive margin.
        def weight(value):
            if config["fenrec_loss_reference"] == "code":
                return value / temperature * (torch.tanh(value / tau) + margin)
            return (value * torch.tanh(value / tau) + margin) / temperature
        negative = weight(raw).masked_fill(~keep, -torch.inf)
        low = max(min(config["mix_ratio_rate"] * (epoch + 1 - config["warm_up_epochs"]) + config["min_mix_ratio"], config["max_mix_ratio"]), config["min_mix_ratio"])
        mixed = enduring_negatives(z, z, low, config["max_mix_ratio"])
        hard_raw = torch.bmm(z[:, None], mixed.transpose(1, 2)).squeeze(1)
        hard_logits = weight(hard_raw).masked_fill(~keep, -torch.inf)
        portion = min(config["mix_portion"] * (epoch+1-config["warm_up_epochs"]), config["mix_portion"])
        components = [positive[:, None], negative]
        if noise is not None and len(noise):
            components.append(weight(z @ noise.float().T))
        if portion > 0:
            components.append(hard_logits + torch.log(z.new_tensor(portion)))
        logits = torch.cat(components, 1)
    return F.cross_entropy(logits, torch.zeros(len(z), device=z.device, dtype=torch.long))


class FENRec(SequentialModel):
    def __init__(self, num_items, config):
        super().__init__(num_items, config)
        variant = config.get("variant", "default")
        if variant not in VARIANTS:
            raise ValueError(f"Unsupported FENRec variant: {variant}")
        self.soft_labels, self.contrastive, self.enduring = VARIANTS[variant]
        self.position = nn.Embedding(self.max_length, self.dim)
        self.input_norm = nn.LayerNorm(self.dim, eps=1e-12)
        self.dropout = nn.Dropout(config["dropout"])
        self.blocks = nn.ModuleList([PostNormBlock(config) for _ in range(config["num_layers"])])
        self.register_buffer("cluster_centers", torch.empty(0, self.dim), persistent=False)
        self.apply(initialize)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, tokens):
        valid = tokens.ne(0)
        hidden = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        hidden = self.dropout(self.input_norm(hidden)) * valid.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, valid, True)
        return hidden

    def set_clusters(self, centers):
        self.cluster_centers = centers.detach().to(self.device)

    def query_clusters(self, hidden):
        if not len(self.cluster_centers):
            raise ValueError("Fine-grained contrastive learning requires fitted training-only clusters")
        with torch.no_grad():
            labels = torch.cdist(hidden.detach().float(), self.cluster_centers.float()).argmin(-1)
            positives = F.normalize(self.cluster_centers[labels], dim=-1)
        return labels, positives

    def loss(self, batch, epoch):
        main = self(batch["tokens"])[:, -1]
        logits = self.logits(main).float()
        if self.soft_labels:
            future = batch["future"]
            weights = soft_label_weights(future, self.config["soft_label_gamma"])
            chosen = F.log_softmax(logits, -1).gather(1, (future-1).clamp_min(0))
            rec_loss = -(chosen * weights).sum(-1).mean()
        else:
            rec_loss = F.cross_entropy(logits, batch["target"]-1)
        coarse, fine = rec_loss.new_zeros(()), rec_loss.new_zeros(())
        if self.contrastive != "none":
            first, second = self(batch["tokens"])[:, -1], self(batch["positive"])[:, -1]
            hard = self.enduring
            noise = noise_negatives(first, second, batch["target"], self.config) if hard and epoch >= self.config["warm_up_epochs"] else None
            coarse = contrastive_loss(first, second, batch["target"], self.config, epoch, hard, noise)
            if self.config["beta_0"]:
                for representation in (first, second):
                    label, positive = self.query_clusters(representation)
                    fine = fine + contrastive_loss(representation, positive, label, self.config, epoch, hard, noise)
        loss = self.config["rec_weight"]*rec_loss + self.config["lambda_0"]*coarse + self.config["beta_0"]*fine
        return loss, {"recommendation_loss": float(rec_loss.detach()), "coarse_loss": float(coarse.detach()), "fine_loss": float(fine.detach())}
