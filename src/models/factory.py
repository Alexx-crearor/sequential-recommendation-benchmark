from src.models.sasrec import SASRec
from src.models.bert4rec import BERT4Rec
from src.models.gru4rec import GRU4RecCE


def build_model(name, num_items, config):
    if name == "gru4rec_ce":
        return GRU4RecCE(num_items, config)
    if name in ("sasrec_bce", "sasrec_ce"):
        return SASRec(num_items, config)
    if name in ("bert4rec", "attention_ablation"):
        return BERT4Rec(num_items, config)
    if name == "fenrec":
        from src.models.fenrec import FENRec
        return FENRec(num_items, config)
    raise ValueError(f"Unsupported model: {name}")
