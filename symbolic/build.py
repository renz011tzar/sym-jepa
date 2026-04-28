"""
Build a LeWM JEPA model adapted for propositional logic rewrite dynamics.

Assembles:
- TreeEncoder (ours) instead of ViT
- RuleEncoder (ours) instead of continuous Embedder
- ARPredictor, SIGReg, JEPA (LeWM, unmodified)
"""

import torch.nn as nn

from lewm.module import ARPredictor, MLP, SIGReg
from lewm.jepa import JEPA

from .encoder import TreeEncoder
from .rule_encoder import RuleEncoder


CONFIGS = {
    "large": {
        "dim": 384,
        "enc_layers": 8,
        "enc_heads": 8,
        "pred_depth": 6,
        "pred_heads": 8,
        "pred_dim_head": 48,
        "pred_mlp_dim": 1536,
        "sigreg_lambda": 0.09,  # LeWM's official value (config/train/lewm.yaml)
        "dropout": 0.1,
    },
    "xl": {
        "dim": 512,
        "enc_layers": 8,
        "enc_heads": 8,
        "pred_depth": 6,
        "pred_heads": 8,
        "pred_dim_head": 64,
        "pred_mlp_dim": 2048,
        "sigreg_lambda": 0.09,  # same as LeWM official
        "dropout": 0.1,
    },
}


def build_model(size="large"):
    """Build JEPA with tree encoder and rule encoder.

    Returns: (model, sigreg, config)
    """
    cfg = CONFIGS[size]
    dim = cfg["dim"]

    encoder = TreeEncoder(
        dim=dim,
        num_layers=cfg["enc_layers"],
        num_heads=cfg["enc_heads"],
        dropout=cfg["dropout"],
    )

    predictor = ARPredictor(
        num_frames=1,
        input_dim=dim,
        hidden_dim=dim,
        output_dim=dim,
        depth=cfg["pred_depth"],
        heads=cfg["pred_heads"],
        dim_head=cfg["pred_dim_head"],
        mlp_dim=cfg["pred_mlp_dim"],
        dropout=cfg["dropout"],
    )

    rule_encoder = RuleEncoder(emb_dim=dim)

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=rule_encoder,
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )

    sigreg = SIGReg(knots=17, num_proj=1024)

    return model, sigreg, cfg
