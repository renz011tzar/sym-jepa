"""
Rule encoder for discrete rewrite rules.

Replaces LeWM's continuous action Embedder. Maps discrete rule IDs (0-14)
to embedding vectors compatible with the ARPredictor's AdaLN conditioning.
"""

import torch
import torch.nn as nn

from .data import NUM_RULES


class RuleEncoder(nn.Module):
    """Maps discrete rule IDs to embeddings.

    LeWM's Embedder takes continuous action vectors (B, T, action_dim)
    through Conv1d → MLP. We replace this with a simple embedding lookup
    since our actions are discrete rule IDs.
    """

    def __init__(self, emb_dim=384):
        super().__init__()
        self.embedding = nn.Embedding(NUM_RULES, emb_dim)

    def forward(self, rule_ids):
        """
        rule_ids: (B, T) int tensor of rule IDs (0-14)
        Returns: (B, T, emb_dim)
        """
        return self.embedding(rule_ids)
