"""
Tree encoder for propositional logic expressions.

Replaces LeWM's ViT encoder. Processes BFS-encoded expression trees via
Transformer self-attention. Outputs match LeWM's expected interface:
    output = encoder(data)
    embedding = output.last_hidden_state[:, 0]  # CLS token
"""

import torch
import torch.nn as nn
import math

from .data import NUM_NODE_TYPES, MAX_NODES, VARIABLES


class EncoderOutput:
    """Matches HuggingFace model output interface used by LeWM's JEPA."""
    def __init__(self, cls_embedding):
        self.last_hidden_state = cls_embedding.unsqueeze(1)  # (B, 1, D)


class TreeEncoder(nn.Module):
    """Transformer encoder for propositional logic expression trees.

    Input: dict with node_types, var_ids, adjacency, num_nodes
    Output: EncoderOutput with .last_hidden_state[:, 0] = CLS embedding

    Architecture:
    - Node content embeddings (type + variable)
    - Tree position embeddings (depth + BFS index)
    - N layers of pre-norm self-attention with LayerScale
    - CLS aggregation via learned query cross-attention
    """

    def __init__(self, dim=384, num_layers=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # Content
        self.node_type_emb = nn.Embedding(NUM_NODE_TYPES + 1, dim // 2, padding_idx=0)
        self.var_emb = nn.Embedding(len(VARIABLES) + 1, dim // 2, padding_idx=len(VARIABLES))

        # Position (depth + BFS index)
        self.depth_emb = nn.Embedding(8, dim)
        self.pos_emb = nn.Embedding(MAX_NODES, dim)
        depths = [min(int(math.log2(i + 1)) if i > 0 else 0, 7) for i in range(MAX_NODES)]
        self.register_buffer("depths", torch.tensor(depths, dtype=torch.long))

        # Input projection
        self.input_proj = nn.Linear(dim + dim, dim)
        self.input_norm = nn.LayerNorm(dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim), nn.Dropout(dropout),
                ),
                "ls1": nn.ParameterList([nn.Parameter(torch.ones(dim) * 1e-4)]),
                "ls2": nn.ParameterList([nn.Parameter(torch.ones(dim) * 1e-4)]),
            }))

        # CLS aggregation
        self.output_norm = nn.LayerNorm(dim)
        self.cls_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cls_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cls_norm = nn.LayerNorm(dim)

        # LeWM compatibility
        class Config:
            hidden_size = dim
        self.config = Config()

    def forward(self, tree_data, interpolate_pos_encoding=False):
        """
        tree_data: dict with keys node_types, var_ids, adjacency, num_nodes
        interpolate_pos_encoding: ignored (LeWM ViT compat)
        Returns: EncoderOutput with .last_hidden_state (B, 1, D)
        """
        node_types = tree_data["node_types"]
        var_ids = tree_data["var_ids"]
        B = node_types.shape[0]
        device = node_types.device

        # Content embeddings
        type_emb = self.node_type_emb(node_types)
        var_safe = var_ids.clone()
        var_safe[var_safe < 0] = len(VARIABLES)
        var_e = self.var_emb(var_safe)
        content = torch.cat([type_emb, var_e], dim=-1)

        # Position embeddings
        pos_idx = torch.arange(MAX_NODES, device=device)
        pos = self.depth_emb(self.depths.to(device)) + self.pos_emb(pos_idx)
        pos = pos.unsqueeze(0).expand(B, -1, -1)

        # Combine and project
        x = self.input_norm(self.input_proj(torch.cat([content, pos], dim=-1)))
        padding_mask = (node_types == 0)

        # Transformer
        for layer in self.layers:
            normed = layer["norm1"](x)
            attn_out, _ = layer["attn"](normed, normed, normed, key_padding_mask=padding_mask)
            x = x + layer["ls1"][0] * attn_out
            x = x + layer["ls2"][0] * layer["ffn"](layer["norm2"](x))

        # CLS aggregation
        cls = self.cls_query.expand(B, -1, -1)
        cls_out, _ = self.cls_attn(
            self.cls_norm(cls), self.output_norm(x), x,
            key_padding_mask=padding_mask
        )

        return EncoderOutput(cls_out.squeeze(1))
