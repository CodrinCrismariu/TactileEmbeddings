"""Inspect actual cross-attention queries without changing model computation."""

import math

import torch
from torch.nn import functional as F


class AttentionCapture:
    """Capture pre-softmax Q/K inputs while the normal inference path runs.

    Hooks do not replace modules or change ``need_weights``. Recomputing QK
    scores separately keeps the model's original SDPA numerical path intact.
    Weights describe the current-frame query reading contextual video tokens;
    they are not calibrated probabilities of physical contact.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.weights = []

    def __enter__(self):
        for layer in self.model.visual.decoder.layers:
            self.handles.append(
                layer.multihead_attn.register_forward_pre_hook(self._capture, with_kwargs=True)
            )
        return self

    @torch.no_grad()
    def _capture(self, attention, args, kwargs):
        query = args[0] if args else kwargs["query"]
        key = args[1] if len(args) > 1 else kwargs["key"]
        if not attention.batch_first:
            query, key = query.transpose(0, 1), key.transpose(0, 1)
        width = attention.embed_dim
        if attention.in_proj_weight is None:
            raise ValueError("The viewer expects equal-width Q/K/V cross attention")
        bias = attention.in_proj_bias
        q = F.linear(
            query, attention.in_proj_weight[:width], None if bias is None else bias[:width]
        )
        k = F.linear(
            key,
            attention.in_proj_weight[width : 2 * width],
            None if bias is None else bias[width : 2 * width],
        )
        batch, queries, _ = q.shape
        heads = attention.num_heads
        per_head = width // heads
        q = q.reshape(batch, queries, heads, per_head).transpose(1, 2).float()
        k = k.reshape(batch, -1, heads, per_head).transpose(1, 2).float()
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(per_head)
        padding = kwargs.get("key_padding_mask")
        if padding is not None:
            if padding.dtype == torch.bool:
                scores = scores.masked_fill(padding[:, None, None], -torch.inf)
            else:
                scores = scores + padding[:, None, None].float()
        mask = kwargs.get("attn_mask")
        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, -torch.inf)
            else:
                scores = scores + mask.float()
        weights = scores.softmax(-1)
        grid = self.model.visual.grid_size
        self.weights.append(weights.reshape(batch, heads, queries, 4, 3, grid, grid).cpu())

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def payload(self):
        if not self.weights:
            return None
        # [layers,heads,time_groups,cameras,rows,cols], first batch/query.
        values = torch.stack([weights[0, :, 0] for weights in self.weights])
        return dict(
            weights=values.tolist(),
            layers=len(values),
            heads=values.shape[1],
            grid=values.shape[-1],
            groups=4,
            description="Current-frame cross-attention to contextual video patch tokens",
        )
