"""An explicit implementation of the current visual readout's attention block.

No nn.MultiheadAttention, nn.TransformerDecoderLayer, or fused attention is used
inside the manual implementation. PyTorch's block is imported only as a
reference for the executable verification. This file is an educational example;
the production model does not import it.

Run from the repository root:
    .venv/bin/python examples/manual_attention_block.py
    .venv/bin/python examples/manual_attention_block.py --checkpoint PATH.pt
"""
import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ManualAttention(nn.Module):
    def __init__(self, width=256, heads=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("The width must be divisible by the head count")
        self.heads = heads
        self.head_width = width // heads
        # All heads' projections are stored together. Splitting the last axis
        # below assigns a separate group of output coordinates to each head.
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_width).transpose(1, 2)

    def forward(self, query, memory, padding_mask=None):
        # query: [B,T,256]; memory: [B,N,256]. Current-frame training has T=1.
        q = self.split_heads(self.q_proj(query))   # [B,4,T,64]
        k = self.split_heads(self.k_proj(memory))  # [B,4,N,64]
        v = self.split_heads(self.v_proj(memory))  # [B,4,N,64]

        # Compare every query with every key, independently within each head.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        if padding_mask is not None:
            # Boolean True means an unavailable memory token. Mask scores,
            # rather than zeroing embeddings: biases and positions are nonzero.
            if padding_mask.dtype != torch.bool or padding_mask.shape != memory.shape[:2]:
                raise ValueError("Expected a boolean [B,N] padding mask")
            if padding_mask.all(-1).any():
                raise ValueError("Every example needs at least one available memory token")
            scores = scores.masked_fill(padding_mask[:, None, None, :], -torch.inf)

        probabilities = scores.softmax(dim=-1)  # [B,4,T,N]; sums to 1 over N
        # Training dropout zeros individual connections and rescales survivors.
        # It does NOT re-normalize each row to 1. eval() disables this dropout.
        used_weights = self.attention_dropout(probabilities)
        per_head = used_weights @ v             # [B,4,T,64]

        # Concatenate, then learn how to mix the heads' feature summaries.
        batch, _, queries, _ = per_head.shape
        combined = per_head.transpose(1, 2).reshape(batch, queries, -1)
        return self.out_proj(combined), probabilities

    @torch.no_grad()
    def copy_from_pytorch(self, source):
        # PyTorch packs Q/K/V projections vertically into one [3D,D] matrix.
        for i, destination in enumerate((self.q_proj, self.k_proj, self.v_proj)):
            width = source.embed_dim
            destination.weight.copy_(source.in_proj_weight[i * width:(i + 1) * width])
            destination.bias.copy_(source.in_proj_bias[i * width:(i + 1) * width])
        self.out_proj.load_state_dict(source.out_proj.state_dict())


class ManualDecoderBlock(nn.Module):
    def __init__(self, width=256, heads=4, feedforward=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.norm3 = nn.LayerNorm(width, eps=1e-5)
        # Self-attention and cross-attention have separate learned parameters.
        self.self_attention = ManualAttention(width, heads, dropout)
        self.cross_attention = ManualAttention(width, heads, dropout)
        self.linear1 = nn.Linear(width, feedforward)
        self.linear2 = nn.Linear(feedforward, width)
        self.hidden_dropout = nn.Dropout(dropout)
        self.residual_dropout1 = nn.Dropout(dropout)
        self.residual_dropout2 = nn.Dropout(dropout)
        self.residual_dropout3 = nn.Dropout(dropout)

    def forward(self, query, memory, padding_mask=None):
        # 1. Pre-normalized query self-attention and its residual update.
        normalized = self.norm1(query)
        update, self_weights = self.self_attention(normalized, normalized)
        query = query + self.residual_dropout1(update)

        # 2. Pre-normalize the updated query, then read the fixed visual memory.
        update, cross_weights = self.cross_attention(self.norm2(query), memory, padding_mask)
        query = query + self.residual_dropout2(update)

        # 3. Independent feedforward transformation of each query token.
        update = self.linear1(self.norm3(query))
        update = self.hidden_dropout(F.gelu(update))
        update = self.linear2(update)
        query = query + self.residual_dropout3(update)
        return query, {"self": self_weights, "cross": cross_weights}

    def copy_from_pytorch(self, source):
        self.self_attention.copy_from_pytorch(source.self_attn)
        self.cross_attention.copy_from_pytorch(source.multihead_attn)
        for name in ("norm1", "norm2", "norm3", "linear1", "linear2"):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict())
        return self


def reference_block(width=256, heads=4, feedforward=1024, dropout=0.1):
    return nn.TransformerDecoderLayer(width, heads, feedforward, dropout,
        activation="gelu", batch_first=True, norm_first=True).double().eval()


def verify_basic():
    torch.manual_seed(45)
    reference = reference_block()
    manual = ManualDecoderBlock().double().eval().copy_from_pytorch(reference)
    q = torch.randn(2, 1, 256, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(2, 31, 256, dtype=torch.float64, requires_grad=True)
    mask = torch.zeros(2, 31, dtype=torch.bool)
    mask[0, 7:14] = True
    mask[1, 20:] = True

    expected = reference(q, memory, memory_key_padding_mask=mask)
    q_manual = q.detach().clone().requires_grad_(True)
    m_manual = memory.detach().clone().requires_grad_(True)
    actual, maps = manual(q_manual, m_manual, mask)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(q_manual.grad, q.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(m_manual.grad, memory.grad, atol=1e-10, rtol=1e-10)
    assert torch.equal(maps["self"], torch.ones_like(maps["self"]))
    torch.testing.assert_close(maps["cross"].sum(-1), torch.ones(2, 4, 1, dtype=torch.float64))
    assert (maps["cross"].masked_select(mask[:, None, None, :]) == 0).all()

    # Verify training-mode dropout can be replayed, as SAM requires. This does
    # not assert identical dropout masks to a different fused PyTorch kernel.
    manual.train()
    with torch.no_grad():
        rng = torch.get_rng_state()
        first, _ = manual(q.detach(), memory.detach(), mask)
        torch.set_rng_state(rng)
        replay, _ = manual(q.detach(), memory.detach(), mask)
    torch.testing.assert_close(first, replay, atol=0, rtol=0)
    return dict(passed=True, output_max_error=float((actual - expected).abs().max().detach()),
        query_gradient_max_error=float((q_manual.grad - q.grad).abs().max()),
        memory_gradient_max_error=float((m_manual.grad - memory.grad).abs().max()),
        attention_shape=list(maps["cross"].shape), masked_tokens_zero=True,
        self_attention_single_query_is_one=True, training_dropout_replay=True)


@torch.no_grad()
def verify_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    config, weights = state["model_config"], state["model"]
    if config.get("prediction_mode") != "current_frame":
        raise ValueError("This demonstration expects a current-frame checkpoint")
    width, heads, ff = config["width"], config["heads"], config["feedforward"]
    # Load the actual trained initial query and its current-frame slot tags.
    query = (weights["visual.frame_queries"][0] + weights["visual.frame_position"][7]
             + weights["visual.pair_slot.weight"][1]).double()[None, None].expand(2, -1, -1)
    # Synthetic visual memory exercises the real 4 x 3 x 24 x 24 layout. One
    # example uses all cameras; the other masks the two wrist-camera slots.
    torch.manual_seed(45)
    tokens = 4 * 3 * config["grid_size"] ** 2
    memory = torch.randn(2, tokens, width, dtype=torch.float64)
    mask = torch.zeros(2, 4, 3, config["grid_size"] ** 2, dtype=torch.bool)
    mask[1, :, 1:] = True
    mask = mask.flatten(1)
    expected, actual = query.clone(), query.clone()
    errors = []
    for index in range(config["layers"]):
        reference = reference_block(width, heads, ff, config["dropout"])
        prefix = f"visual.decoder.layers.{index}."
        reference.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)})
        manual = ManualDecoderBlock(width, heads, ff, config["dropout"]).double().eval().copy_from_pytorch(reference)
        expected = reference(expected, memory, memory_key_padding_mask=mask)
        actual, maps = manual(actual, memory, mask)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        errors.append(float((actual - expected).abs().max()))
        assert (maps["cross"].masked_select(mask[:, None, None, :]) == 0).all()
    return dict(passed=True, checkpoint=str(path), epoch=state["epoch"] + 1,
        latent_dim=config["latent_dim"], memory_shape=list(memory.shape),
        query_shape=list(actual.shape), cross_attention_shape=list(maps["cross"].shape),
        layer_output_max_errors=errors, note="Trained block weights; synthetic visual memory; CPU float64")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(basic=verify_basic(), checkpoints=[verify_checkpoint(p) for p in args.checkpoint])
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
