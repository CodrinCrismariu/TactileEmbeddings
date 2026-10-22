# Coding our attention block by hand

This describes the two trainable decoder blocks after frozen V-JEPA in the
current-frame 32D and 128D models. The runnable implementation is
[examples/manual_attention_block.py](../examples/manual_attention_block.py).
It explicitly computes attention with linear projections, matrix multiplication,
masking, softmax, dropout, concatenation and an output projection. Its manual
classes do not call `nn.MultiheadAttention`, `nn.TransformerDecoderLayer`, or
fused attention. Linear layers, LayerNorm, GELU and dropout use PyTorch primitives.

The block receives two tensors and one optional camera-availability mask:

| Name | Shape | Meaning |
|---|---|---|
| `query` | `[B, T, 256]` | Prediction tokens; `T=1` for current-frame training |
| `memory` | `[B, N, 256]` | Already projected V-JEPA tokens plus position/camera embeddings |
| `padding_mask` | `[B, N]` | Boolean `True` for memory tokens the query must not read |

`B` is batch size. The actual layout reserves `N=4*3*24*24=6912` memory slots:
four temporal groups, three camera slots, and 576 spatial positions. Missing or
dropped cameras are masked. The input memory projection and position embeddings
are outside this block. The block returns an updated `[B,T,256]` query and
optionally the attention maps; it does not update `memory`.

1. **Normalize the incoming query.**

   Each query token has 256 coordinates. LayerNorm computes its mean and
   population variance across those coordinates, separately for each token and
   batch item. It then applies a learned scale and offset to each coordinate:

   ```python
   mean = x.mean(dim=-1, keepdim=True)
   variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
   normalized = (x - mean) / torch.sqrt(variance + 1e-5)
   normalized = normalized * learned_scale + learned_offset
   ```

   Each of the block's three LayerNorms has its own scale and offset. This
   normalizes features within a token, not attention weights or the token axis.

2. **Perform query self-attention and add its result to the original query.**

   ```python
   normalized = norm1(query)
   update = self_attention(normalized, normalized)
   query = query + residual_dropout1(update)
   ```

   In self-attention, query, key and value inputs all come from the same
   normalized query-token sequence. Their three projections are still different.
   For current-frame prediction there is only one query token. Each head's
   self-attention matrix is therefore `[1,1]`, and its softmax is exactly `[1]`
   before training dropout. This sublayer cannot choose between visual patches;
   it applies learned value/output transformations to the query. During training,
   attention dropout can drop its single connection in a head. Earlier models
   with eight output-query tokens instead have an `[8,8]` self-attention matrix.

3. **Normalize the updated query, then calculate cross-attention Q, K and V.**

   ```python
   q_input = norm2(query)           # [B,T,256]
   Q = q_proj(q_input)             # [B,T,256]
   K = k_proj(memory)              # [B,N,256]
   V = v_proj(memory)              # [B,N,256]
   ```

   Each projection is its own `nn.Linear(256,256)` with a learned weight matrix
   and bias. A linear layer computes `input @ weight.T + bias`. We do not apply
   a new query LayerNorm to `memory` in this operation. The input projection
   before the decoder has already normalized/projected the V-JEPA features.

   Query self-attention and cross-attention have different Q/K/V parameters.
   Block 1 and block 2 also have different parameters. Q/K/V vectors are computed
   from inputs each forward pass; the projection matrices are model parameters.

4. **Split the projected coordinates into four heads.**

   ```python
   Q = Q.reshape(B, T, 4, 64).transpose(1, 2)  # [B,4,T,64]
   K = K.reshape(B, N, 4, 64).transpose(1, 2)  # [B,4,N,64]
   V = V.reshape(B, N, 4, 64).transpose(1, 2)  # [B,4,N,64]
   ```

   Each head gets 64 projected features. Every head can use all 256 input
   coordinates through its learned projection; we are not restricting Head 1
   to the first 64 coordinates of the original input. Packing all four projections
   into a 256-output linear layer is equivalent to four 64-output linear layers.

5. **Compare each query with all keys, mask unavailable tokens and apply softmax.**

   ```python
   scores = (Q @ K.transpose(-2, -1)) / 8.0    # [B,4,T,N]
   scores = scores.masked_fill(padding_mask[:, None, None, :], -torch.inf)
   probabilities = scores.softmax(dim=-1)     # [B,4,T,N]
   ```

   For a particular batch item, head, query token and memory token, this computes
   a 64-term dot product and divides by `sqrt(64)=8`. Softmax runs over memory
   tokens, so it produces a relative distribution over all available cameras,
   temporal groups and spatial positions. Masked tokens get zero probability.
   Every item must retain at least one unmasked memory token.

   For current-frame training the cross-attention shape is `[B,4,1,N]`.
   **These probabilities are what the viewer visualizes.** It selects a layer,
   head, temporal group and camera, then reshapes the corresponding 576 values
   into a 24x24 map. Default mean-head visualization averages the maps, whereas
   the actual model combines value summaries as described below.

6. **Apply attention dropout during training.**

   ```python
   used_weights = F.dropout(probabilities, p=0.1, training=self.training)
   ```

   Individual attention connections are independently zeroed. Surviving weights
   are divided by 0.9; a realized row need not sum to one. The same patch can
   remain available to another head or the next block. `eval()` disables this
   dropout. Whole-patch dropout is not present in the current model.

7. **Use those weights to collect values.**

   ```python
   per_head = used_weights @ V                # [B,4,T,64]
   combined = per_head.transpose(1, 2).reshape(B, T, 256)
   update = out_proj(combined)                # [B,T,256]
   ```

   Each head produces the weighted sum of its memory value vectors. Four 64D
   summaries are concatenated, then a learned `Linear(256,256)` mixes their
   coordinates. The model does not average the heads' attention maps to produce
   its output. A large attention probability alone does not establish a large
   causal effect: values, output mixing and subsequent transformations matter.

8. **Add the cross-attention update through a residual connection.**

   ```python
   query = query + residual_dropout2(update)
   ```

   The existing query is preserved and the new result is added to it. The block
   uses normalization before each sublayer; the residual adds to the unnormalized
   running query. Camera/patch memory still has the same shape and values.

9. **Apply the feedforward MLP and its third residual update.**

   ```python
   update = norm3(query)                      # [B,T,256]
   update = linear1(update)                   # [B,T,1024]
   update = F.gelu(update)
   update = hidden_dropout(update)
   update = linear2(update)                   # [B,T,256]
   query = query + residual_dropout3(update)
   ```

   This MLP transforms each query token independently. It does not read patches
   or exchange information between tokens. The hidden and residual dropouts
   have probability 0.1 during training and are disabled during evaluation.

10. **Repeat with block 2, then apply the separate latent MLP.**

    ```python
    query, maps1 = block1(query, memory, padding_mask)
    query, maps2 = block2(query, memory, padding_mask)
    latent = latent_mlp(query)                 # [B,1,32] or [B,1,128]
    ```

    Block 2 reads the same base memory using its own key/value projections and
    the updated query from block 1. Heads are combined between blocks; Head 1
    in block 2 does not receive only Head 1 from block 1. The initial query for
    block 1 is a shared learned vector plus current-slot embeddings. Its first
    cross-attention query is input-independent during evaluation, but its keys
    depend on the clip. Block 2's query also depends on the clip through block 1.

    The final latent MLP is `LayerNorm(256) -> Linear(256,128) -> GELU ->
    Linear(128,latent_dim)`. It differs from each block's `256 -> 1024 -> 256`
    feedforward MLP and has no dropout. Neither block nor the latent MLP receives
    tactile readings as inputs.

The runnable example verifies the explicit implementation against
`nn.TransformerDecoderLayer(norm_first=True, activation='gelu')`. It checks output
and input-gradient agreement, probability normalization, masked-token zeros,
the one-query self-attention case, and replay of training dropout. It also loads
the actual trained blocks from both models and compares their two-layer outputs
on synthetic visual memory with the full 6912-token layout, including ego-only
camera masking. This is mathematical equivalence verification, not an additional
dataset or transfer evaluation.

From the repository root:

```bash
.venv/bin/python examples/manual_attention_block.py \
  --checkpoint processing/runs/joint_vjepa21_spatial_current32d_v1/viewer_epochs/epoch_028.pt \
  --checkpoint processing/runs/joint_vjepa21_spatial_current128d_v1/viewer_epochs/epoch_030.pt \
  --report processing/attention_block_explained/verification.json
```

Verification uses CPU float64, with dropout disabled for reference output and
gradient comparisons. Matching dropout probabilities and placement does not
promise identical random masks to a different fused GPU kernel. The largest
observed two-layer output difference using trained weights was about `1.4e-15`.
The report is [verification.json](../processing/attention_block_explained/verification.json).
The production implementation remains in
[vjepa_spatial.py](../src/tactilelatent/models/attention.py).
