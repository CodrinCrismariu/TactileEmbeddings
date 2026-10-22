# The current tactile model, taught from first principles

This is a guide to the **code that exists now**, not a description of every
idea in `main.pdf`. It corresponds to
`configs/egotouch_vae_image_only_latent32.json` and the implementation in
`src/tactilelatent/model.py`, `train.py`, `data.py`, and `losses.py`. Parameter
counts below were measured from the instantiated models, including the pinned
DINOv3 checkpoint. Shapes use the normal training batch size **B = 2**, clip
length **T = 8**, and all three camera slots. A batch dimension is retained
unless explicitly stated. `MiB` means 1,048,576 bytes.

This guide describes the **established EgoTouch-only model**. A separate
six-source VAE extension with native sensor decoders is prepared in
`joint_model.py`; see `JOINT_DATASETS.md` for that experiment.

## 1. What is this model trying to do?

Imagine watching someone grasp an object. You can often infer that there is
contact, which hand is involved, and whether pressure is probably increasing.
You cannot directly *see* the pressure grid inside a glove. The model learns
statistical clues from RGB video, then predicts the pressure that was actually
measured during training.

The current experiment uses **EgoTouch only**. It has three synchronized video
views—chest, left wrist, right wrist—and two 21×21 pressure grids, one for each
hand. The model's input is RGB images plus a three-camera availability mask.
The pressure grids are **supervision**, not an input to the encoder. Hand pose,
language, actions, and robot state are not used.

One picture of the forward pass:

```text
8 times × 3 RGB cameras
        │
        ▼
frozen DINOv3-B/16, independently on each image
        │   196 image-patch vectors per image, each 768 numbers
        ▼
learned cross-camera weighting (2 transformer layers)
        │   one fused 14×14 patch grid per time
        ▼
3 learned space-time transformer blocks
        │   patches can exchange information through time and space
        ▼
pool patches → two 32-number vectors, μ and log-variance
        │   these describe one Gaussian distribution per time
        ▼
sample a 32-number z during training; use μ by default at evaluation
        │
        ▼
MLP decoder 32 → 512 → 882
        │
        ▼
8 times × 2 hands × 21 × 21 predicted pressure values
```

There is **one shared visual encoder and one pressure decoder**, not separate
ego-view and all-view encoders. Camera masking changes which views can
contribute, but uses the same weights.

## 2. Small vocabulary before the shapes

- **Tensor shape:** a way to say how numbers are arranged. `[2,8,3,3,224,224]`
  means 2 clips, 8 times, 3 cameras, 3 color channels, 224 rows, 224 columns.
- **Feature/vector/token:** a learned numerical description. A 768D patch
  token is 768 numbers describing one image region after DINO has examined it.
- **Patch:** here a non-overlapping 16×16-pixel square. A 224×224 image has
  14×14 = **196** patches.
- **Attention:** a learned operation that lets one token use information from
  other tokens. Heads are parallel attention subspaces; they do **not** add
  extra tokens to the output.
- **Transformer block:** attention plus a feed-forward network, normalization,
  and residual connections. Residual means adding the new information back to
  the block's input.
- **Pooling:** combining many tokens into a smaller description; here by
  averaging patch tokens.
- **Latent bottleneck:** an intermediate representation restricted to 32
  numbers per time. Every predicted pressure pixel must depend on these 32
  numbers once the image has passed through the encoder.
- **Gaussian distribution:** specified by a mean `μ` and variance `σ²`. The
  model outputs `log(σ²)` for numerical stability, then samples with
  `z = μ + exp(0.5 logvar) × ε`, where `ε ~ N(0,I)`.
- **Parameter:** a learned weight or bias, reused for every clip. Parameter
  count is *not* the same as the number of activation values for a batch.

## 3. Exact input and DINOv3 backbone

`RawEgoTouch` selects eight video frames at source indices
`s, s+2, …, s+14`. Thus the observations span **15 original frame positions**
from first to last. It decodes and rescales each view to 224×224 and returns:

| Item | Shape, B=2 | Meaning |
| --- | ---: | --- |
| `rgb` | `[2,8,3,3,224,224]` | Values in `[0,1]`; camera order chest, left, right. |
| `view_mask` | `[2,8,3]` | Boolean flag: may this camera contribute at this time? |
| `target` | `[2,8,2,21,21]` | Measured, normalized left/right pressure; used only for loss. |
| `target_mask` | `[2,8,2,21,21]` | Which target cells are finite/valid. |

The backbone is pinned to `facebook/dinov3-vitb16-pretrain-lvd1689m` at
revision `5931719e67bbdb9737e363e781fb0c67687896bc`. Its weights are
**frozen** (`requires_grad=False`); it stays in evaluation mode. Each RGB image
is normalized by the ImageNet channel means and standard deviations before
entering DINO. In a B=2 batch there are 2×8×3 = **48 images**; the training
helper processes them in chunks of at most eight images through DINO.

Inside one DINO image:

| DINO stage | Input → output for one image | Learned parameters |
| --- | --- | ---: |
| Patch embedding convolution, 3→768, kernel/stride 16 | `[1,3,224,224]` → `[1,768,14,14]` → 196 patch tokens `[1,196,768]` | 590,592 |
| CLS, 4 register, and mask-token parameters | CLS + registers are included in the ordinary token sequence; the mask token is stored but not used for ordinary unmasked images | 4,608 |
| 12 DINOv3 transformer blocks | `[1,201,768]` → `[1,201,768]` in every block | 7,088,640 **each**; 85,063,680 total |
| Final layer normalization | `[1,201,768]` → `[1,201,768]` | 1,536 |
| DINO total | **201 = 1 CLS + 4 registers + 196 patches** | **85,660,416, all frozen** |

Each DINO transformer block has 12 attention heads of width 64
(`12×64=768`), attention projections, a `768→3072→768` feed-forward network,
two normalizations, and two LayerScale vectors. Its parameter breakdown is
2,361,600 attention + 4,722,432 feed-forward + 3,072 normalizations + 1,536
LayerScale = **7,088,640 per block**. The DINO wrapper discards the CLS and
register outputs and keeps the **last 196 patch tokens**, so its output for
the training batch is `[2,8,3,196,768]`.

Why use DINO? The frozen pretrained transformer supplies informative visual
features without learning image recognition from scratch on the much smaller
tactile dataset. Why keep patches instead of just one image vector? Contact
location matters; keeping 14×14 spatial tokens preserves more location detail
for the later transformer.

## 4. Exact trainable architecture, layer by layer

The next table follows `EgoTouchLatentModel.forward` literally. Shapes are
for B=2, T=8, V=3, N=196, width D=768. Repeated block parameter counts are
per copy unless a total is shown. Nonlearned operations such as reshaping,
averaging, softmax, and dropout have **zero** parameters.

| Layer or operation | Tensor in → tensor out | Parameters | What it contributes |
| --- | --- | ---: | --- |
| `project`: Linear 768→768 | `[2,8,3,196,768]` → same shape | 590,592 | Trainable adaptation of frozen DINO features. |
| `view_embeddings` | Add `[3,768]` to corresponding camera tokens | 2,304 | Identifies which camera produced each token. |
| Mean over 196 patches | `[2,8,3,196,768]` → `[2,8,3,768]` | 0 | One camera summary per time. |
| Reshape for camera attention | `[2,8,3,768]` → `[16,3,768]` | 0 | Treat 16 clip-times as separate 3-token sequences. |
| `cross_view`: 2 transformer encoder layers | `[16,3,768]` → same | 7,087,872 each; **14,175,744** total | Camera summaries can attend to one another. |
| `view_gate`: Linear 768→192, GELU, Linear 192→1 | `[2,8,3,768]` → `[2,8,3,1]` | 147,841 | Scores each camera at each time. |
| Mask + softmax over 3 cameras | `[2,8,3,1]` → camera weights `[2,8,3]` | 0 | Absent cameras receive zero weight; present weights sum to 1. |
| Weighted camera sum | `[2,8,3,196,768]` → `[2,8,196,768]` | 0 | Produces a fused patch grid per time. |
| `view_norm` LayerNorm | `[2,8,196,768]` → same | 1,536 | Stabilizes fused features. |
| Sinusoidal temporal embedding | Add a slice of `[64,768]` to each time's patches | 0 trainable; 49,152 nonpersistent buffer values | Distinguishes frame positions 0–7. |
| `temporal`: 3 divided space-time blocks | `[2,8,196,768]` → same in each block | 9,451,776 each; **28,355,328** total | Shares information over time and image locations. |
| `temporal_norm` LayerNorm | `[2,8,196,768]` → same | 1,536 | Stabilizes final patch features. |
| Mean over 196 patches | `[2,8,196,768]` → `[2,8,768]` | 0 | Global visual context for each time. |
| `image_pool`: LayerNorm, Linear 768→768, GELU | `[2,8,768]` → same | 592,128 | Refines pooled context. |
| `prior_mu`: Linear 768→32 | `[2,8,768]` → `[2,8,32]` | 24,608 | Mean of image-derived latent Gaussian. |
| `prior_logvar`: Linear 768→32, clamp to `[-8,4]` | `[2,8,768]` → `[2,8,32]` | 24,608 | Log-variance of that Gaussian. |
| Reparameterized sample `z`, or mean `z=μ` | Two `[2,8,32]` vectors → one `[2,8,32]` | 0 | 32D bottleneck, one latent per time. |
| `decoder.0`: Linear 32→512, GELU, dropout 0.1 | `[2,8,32]` → `[2,8,512]` | 16,896 | Expands latent into decoder hidden features. |
| `decoder.3`: Linear 512→882, sigmoid | `[2,8,512]` → `[2,8,882]` | 452,466 | Produces bounded pressure values. |
| Reshape | `[2,8,882]` → `[2,8,2,21,21]` | 0 | `882=2×21×21`, left and right hands. |

Parameter check: **44,385,587 trainable** after DINO; combined with frozen
DINO the forward pipeline contains **130,046,003 parameters**. There is no
separate learned pressure model for each camera, no hand-pose branch, and no
second distribution encoder.

### 4.1 What exactly happens in the cross-view block?

For each time, the model first averages all 196 patches from each camera.
Those three summaries then enter a 2-layer transformer. An attention head can
ask, for example, whether the left-wrist view is informative given what the
chest and right-wrist views show. The output summaries are turned into three
scores, normalized with softmax. These **three scalar weights** are applied to
the original projected patch tokens and summed camera-wise.

This is subtle but important: cross-camera attention is over **three global
camera summaries**, not direct patch-to-patch attention between views. The
same camera weight applies to all 196 patch positions at that time. Thus the
model can choose a view globally per frame, but it cannot choose chest for one
patch and wrist for another using this gate. When evaluating `ego`, the wrists
are masked, so the chest weight is exactly 1. For `all`, all three may
contribute. The camera-attention layers have 8 heads of width 96
(`8×96=768`), then a `768→3072→768` feed-forward network.

One cross-view transformer layer's exact parameters are **2,362,368** for
attention projections, **4,722,432** for the feed-forward network, and
**3,072** for two layer norms, totaling **7,087,872**. It retains shape
`[16,3,768]` throughout except for internal Q/K/V projections
`[16,3,2304]` and the feed-forward hidden tensor `[16,3,3072]`.

### 4.2 What exactly happens in a space-time block?

Each of the three blocks performs two different kinds of attention, in order:

1. **Time attention at each patch position.** Rearrange
   `[2,8,196,768]` to `[392,8,768]`, where `392=2 clips×196 positions`.
   Each length-8 sequence shares information among the eight times at that
   patch position. The result is restored to a `[2,8,196,768]` grid.
2. **Space attention within each time.** Rearrange to `[16,196,768]`,
   where `16=2 clips×8 times`. Each length-196 sequence lets image locations
   interact. Then a `768→3072→768` feed-forward network acts on every token,
   and the tensor is restored to `[2,8,196,768]`.

Each attention operation has 8 heads of width 96. The block has three layer
norms, two attention modules and one feed-forward module. Its exact parameter
count is **4,724,736** for both attention modules together + **4,722,432**
for feed-forward + **4,608** for the norms = **9,451,776**.

There is **no causal attention mask**. At time position 0, temporal attention
can inspect images at positions 1–7. Only the prediction at the last position
is based solely on images at or before that position. A strictly online,
per-frame interpretation would require changing the architecture and
retraining.

### 4.3 How does a Gaussian latent make pressure predictions?

After attention, averaging the 196 patch tokens gives one 768-number context
vector per time. Two different linear heads create a 32-number `μ` and a
32-number `logvar`. Because log-variance is clamped to `[-8,4]`, the implied
standard deviation per latent coordinate lies roughly between `0.018` and
`7.39` before training losses constrain it further. The 32 coordinates are
independent **conditional on the images** in the Gaussian head; correlations
between pressure cells may still arise after decoding the same sampled `z`.

During training, `z = μ + σ×ε`, which allows gradients to pass through `μ`
and `σ` despite the random draw. During ordinary validation/evaluation, the
model uses `z=μ`, giving a repeatable point prediction. The uncertainty
evaluator draws many `z` samples and decodes each. The decoder is fully
connected—not an image convolution—and turns one 32D vector into both hands'
pressure maps. Its final sigmoid constrains each cell to `[0,1]`. These values
are **normalized pressure**, not pressure in pascals or force in newtons.

## 5. Exact sizes: parameters, batch activations, and checkpoints

| Component | Parameters | Trainable? | fp32 weight storage only |
| --- | ---: | --- | ---: |
| DINO patch/tokens/final norm | 596,736 | No | 2.28 MiB |
| DINO's 12 transformer blocks | 85,063,680 | No | 324.49 MiB |
| All DINOv3 | **85,660,416** | **No** | **326.77 MiB** |
| Trainable projection + view embeddings | 592,896 | Yes | 2.26 MiB |
| Cross-view transformer, 2 layers | 14,175,744 | Yes | 54.08 MiB |
| Camera gate + view normalization | 149,377 | Yes | 0.57 MiB |
| Space-time transformer, 3 blocks | 28,355,328 | Yes | 108.17 MiB |
| Temporal norm + image pool | 593,664 | Yes | 2.26 MiB |
| Gaussian mean + log-variance heads | 49,216 | Yes | 0.19 MiB |
| Pressure decoder | 469,362 | Yes | 1.79 MiB |
| Trainable model after DINO | **44,385,587** | **Yes** | **169.32 MiB** |
| Whole inference pipeline | **130,046,003** | Mixed | **496.09 MiB** |

For the standard **B=2** batch, examples of activation sizes are:

| Tensor | Elements | fp32 storage for one such tensor |
| --- | ---: | ---: |
| RGB `[2,8,3,3,224,224]` | 7,225,344 | 27.56 MiB |
| DINO patch features `[2,8,3,196,768]` | 7,225,344 | 27.56 MiB |
| Fused features `[2,8,196,768]` | 2,408,448 | 9.19 MiB |
| Camera summaries `[2,8,3,768]` | 36,864 | 0.14 MiB |
| One latent mean `[2,8,32]` | 512 | 0.002 MiB |
| Pressure output `[2,8,2,21,21]` | 14,112 | 0.054 MiB |

The table is **not GPU peak memory**: attention needs Q/K/V and temporary
workspaces, training stores intermediate activations and gradients, and AdamW
stores optimizer state. CUDA uses bfloat16 autocast for many operations, but
the stored model weights remain fp32. A nominal optimizer step covers
`batch_size=2 × accumulation=144 = 288` clips; **only two clips, not 288,
are in a microbatch at once**. The full checkpoint contains model weights,
AdamW state, scheduler, epoch, configuration, and random-number state. DINO's
frozen weights are loaded separately from Hugging Face, not saved in the
training checkpoint.

### 5.1 Exact learned weight shapes inside repeated layers

The input/output table above shows activation shapes. This appendix shows the
**weight matrices and vectors themselves**. PyTorch stores a linear layer's
weight as `[out_features,in_features]`; a bias has shape `[out_features]`.
These shapes repeat identically for each copy of the named block.

| DINOv3 component | Weight and bias shapes | Parameters per block |
| --- | --- | ---: |
| Patch convolution | weight `[768,3,16,16]`, bias `[768]` | 590,592, once before the blocks |
| CLS, register, mask tokens | `[1,1,768]`, `[1,4,768]`, `[1,1,768]` | 4,608, once before the blocks |
| Attention K projection | weight `[768,768]`, **no bias** | 589,824 |
| Attention Q, V, and output projections | each weight `[768,768]` and bias `[768]` | 590,592 each; 1,771,776 together |
| Feed-forward up | weight `[3072,768]`, bias `[3072]` | 2,362,368 |
| Feed-forward down | weight `[768,3072]`, bias `[768]` | 2,360,064 |
| Two LayerNorms | each weight `[768]`, bias `[768]` | 3,072 together |
| Two LayerScale vectors | each `[768]` | 1,536 together |
| DINO transformer block total | 12 identical blocks | **7,088,640 each** |
| Final DINO LayerNorm | weight `[768]`, bias `[768]` | 1,536, once after the blocks |

DINO's internal token activations are `[number_of_images,201,768]` in each
block; with the training helper's maximum image chunk of eight, this is
`[8,201,768]`. Its internal Q/K/V each have that same last dimension, and
its feed-forward expansion is `[8,201,3072]`. Rotary position encoding adds
no learned weights here.

| Cross-view transformer component | Weight and bias shapes | Parameters per layer |
| --- | --- | ---: |
| Attention fused Q/K/V projection | weight `[2304,768]`, bias `[2304]` | 1,771,776 |
| Attention output projection | weight `[768,768]`, bias `[768]` | 590,592 |
| Feed-forward first linear | weight `[3072,768]`, bias `[3072]` | 2,362,368 |
| Feed-forward second linear | weight `[768,3072]`, bias `[768]` | 2,360,064 |
| Two LayerNorms | each weight `[768]`, bias `[768]` | 3,072 together |
| Cross-view layer total | 2 identical layers | **7,087,872 each** |

| Space-time block component | Weight and bias shapes | Parameters per block |
| --- | --- | ---: |
| Temporal attention fused Q/K/V + output | `[2304,768]`/`[2304]` and `[768,768]`/`[768]` | 2,362,368 |
| Spatial attention fused Q/K/V + output | same shapes | 2,362,368 |
| Feed-forward first linear | `[3072,768]`/`[3072]` | 2,362,368 |
| Feed-forward second linear | `[768,3072]`/`[768]` | 2,360,064 |
| Three LayerNorms | each weight `[768]`, bias `[768]` | 4,608 together |
| Space-time block total | 3 identical blocks | **9,451,776 each** |

The remaining learned components occur once in the model:

| One-off component | Exact learned shapes | Parameters |
| --- | --- | ---: |
| DINO feature projection | `[768,768]`/`[768]` | 590,592 |
| Three camera embeddings | `[3,768]` | 2,304 |
| Camera gate's two linear layers | `[192,768]`/`[192]`, then `[1,192]`/`[1]` | 147,841 |
| Fused-view normalization | `[768]` weight + `[768]` bias | 1,536 |
| Final temporal normalization | `[768]` weight + `[768]` bias | 1,536 |
| Image-pool normalization and linear | `[768]`/`[768]`, then `[768,768]`/`[768]` | 592,128 |
| Gaussian mean head | `[32,768]`/`[32]` | 24,608 |
| Gaussian log-variance head | `[32,768]`/`[32]` | 24,608 |
| Decoder's two linear layers | `[512,32]`/`[512]`, then `[882,512]`/`[882]` | 469,362 |

## 6. How learning works, step by step

1. Read a clip and its measured pressure grids. Randomly hide each wrist view
   with probability 0.3 during training; chest remains available.
2. Run images through frozen DINO, then the trainable cross-view, space-time,
   Gaussian, and decoder modules.
3. Compare the predicted pressure grid with valid measured cells. Pressure
   greater than `0.1` gets weight **3**, other valid cells weight **1**.
   The regression term is weighted `squared error + 0.5×absolute error`,
   divided by the count of valid cells.
4. Add `0.01×` spatial total-variation loss, discouraging unnecessarily
   jagged neighboring pressure predictions.
5. Add `β×KL[N(μ,diag(exp(logvar))) || N(0,I)]`, averaged across batch,
   times, and the 32 latent coordinates. `β` rises from 0 to `0.001` over
   ten epochs. This keeps the latent distribution somewhat regularized.
6. Backpropagate only into the **44.39M trainable** parameters after DINO.
   AdamW updates them after gradient accumulation, with a learning-rate
   warmup followed by cosine decay. The active config is 20 epochs, but
   simply having that config does not mean a new run has started.

At evaluation the point metrics are MAE, contact IoU/F1, and volumetric IoU.
Contact is thresholded at `0.05` for evaluation, **not** the `0.1` used for
training's contact-weighted loss. The uncertainty evaluator also samples the
Gaussian and reports CRPS, interval coverage, sample spread and other
diagnostics.

## 7. What “VAE” means here—and what it does not

The code uses a VAE-like Gaussian latent, reparameterization, a decoder, and
KL regularization. However, a textbook variational autoencoder would infer a
latent from the observed thing it reconstructs; a conditional version might
use images **and** observed pressure in a training-time inference network.
This active model has **no pressure-informed inference network**. It learns
`q(z | images)` directly and trains samples from that distribution to predict
pressure. Pressure enters only through the reconstruction loss.

That choice is simple and matches the image-conditioned latent formula in
`main.pdf`, but it has a limitation: given one training pressure map for a
visual clip, minimizing pointwise reconstruction loss may reward a narrow or
averaged prediction. A nonzero Gaussian width does **not** prove that the
model has captured all plausible hidden forces. The measured uncertainty
calibration matters more than the name “VAE.”

Also, the latent is **not yet proven dataset-invariant**. This phase uses one
dataset and one decoder. The proposal's future multi-dataset decoders, domain
adversary, and physical probes are not present in the current model.

## 8. A concrete mental model

At one time in a clip, DINO turns each camera image into 196 descriptions of
what appears in each region. The view gate decides, once for that time, how
much to trust each camera. Three space-time blocks let each region incorporate
nearby image regions and all eight observed times. The model averages those
regional descriptions and writes down **32 means and 32 uncertainties**.
It then chooses one 32-number latent and asks a small decoder to paint two
21×21 pressure grids. During learning, the real measured grids tell the
network how wrong that painting was. Everything trainable before the latent
learns to preserve the visual clues that the pressure decoder needs.

The simplest way to remember the architecture is:
**frozen image understanding → learned camera/time reasoning → 32D uncertain
summary → two pressure maps**.
