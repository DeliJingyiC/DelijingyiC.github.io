---
layout: blog-post
title: "EnCodec: High-Fidelity Neural Audio Codec with Streaming and Variable Bitrate"
category: "Speech Synthesis"
date: 2026-06-01
description: "A technical deep dive into EnCodec (Défossez et al., 2022) — the neural audio codec behind VALL-E and VoiceCraft. Covers encoder/decoder architecture, RVQ, MS-STFT discriminator, loss balancer, streaming mode, and ablation results."
read_time: "18 min read"
---

## 1. Motivation

With the growth of internet traffic, audio and video streams now account for a large share of bandwidth. Efficient audio compression is critical for reducing storage and improving user experience, especially on poor connections.

Traditional codecs such as **Opus** and **EVS** degrade significantly at low bitrates — quality is noticeably poor at 3 kbps, particularly for non-speech audio like music. These codecs were designed before the neural network era and are difficult to optimize further.

EnCodec replaces the hand-crafted components of a traditional codec with a neural network trained end-to-end, achieving higher perceptual quality at the same bitrate. It has since become the de-facto codec for autoregressive speech synthesis systems such as VALL-E, VoiceCraft, and AudioLM.

---

## 2. Model Architecture

An audio signal of duration $d$ is represented as $\mathbf{x} \in [-1, 1]^{C_a \times T}$, where $C_a$ is the number of audio channels and $T = d \cdot f_\text{sr}$ is the total number of samples. EnCodec has three main components:

- **Encoder $E$** — takes a segment of audio and outputs a continuous latent $\mathbf{z}$
- **Quantizer $Q$** — converts $\mathbf{z}$ into a compressed discrete representation $\mathbf{z}_q$ via Residual Vector Quantization (RVQ)
- **Decoder $G$** — reconstructs the time-domain waveform $\hat{\mathbf{x}}$ from $\mathbf{z}_q$

The system is trained end-to-end, minimizing both reconstruction loss and perceptual loss via multi-scale discriminators.

```
                     Encoder E                Quantizer Q             Decoder G
                ┌─────────────────┐       ┌──────────────────┐   ┌─────────────────┐
                │  Conv1d(C=32,7) │       │  RVQ             │   │  Conv1d(D, 7)   │
Audio x  ──────▶│  ConvBlock × 4  │──z──▶ │  N_q codebooks   │─z_q▶ ConvBlock × 4 │──▶ x̂
                │  (s=2, 4, 5, 8) │       │  1024 entries ea.│   │  (s=8, 5, 4, 2) │
                │  LSTM × 2       │       └──────────────────┘   │  LSTM × 2       │
                │  Conv1d(D, 7)   │                               │  Conv1d(1, 7)   │
                └─────────────────┘                               └────────┬────────┘
                                                                           │  x̂
                                                              ┌────────────▼──────────────┐
                                                              │    MS-STFT Discriminator   │
                                                              │  windows: [2048..128]      │
                                                              │  6 × Conv2D sub-networks   │
                                                              └───────────────────────────┘
         Losses: ℓ_t (time domain)  ℓ_f (freq domain)  ℓ_g (adversarial)  ℓ_feat (feature)  ℓ_w (VQ)
```

### 2.1 Encoder

The encoder $E$ is a 1D convolutional network:

1. **Initial conv**: 1D conv with $C = 32$ channels, kernel size 7
2. **$B = 4$ convolutional blocks**, each consisting of:
   - A **residual unit**: two kernel-3 convolutions with a skip connection
   - A **downsampling layer**: strided conv with stride $S$ and kernel size $2S$ (channels double after each downsampling)
   - Stride sequence: $(2, 4, 5, 8)$ → total downsampling factor of $320$
3. **2-layer LSTM** for temporal sequence modeling
4. **Final conv**: kernel-7 1D conv, $D$ output channels

At 24 kHz, the encoder outputs **75 latent steps/second**; at 48 kHz, **150 steps/second**.

**Activation**: ELU. **Normalization**: LayerNorm (non-streaming) or WeightNorm (streaming).

### 2.2 Decoder

The decoder $G$ is symmetric to the encoder — strided convolutions are replaced by transposed convolutions, applied in **reverse stride order** $(8, 5, 4, 2)$. The final layer outputs a single-channel (mono) or stereo waveform.

---

## 3. Residual Vector Quantization

The encoder output $\mathbf{z} \in [B, D, T]$ is quantized via RVQ into a discrete token sequence $[B, N_q, T]$:

```
z              → Codebook 1 → code_1,   residual_1 = z − decode(code_1)
residual_1     → Codebook 2 → code_2,   residual_2 = residual_1 − decode(code_2)
...
residual_{N-1} → Codebook N → code_N
```

**Training details:**

- Codebook entries are updated via **EMA** with decay 0.99: unused entries are replaced by random samples from the current batch
- The **straight-through estimator** is used to pass gradients through the quantization step to the encoder
- A **commitment loss** is added (see Section 5.3)

**Variable bitrate:** During training, $N_q$ is sampled randomly as a multiple of 4. Each codebook contains **1,024 entries** (10 bits). This yields:

<table>
  <thead><tr><th>Sample Rate</th><th>Supported Bitrates (kbps)</th></tr></thead>
  <tbody>
    <tr><td>24 kHz</td><td>1.5, 3, 6, 12, 24</td></tr>
    <tr><td>48 kHz</td><td>3, 6, 12, 24</td></tr>
  </tbody>
</table>

A maximum of 32 codebooks is used (16 for 48 kHz).

---

## 4. Small Transformer Language Model

EnCodec also trains a compact Transformer LM on the discrete codes to enable entropy coding and further compression.

**Architecture:**
- 5 transformer layers, 8 attention heads
- 200 hidden channels, FFN dimension 800
- No dropout
- Per-attention-layer causal receptive field: **3.5 s**
- Trained on **5-second sequences** with random sinusoidal position offset

**At inference:**

- At each timestep $t$: sum the learned embeddings of the $N_q$ codebook tokens from $t-1$
- At $t=0$: use a special start token
- $N_q$ separate linear output heads, each predicting the distribution over one codebook

> **Note:** The arithmetic coder uses interval-based probability estimation. Due to floating-point approximations, probability estimates are rounded to $10^{-6}$ precision, with total interval width $2^{24}$ and minimum per-symbol interval width 2.

---

## 5. Training Objectives

### 5.1 Reconstruction Loss

Two terms are combined.

**Time-domain L1:**

$$\ell_t(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_1$$

**Frequency-domain multi-scale mel spectrogram loss** ($L_1 + L_2$ combination over multiple resolutions):

$$\ell_f(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{|\alpha| \cdot |s|} \sum_{\alpha_i \in \alpha} \sum_{e \in s} \left( \|S_i(\mathbf{x}) - S_i(\hat{\mathbf{x}})\|_1 + \alpha_i \|S_i(\mathbf{x}) - S_i(\hat{\mathbf{x}})\|_2 \right)$$

where $S_i$ is a 64-channel mel spectrogram (normalized STFT) with window $2^e$ and hop $2^e/4$, scale set $s = \{5, \ldots, 11\}$, and $\alpha = 1$.

### 5.2 MS-STFT Discriminator

EnCodec uses a Multi-Scale STFT (MS-STFT) discriminator for perceptual loss. It consists of multiple sub-networks operating at different STFT resolutions.

```
                ┌──────────────────────────────────────────────────────────┐
 Audio ─────▶  │               MS-STFT Discriminator                      │
                │  ┌─────────────────┐  ┌─────────────────┐  (×5 scales) │
                │  │ STFT(w=2048)    │  │ STFT(w=1024)    │  ...         │
                │  │  [Re, Im]       │  │  [Re, Im]       │              │
                │  │  Conv2d(3×8,32) │  │  Conv2d(3×8,32) │              │
                │  │  dilation 1,2,4 │  │  dilation 1,2,4 │              │
                │  │  Conv2d(3×3, 1) │  │  Conv2d(3×3, 1) │              │
                │  └────────┬────────┘  └────────┬────────┘              │
                │           └───────────┬─────────┘                       │
                │                  logits (per scale)                     │
                └──────────────────────────────────────────────────────────┘
```

STFT windows: $[2048, 1024, 512, 256, 128]$ for 24 kHz (doubled for 48 kHz). Each sub-network concatenates the real and imaginary parts of the STFT.

The discriminator is updated with $\frac{2}{3}$ probability at 24 kHz and $0.5$ at 48 kHz. Stereo audio is processed with left and right channels separately. All networks use LeakyReLU and weight normalization.

**Generator adversarial loss:**

$$\ell_g(\hat{\mathbf{x}}) = \frac{1}{K} \sum_k \max(0, 1 - D_k(\hat{\mathbf{x}}))$$

**Feature matching loss** (relative L1 on intermediate discriminator features):

$$\ell_\text{feat}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{KL} \sum_{k=1}^{K} \sum_{l=1}^{L} \frac{\|D_k^l(\mathbf{x}) - D_k^l(\hat{\mathbf{x}})\|_1}{\text{mean}(\|D_k^l(\mathbf{x})\|_1)}$$

**Discriminator hinge loss:**

$$\mathcal{L}_d(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{K} \sum_k \left[\max(0, 1 - D_k(\mathbf{x})) + \max(0, 1 + D_k(\hat{\mathbf{x}}))\right]$$

### 5.3 VQ Commitment Loss

For each residual step $c \in \{1, \ldots, C\}$, where $\mathbf{z}_c$ is the current residual and $q_c(\mathbf{z}_c)$ is its nearest codebook entry:

$$\ell_w = \sum_{c=1}^{C} \|\mathbf{z}_c - q_c(\mathbf{z}_c)\|_2^2$$

### 5.4 Total Generator Loss and the Loss Balancer

The total generator loss is:

$$\mathcal{L}_G = \lambda_t \cdot \ell_t + \lambda_f \cdot \ell_f + \lambda_g \cdot \ell_g + \lambda_\text{feat} \cdot \ell_\text{feat} + \lambda_w \cdot \ell_w$$

**The loss balancer** — to stabilize training against varying gradient magnitudes from the discriminator, EnCodec introduces a gradient normalization scheme. For each loss $\ell_i$, define the gradient with respect to $\hat{\mathbf{x}}$:

$$g_i = \frac{\partial \ell_i}{\partial \hat{\mathbf{x}}}$$

Track the EMA of its norm: $\langle \|g_i\|_2 \rangle_\beta$ over recent training batches. The rescaled gradient is:

$$\tilde{g}_i = R \cdot \frac{\lambda_i}{\sum_j \lambda_j} \cdot \frac{g_i}{\langle \|g_i\|_2 \rangle_\beta}$$

with $R = 1$, $\beta = 0.999$. The model back-propagates $\sum_i \tilde{g}_i$ instead of $\sum_i \lambda_i g_i$. This ensures each loss contributes a share of the total gradient proportional to its weight $\lambda_i$, regardless of its absolute gradient scale.

> **Note:** The commitment loss $\ell_w$ is excluded from the balancer because it does not directly depend on the model output $\hat{\mathbf{x}}$.

---

## 6. Streaming vs. Non-Streaming Mode

<table>
  <thead>
    <tr><th></th><th>Non-Streaming</th><th>Streaming</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Padding</strong></td>
      <td>$K - S$ padding at both ends of each conv layer</td>
      <td>All $K - S$ padding placed before the first timestep</td>
    </tr>
    <tr>
      <td><strong>Chunking</strong></td>
      <td>Split input into 1-second chunks, overlap 10 ms to avoid clicks</td>
      <td>Process sample-by-sample; buffer $K - s$ steps until next frame</td>
    </tr>
    <tr>
      <td><strong>Normalization</strong></td>
      <td>LayerNorm (includes time dimension for relative scale)</td>
      <td>WeightNorm (LayerNorm is incompatible with causal streaming)</td>
    </tr>
    <tr>
      <td><strong>First-output latency</strong></td>
      <td>~1 s (full chunk needed)</td>
      <td>~13 ms (320 samples in → 320 samples out)</td>
    </tr>
  </tbody>
</table>

In streaming mode, for a transposed conv with stride $s$: the model outputs $s$ time steps immediately and keeps the remaining $K - s$ steps in a buffer, completing the computation once the next frame arrives (or discarding at stream end).

---

## 7. Ablation Results

The table below reports the **Real-Time Factor** (RTF, higher = faster than real-time) and audio quality metrics **SI-SNR** and **ViSQOL** for the base model and architectural variants.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>RTF Enc ↑</th>
      <th>RTF Dec ↑</th>
      <th>SI-SNR ↑</th>
      <th>ViSQOL ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>EnCodec base</strong></td><td>9.8</td><td>10.4</td><td>6.67</td><td>4.35</td></tr>
    <tr><td>Channels = 16</td><td>26.0</td><td>25.7</td><td>6.40</td><td>4.32</td></tr>
    <tr><td>Channels = 64</td><td>1.3</td><td>3.1</td><td>6.70</td><td>4.38</td></tr>
    <tr><td>norm = None</td><td>10.1</td><td>10.4</td><td>6.45</td><td>4.29</td></tr>
    <tr><td>LSTM = 0</td><td>15.0</td><td>14.6</td><td>6.40</td><td>4.35</td></tr>
    <tr><td>Residual Layer = 3, LSTM = 0</td><td>6.0</td><td>7.3</td><td>6.32</td><td>4.35</td></tr>
  </tbody>
</table>

**Takeaways:**
- **Channels = 64** achieves the best quality (SI-SNR 6.70, ViSQOL 4.38) but is 7–8× slower than the base.
- **Channels = 16** is the fastest (RTF ~26) but sacrifices ~0.3 dB SI-SNR.
- **Removing the LSTM** speeds up both encoder and decoder (RTF 15/14.6) but degrades SI-SNR from 6.67 to 6.40.
- **No normalization** hurts ViSQOL (4.29 vs 4.35) despite similar SI-SNR.
- The base model strikes the best speed/quality balance: RTF ≈ 10 on a single CPU core.

---

## 8. Strengths and Limitations

<table>
  <thead>
    <tr><th>Strengths</th><th>Limitations</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>High-quality audio at low bitrates.</strong> EnCodec significantly outperforms traditional codecs like Opus and EVS, especially on music and non-speech audio.</td>
      <td><strong>Training cost.</strong> Despite being real-time at inference, training requires substantial compute (multi-GPU, discriminator, Transformer LM).</td>
    </tr>
    <tr>
      <td><strong>Real-time on a single CPU core.</strong> The base model runs at RTF ~10, making it practical for on-device and streaming applications.</td>
      <td><strong>High initial latency in non-streaming mode.</strong> At 48 kHz, the 1-second chunk size introduces ~1 s of latency, unsuitable for interactive use without the streaming variant.</td>
    </tr>
    <tr>
      <td><strong>Flexible, variable bitrate.</strong> A single model supports 1.5–24 kbps at 24 kHz by varying $N_q$ at inference time.</td>
      <td><strong>Model complexity.</strong> The full training pipeline involves the codec, multi-scale discriminator, and a separate Transformer LM — each requiring careful tuning.</td>
    </tr>
    <tr>
      <td><strong>Novel loss balancer.</strong> Gradient normalization across losses simplifies hyperparameter tuning and improves training stability across discriminator scales.</td>
      <td><strong>Limits of objective metrics.</strong> SI-SNR and ViSQOL do not fully capture perceptual quality. More subjective listening evaluations (e.g., MUSHRA) are needed to validate real-world performance.</td>
    </tr>
  </tbody>
</table>
