---
layout: blog-post
title: "Codec-based TTS Pipeline: RVQ, Semantic Tokens, and Acoustic Tokens"
category: "Speech Synthesis"
date: 2026-06-01
description: "A detailed walkthrough of modern neural codec-based TTS — from RVQ mechanics and the codebook delay pattern to the semantic vs. acoustic token distinction, with deep dives on codebook collapse, exposure bias, streaming code, and EnCodec vs. DAC vs. Mimi."
read_time: "15 min read"
---

## 1. Global Pipeline: Waveform → Tokens → Waveform

A neural codec-based TTS system works in three stages:

1. **Encode** — A neural audio codec (EnCodec, DAC, Mimi) compresses the raw waveform into discrete tokens via RVQ.
2. **Generate** — An autoregressive (AR) language model produces a token sequence conditioned on text or phonemes.
3. **Decode** — The codec decoder reconstructs audio from the generated tokens.

```
Text / Phonemes
      │
      ▼
┌──────────────┐       ┌──────────────────────────────────┐       ┌──────────┐
│   AR Model   │──────▶│  Codec Encoder → RVQ → [tokens]  │──────▶│ Decoder  │──▶ Waveform
└──────────────┘ tokens└──────────────────────────────────┘ tokens└──────────┘
```

The key insight: by mapping audio to discrete tokens with a finite vocabulary, speech synthesis becomes a language modeling problem — the same next-token prediction machinery behind GPT can now generate speech.

---

## 2. RVQ: What It Is and Why We Need It

#### The core problem

Audio is a continuous, high-dimensional signal. One second at 24 kHz is 24,000 floating-point samples — impossible to feed directly into an AR model for next-token prediction. We need to convert it into a finite vocabulary of discrete symbols.

**Plain VQ (Vector Quantization)** encodes each frame as a single index into a codebook of K vectors. With K = 1024, each frame becomes one integer. The problem: reconstruction quality is poor. Compressing one high-dimensional vector into a choice among 1,024 entries loses too much information.

#### The RVQ solution: cascade of residuals

**Residual Vector Quantization (RVQ)** stacks multiple codebooks in series, each quantizing the residual error left by the previous:

```
x              → Codebook 1 → code_1,   residual_1 = x − decode(code_1)
residual_1     → Codebook 2 → code_2,   residual_2 = residual_1 − decode(code_2)
residual_2     → Codebook 3 → code_3,   ...
```

Result: **1 frame of audio = [code₁, code₂, …, codeₙ]** — a column of N tokens.

- **Codebook 1** captures coarse structure: pitch contour, phoneme identity, broad prosody.
- **Deeper codebooks** progressively refine finer detail: timbre, speaker texture, subtle resonances.

EnCodec uses 8 codebooks of size 1,024 at 75 Hz, giving 600 tokens/second total. Reconstruction quality improves monotonically as you include more codebook layers.

---

## 3. AR Modeling with RVQ: The Delay Pattern

With N codebooks per frame, how should an AR transformer generate them? Two main approaches:

#### Option A: Flat interleaving

Flatten all tokens into one sequence: `[f0_cb1, f0_cb2, …, f0_cb8, f1_cb1, …]`. Works with a standard single-head transformer, but the sequence is 8× longer and generation is slow — 600 sequential steps per second of audio.

#### Option B: Codebook delay pattern (MusicGen)

Offset each codebook by one additional time step relative to the previous:

```
Timestep →   t0     t1     t2     t3     t4     t5
Codebook 1:  f0     f1     f2     f3     f4     f5
Codebook 2:  [pad]  f0     f1     f2     f3     f4
Codebook 3:  [pad]  [pad]  f0     f1     f2     f3
Codebook 4:  [pad]  [pad]  [pad]  f0     f1     f2
```

At each step t, the transformer predicts all N codebook tokens in a single forward pass using N output heads — one per codebook. Head k targets the token at position t − k. This collapses the effective sequence length from T×N to T+N−1 while keeping generation depth at T steps.

> **Why this matters for streaming:** You can start emitting audio after just N−1 steps of buffering. For 8 codebooks that is 7 frames ≈ 93 ms at 75 Hz — a negligible first-token latency.

---

## 4. Semantic Token vs. Acoustic Token

**One sentence:** semantic tokens answer *what was said*; acoustic tokens answer *how it was said*.

<table>
  <thead>
    <tr>
      <th></th>
      <th>Semantic Token</th>
      <th>Acoustic Token</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Source</strong></td>
      <td>HuBERT / w2v-BERT middle layers + k-means</td>
      <td>Neural codec RVQ</td>
    </tr>
    <tr>
      <td><strong>Token rate</strong></td>
      <td>~50 Hz, 1 layer</td>
      <td>75 Hz × 8 layers = 600 tok/s</td>
    </tr>
    <tr>
      <td><strong>Captures</strong></td>
      <td>Phonemic content, word identity, broad prosody</td>
      <td>Timbre, speaker identity, fine rhythm, texture</td>
    </tr>
    <tr>
      <td><strong>Sequence length (1 s)</strong></td>
      <td>~50 tokens</td>
      <td>~600 tokens (~12× longer)</td>
    </tr>
    <tr>
      <td><strong>AR difficulty</strong></td>
      <td>Easy — short sequences, strong linguistic structure</td>
      <td>Hard — very long, fine-grained dependencies</td>
    </tr>
    <tr>
      <td><strong>Loses</strong></td>
      <td>Speaker identity, fine timbre</td>
      <td>Nothing (full signal reconstruction)</td>
    </tr>
  </tbody>
</table>

**Semantic tokens** come from the middle layers of a self-supervised model like HuBERT (layer 6 is standard), then k-means clustered into 200–500 discrete units. They mirror phoneme boundaries and lexical identity closely, but discard most speaker-specific and fine-grained prosodic information.

**Acoustic tokens** come directly from RVQ quantization in a neural codec. They preserve all perceptual detail — but the sequence is ~12× longer than a semantic token sequence.

#### Why hierarchical systems use both

The design principle behind AudioLM and VALL-E:

1. **Stage 1 — "What to say":** A lightweight AR model predicts semantic tokens from text. Short sequences, easy to learn, stable training.
2. **Stage 2 — "How to say it":** A second AR model generates acoustic tokens conditioned on the semantic tokens. The semantic anchor prevents drift and makes speaker/style conditioning more controllable.

Without the semantic anchor, training an AR model directly on 600 tok/s acoustic sequences is prone to instability and exposure bias. The two-stage split makes each sub-problem tractable.

---

## 5. Key Numbers

<table>
  <thead>
    <tr><th>Concept</th><th>Typical Value</th><th>Significance</th></tr>
  </thead>
  <tbody>
    <tr><td>Codec frame rate</td><td>75 Hz (EnCodec)</td><td>75 frames per second of audio</td></tr>
    <tr><td>RVQ codebooks</td><td>8 (EnCodec)</td><td>8 tokens per frame</td></tr>
    <tr><td>Acoustic token rate</td><td>75 × 8 = 600 tok/s</td><td>AR sequence density for full-quality audio</td></tr>
    <tr><td>Semantic token rate</td><td>~50 Hz, 1 layer</td><td>~12× shorter than acoustic</td></tr>
    <tr><td>Codebook size</td><td>1,024</td><td>Vocabulary size per RVQ layer</td></tr>
    <tr><td>EnCodec sample rate</td><td>24 kHz</td><td>Input/output audio resolution</td></tr>
  </tbody>
</table>

---

## 6. Deep Dives

### 6.1 Codebook Collapse: Causes and Fixes

**The problem.** During RVQ training, a large fraction of codebook entries ("dead codes") are never selected. Random initialization places many vectors far from the data manifold — they are never the nearest neighbor, never receive updates, and stay dead forever. Diagnose it by tracking codebook *perplexity* (effective number of active codes); a value like 120/1024 signals severe collapse.

**Fixes, in order of effectiveness:**

**1. EMA updates.** Instead of gradient descent on codebook vectors, update each entry as a running average of the encoder outputs assigned to it:

```
eᵢ ← α · eᵢ + (1 − α) · mean({x : nearest(x) = eᵢ})
```

This keeps codes anchored to the actual data distribution and is the most stable recipe.

**2. Dead-code reinitialization.** If a code hasn't been selected for K training steps, replace it by sampling a random encoder output from the current batch. EnCodec uses this. It continuously repopulates the codebook and prevents permanent collapse.

**3. Commitment loss weight tuning.** The commitment loss `‖sg(z) − e‖²` forces the encoder to stay close to codebook entries. Too small → encoder drifts; too large → encoder underfits. Typical weight: 0.25.

**4. Entropy regularization.** Add a loss term that maximizes the entropy of the code assignment distribution, encouraging uniform usage across all entries.

In practice, **EMA updates + dead-code reinit** (the EnCodec recipe) is the combination that works reliably across domains and codebook sizes.

---

### 6.2 Teacher Forcing and Exposure Bias in Codec AR

**Teacher forcing** is the standard AR training procedure: at every training step, the model receives the ground-truth token sequence as context, regardless of what it would have predicted. This makes gradient computation clean and training fast.

**Exposure bias** is the resulting mismatch: at inference, the model conditions on its *own* previously generated tokens — a distribution never seen during training. Errors compound over time.

In codec AR, this is amplified because:
- Sequences are extremely long (600 tok/s), giving many steps for errors to cascade.
- RVQ tokens at deeper layers are correlated with earlier layers in the same frame — an error in codebook 1 propagates structurally into codebooks 2–8.
- Acoustic token space has much weaker linguistic structure than text, so the model has fewer semantic constraints to self-correct.

**Mitigations:**

**1. Scheduled sampling.** Gradually replace ground-truth tokens with model-predicted tokens during training — a curriculum from 100% teacher forcing at the start down to free-running generation by the end.

**2. Classifier-Free Guidance (CFG).** At inference, blend conditional and unconditional logits:

```
logits = logits_uncond + α · (logits_cond − logits_uncond)
```

Increasing α sharpens the conditional distribution and reduces drift. Used in MusicGen and most modern codec AR systems.

**3. Semantic token conditioning.** Providing a strong semantic anchor (HuBERT tokens or phonemes) gives the acoustic AR model a hard constraint at every step — even if it drifts in acoustic space, it stays grounded in the correct phonemic content.

**4. Non-autoregressive decoding for the acoustic stage.** Systems like SoundStorm use masked prediction (BERT-style) rather than causal AR for acoustic tokens. This avoids sequential error accumulation at the cost of multiple refinement passes.

---

### 6.3 Streaming Generation: Delay Pattern Implementation

```python
import torch

VOCAB_SIZE = 1024
PAD_TOKEN  = VOCAB_SIZE      # padding / not-yet-generated
EOS_TOKEN  = VOCAB_SIZE + 1


def apply_delay_pattern(tokens: torch.Tensor, n_codebooks: int) -> torch.Tensor:
    """
    Shift codebook k forward by k steps, padding with PAD_TOKEN.
    Input:  (batch, n_codebooks, time)
    Output: (batch, n_codebooks, time + n_codebooks - 1)
    Use this to preprocess ground-truth tokens before computing training loss.
    """
    B, K, T = tokens.shape
    out = torch.full((B, K, T + K - 1), PAD_TOKEN, dtype=tokens.dtype)
    for k in range(K):
        out[:, k, k : k + T] = tokens[:, k, :]
    return out


def build_step_input(history: list, t: int, n_codebooks: int) -> torch.Tensor:
    """
    At generation step t, build the input token for each codebook head.
    Codebook k reads the token generated at step t - k.
    Returns: (n_codebooks,) int tensor.
    """
    tokens = []
    for k in range(n_codebooks):
        src = t - k
        tokens.append(history[src][k].item() if 0 <= src < len(history) else PAD_TOKEN)
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def generate(model, prompt: list, max_frames: int, n_codebooks: int = 8) -> torch.Tensor:
    """
    AR generation with delay pattern.
    model:   (seq_len, n_codebooks) → (n_codebooks, vocab_size)
    prompt:  list of (n_codebooks,) tensors (seed context)
    Returns: (max_frames, n_codebooks)
    """
    history = list(prompt)

    for t in range(max_frames):
        context = torch.stack(history)                  # (seq_len, n_codebooks)
        logits  = model(context)                        # (n_codebooks, vocab_size)
        new_tok = logits.argmax(dim=-1)                 # greedy; use sampling in practice
        history.append(new_tok)

        if new_tok[0].item() == EOS_TOKEN:
            break

    return torch.stack(history[len(prompt):])           # (frames, n_codebooks)
```

Key points:
- At step t, codebook k targets the token generated at step `t − k`. All K heads run in one forward pass.
- The first output frame is available after K−1 buffered steps (~93 ms at 75 Hz for 8 codebooks).
- In production, add KV-caching: only the new position needs a full forward pass; the rest is cached.
- Replace `argmax` with top-k / nucleus sampling and CFG for quality.

---

### 6.4 EnCodec vs. DAC vs. Mimi

<table>
  <thead>
    <tr>
      <th></th>
      <th>EnCodec</th>
      <th>DAC</th>
      <th>Mimi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Source</strong></td>
      <td>Meta FAIR, 2022</td>
      <td>Descript, 2023</td>
      <td>Kyutai (Moshi), 2024</td>
    </tr>
    <tr>
      <td><strong>Sample rate</strong></td>
      <td>24 kHz</td>
      <td>44.1 kHz</td>
      <td>24 kHz</td>
    </tr>
    <tr>
      <td><strong>Frame rate</strong></td>
      <td>75 Hz</td>
      <td>86 Hz</td>
      <td>12.5 Hz</td>
    </tr>
    <tr>
      <td><strong>RVQ layers</strong></td>
      <td>8</td>
      <td>12</td>
      <td>8</td>
    </tr>
    <tr>
      <td><strong>Token rate</strong></td>
      <td>600 tok/s</td>
      <td>1,032 tok/s</td>
      <td>100 tok/s</td>
    </tr>
    <tr>
      <td><strong>Key innovation</strong></td>
      <td>First practical neural codec for AR; causal mode for streaming</td>
      <td>Improved VQ training (factorized codebooks, L2-norm); better codebook utilization</td>
      <td>12.5 Hz frame rate; first codebook distilled from WavLM to be semantic-like</td>
    </tr>
    <tr>
      <td><strong>Best for</strong></td>
      <td>TTS research baseline; maximum ecosystem support</td>
      <td>High-fidelity speech and music</td>
      <td>Real-time dialogue; low-cost AR generation</td>
    </tr>
  </tbody>
</table>

**When to use each:**

- **EnCodec** — maximum compatibility. VALL-E, VoiceCraft, and most public TTS codebases are built on EnCodec. Start here unless you have a specific reason not to.

- **DAC** — when reconstruction quality is the top priority. DAC's improved VQ training yields meaningfully better perceptual quality at the same bitrate. The higher token rate (1,032 tok/s) makes end-to-end AR generation expensive, so pair it with a non-autoregressive acoustic decoder.

- **Mimi** — when building a real-time or streaming dialogue system. The 12.5 Hz frame rate means only 100 tokens/second — an 8× reduction over EnCodec — making AR generation fast enough for interactive latency. The first codebook is distilled to be semantically grounded (via WavLM supervision), collapsing the usual two-stage pipeline into a single codec.

> **Rule of thumb for new projects (2025–2026):** Start with **EnCodec** to leverage existing baselines. Switch to **DAC** if quality is the bottleneck. Evaluate **Mimi** early if you need real-time dialogue — its 8× frame-rate reduction pays off significantly at inference.
