---
layout: homepage
---

<style>
.blog-post h3 { margin-top: 2em; }
.blog-post h4 { margin-top: 1.5em; color: #555; }
.blog-post table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
.blog-post th, .blog-post td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
.blog-post th { background: #f5f5f5; font-weight: 600; }
.blog-post pre { background: #f6f8fa; padding: 14px; border-radius: 6px; font-size: 0.85em; overflow-x: auto; }
.blog-post code { font-family: monospace; }
.blog-post .note { background: #fff8e1; border-left: 4px solid #f9a825; padding: 10px 14px; margin: 1em 0; font-size: 0.92em; }
.blog-post .divider { border: none; border-top: 1px solid #eee; margin: 2em 0; }
.blog-back { margin-bottom: 1.5em; font-size: 0.9em; }
</style>

<div class="blog-post">

<p class="blog-back">← <a href="/blog/">Back to Blog</a> &nbsp;|&nbsp; <a href="/">Home</a></p>

<h2>Codec-based TTS Pipeline: RVQ, Semantic Tokens, and Acoustic Tokens</h2>
<p style="color:#888; font-size:0.9em;">Series: Autoregressive Models for Speech &nbsp;·&nbsp; June 2026</p>

<hr class="divider">

<h3>1. Global Pipeline: Waveform → Tokens → Waveform</h3>

<p>A neural codec-based TTS system runs in three stages:</p>

<ol>
  <li><strong>Encode</strong>: A neural audio codec (EnCodec, DAC, Mimi) compresses the waveform into discrete tokens via RVQ.</li>
  <li><strong>Generate</strong>: An autoregressive (AR) language model generates a token sequence conditioned on text or phonemes.</li>
  <li><strong>Decode</strong>: The codec decoder reconstructs audio from the generated tokens.</li>
</ol>

<pre><code>Text / Phonemes
      │
      ▼
 ┌─────────────┐        ┌─────────────────────────────────┐        ┌──────────────┐
 │  Text LM /  │        │       Neural Audio Codec         │        │              │
 │  G2P / TTS  │──────▶│  Encoder → RVQ → [code sequence] │──────▶│   Decoder    │──▶ Waveform
 │  AR Model   │  tokens│                                 │ tokens │              │
 └─────────────┘        └─────────────────────────────────┘        └──────────────┘</code></pre>

<p>The key insight: by mapping audio to discrete tokens with a finite vocabulary, we turn speech synthesis into a language modeling problem. The same next-token prediction machinery behind GPT can now generate speech.</p>

<hr class="divider">

<h3>2. RVQ: What It Is and Why We Need It</h3>

<h4>The core problem</h4>

<p>Audio is a continuous, high-dimensional signal. A 1-second clip at 24 kHz is 24,000 floating-point samples. You cannot feed this directly into an AR model for next-token prediction—you need a finite vocabulary of discrete symbols.</p>

<p><strong>Plain VQ (Vector Quantization)</strong> encodes each frame as a single index into a codebook of K vectors. With K = 1024, each frame becomes one integer. The problem: reconstruction quality is poor. Mapping one high-dimensional vector to a choice among 1,024 entries loses too much information.</p>

<h4>RVQ: cascade of residuals</h4>

<p><strong>Residual Vector Quantization (RVQ)</strong> stacks multiple codebooks in series, each quantizing the residual error left by the previous:</p>

<pre><code>x              → Codebook 1 → code_1,   residual_1 = x − decode(code_1)
residual_1     → Codebook 2 → code_2,   residual_2 = residual_1 − decode(code_2)
residual_2     → Codebook 3 → code_3,   residual_3 = ...
...
residual_{N-1} → Codebook N → code_N</code></pre>

<p>Result: <strong>1 frame of audio = [code_1, code_2, …, code_N]</strong> — a column of N tokens.</p>

<ul>
  <li><strong>Codebook 1</strong> captures coarse structure: pitch contour, phoneme identity, broad prosody.</li>
  <li><strong>Deeper codebooks</strong> progressively refine finer detail: timbre, speaker texture, subtle resonances.</li>
</ul>

<p>EnCodec (Meta, 2022) uses 8 codebooks of size 1,024 at 75 Hz, giving 600 tokens/second total. The reconstruction quality improves monotonically as you include more codebook layers.</p>

<hr class="divider">

<h3>3. AR Modeling with RVQ: The Delay Pattern</h3>

<p>With N codebooks per frame, how should an AR transformer generate them? There are two main approaches.</p>

<h4>Option A: Flat interleaving</h4>

<p>Flatten all codebook tokens into a single sequence:</p>
<pre><code>[f0_cb1, f0_cb2, …, f0_cb8, f1_cb1, f1_cb2, …, f1_cb8, f2_cb1, …]</code></pre>
<p>This works with a standard single-head transformer, but the sequence is 8× longer and generation is slow: to decode 1 second of audio you need 600 sequential steps.</p>

<h4>Option B: Codebook delay pattern (MusicGen)</h4>

<p>Offset each codebook by one additional time step relative to the previous:</p>

<pre><code>Timestep →   t0    t1    t2    t3    t4    t5
Codebook 1:  f0    f1    f2    f3    f4    f5
Codebook 2:  [pad] f0    f1    f2    f3    f4
Codebook 3:  [pad] [pad] f0    f1    f2    f3
Codebook 4:  [pad] [pad] [pad] f0    f1    f2</code></pre>

<p>At each step t, the transformer predicts all N codebook tokens simultaneously using N output heads—one per codebook. Head k reads the token at position t − k as its ground-truth target during training. This gives full parallelism across codebooks at each step, reducing effective sequence length from T×N to T+N−1 while keeping generation depth at T steps.</p>

<div class="note">
  <strong>Why this matters for streaming:</strong> The delay pattern means you can start emitting audio after just N−1 steps of buffering. For 8 codebooks that is 7 extra frames (~93 ms at 75 Hz)—a negligible first-token latency.
</div>

<hr class="divider">

<h3>4. Semantic Token vs. Acoustic Token</h3>

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
      <td>Neural codec RVQ quantization</td>
    </tr>
    <tr>
      <td><strong>Token rate</strong></td>
      <td>~50 Hz, 1 layer</td>
      <td>75 Hz × 8 layers = 600 tok/s</td>
    </tr>
    <tr>
      <td><strong>What it captures</strong></td>
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

<p><strong>One sentence:</strong> Semantic tokens answer <em>what was said</em>; acoustic tokens answer <em>how it was said</em>.</p>

<h4>Extraction</h4>

<p><strong>Semantic tokens</strong> come from the intermediate layers of a self-supervised model like HuBERT (layer 6 is common). These features are then k-means clustered into 200–500 discrete units. The result is a sequence that closely mirrors phoneme boundaries and lexical identity, but has discarded most speaker-specific and prosodic fine-grained information.</p>

<p><strong>Acoustic tokens</strong> come directly from RVQ quantization inside a neural codec. They are trained end-to-end to minimize reconstruction error, so they preserve all perceptual detail—but the resulting sequence is ~12× longer than a semantic token sequence.</p>

<h4>Why hierarchical systems use both</h4>

<p>The design principle behind AudioLM and VALL-E is to separate the two prediction problems:</p>

<ol>
  <li><strong>Stage 1 — "What to say":</strong> A lightweight AR model predicts semantic tokens from text or a transcription. Short sequences, easy to learn, stable training.</li>
  <li><strong>Stage 2 — "How to say it":</strong> A second AR model generates acoustic tokens conditioned on the semantic tokens. The semantic anchor prevents the acoustic model from drifting and makes speaker/style conditioning much more controllable.</li>
</ol>

<p>Without the semantic anchor, training an AR model directly on 600 tok/s acoustic sequences is prone to instability and exposure bias. The two-stage split makes each sub-problem tractable.</p>

<hr class="divider">

<h3>5. Key Numbers Reference</h3>

<table>
  <thead>
    <tr><th>Concept</th><th>Typical Value</th><th>Significance</th></tr>
  </thead>
  <tbody>
    <tr><td>Codec frame rate</td><td>75 Hz (EnCodec)</td><td>75 frames per second of audio</td></tr>
    <tr><td>RVQ codebook count</td><td>8 (EnCodec)</td><td>8 tokens per frame</td></tr>
    <tr><td>Acoustic token rate</td><td>75 × 8 = 600 tok/s</td><td>AR sequence density for full-quality audio</td></tr>
    <tr><td>Semantic token rate</td><td>~50 Hz, 1 layer</td><td>Sequence is ~12× shorter than acoustic</td></tr>
    <tr><td>Codebook size</td><td>1,024</td><td>Vocabulary size per RVQ layer</td></tr>
    <tr><td>EnCodec sample rate</td><td>24 kHz</td><td>Input/output audio resolution</td></tr>
  </tbody>
</table>

<hr class="divider">

<h3>6. Deep Dives</h3>

<h4>6.1 Codebook Collapse: Causes and Fixes</h4>

<p><strong>The problem.</strong> During RVQ training, a large fraction of codebook entries ("dead codes") are never selected. This happens because random initialization places many code vectors far from the data manifold—they are never the nearest neighbor, never receive gradient updates, and stay dead forever. You can diagnose it by tracking codebook <em>perplexity</em> (effective number of codes used); a value like 120 out of 1,024 signals severe collapse.</p>

<p><strong>Why it matters.</strong> Dead codes reduce the effective vocabulary, degrading reconstruction quality and making the compression inefficient.</p>

<p><strong>Fixes, roughly in order of effectiveness:</strong></p>

<ol>
  <li>
    <strong>EMA (Exponential Moving Average) updates.</strong> Instead of gradient descent on codebook vectors, update each entry as a running average of the encoder outputs assigned to it:
    <pre><code>e_i ← α · e_i + (1 − α) · mean({x : nearest(x) = e_i})</code></pre>
    This keeps codes anchored to the actual data distribution and is the most stable training recipe.
  </li>
  <li>
    <strong>Dead-code reinitialization.</strong> If a code hasn't been selected for K training steps, replace it by sampling a random encoder output from the current batch. EnCodec uses this. It continuously repopulates the codebook and prevents permanent collapse.
  </li>
  <li>
    <strong>Commitment loss weight tuning.</strong> The commitment loss <code>‖sg(z) − e‖²</code> forces the encoder to stay close to codebook entries. Too small → encoder drifts; too large → encoder underfits. Typical weight: 0.25.
  </li>
  <li>
    <strong>Entropy regularization.</strong> Add a loss term that maximizes the entropy of the code assignment distribution, encouraging uniform usage across all codebook entries.
  </li>
  <li>
    <strong>Codebook splitting / hierarchical initialization.</strong> Start with a smaller codebook and split underused codes once training stabilizes.
  </li>
</ol>

<p>In practice, <strong>EMA updates + dead-code reinit</strong> (the EnCodec recipe) is the combination that works reliably across different domains and codebook sizes.</p>

<hr>

<h4>6.2 Teacher Forcing and Exposure Bias in Codec AR</h4>

<p><strong>Teacher forcing</strong> is the standard AR training procedure: at every training step, the model receives the ground-truth token sequence as context, regardless of what it would have predicted. This makes gradient computation clean and training fast.</p>

<p><strong>Exposure bias</strong> is the mismatch that teacher forcing creates: at inference time, the model must condition on its <em>own previously generated tokens</em>, which may be wrong. Errors compound over time—a mistake at step t shifts the context distribution, increasing the probability of mistakes at t+1, t+2, etc.</p>

<p><strong>Why codec AR amplifies this problem:</strong></p>
<ul>
  <li>Sequences are extremely long (600 tok/s), giving more steps for errors to cascade.</li>
  <li>RVQ tokens at deeper layers are correlated with earlier layers in the same frame. An error in codebook 1 propagates structurally into codebooks 2–8.</li>
  <li>The acoustic token space has much weaker linguistic structure than text, so the model has fewer semantic constraints to "self-correct."</li>
</ul>

<p><strong>Mitigations:</strong></p>
<ol>
  <li>
    <strong>Scheduled sampling.</strong> Gradually replace ground-truth tokens with model-predicted tokens during training—starting near 100% teacher forcing and decaying to free-running generation by the end of training. This is a curriculum from "easy" (ground truth) to "hard" (self-conditioned).
  </li>
  <li>
    <strong>Classifier-Free Guidance (CFG).</strong> At inference, blend conditional and unconditional logits:
    <pre><code>logits = logits_uncond + α · (logits_cond − logits_uncond)</code></pre>
    Increasing α sharpens the conditional distribution and reduces random drift. Used in MusicGen and many codec AR systems.
  </li>
  <li>
    <strong>Semantic token conditioning.</strong> Providing a strong semantic anchor (HuBERT tokens or phonemes) gives the acoustic AR model a hard constraint at every step. Even if it drifts in acoustic space, it stays grounded in the correct phonemic content.
  </li>
  <li>
    <strong>Non-autoregressive decoding for acoustic tokens.</strong> Systems like SoundStorm and MELLE use masked prediction (BERT-style) rather than causal AR for the acoustic stage. This avoids sequential error accumulation at the cost of needing multiple refinement passes.
  </li>
</ol>

<hr>

<h4>6.3 Implementing Streaming Generation with the Delay Pattern</h4>

<p>Below is a minimal Python implementation of the codebook delay pattern for AR inference:</p>

<pre><code class="language-python">import torch

VOCAB_SIZE = 1024
PAD_TOKEN  = VOCAB_SIZE      # out-of-vocab index used as padding
EOS_TOKEN  = VOCAB_SIZE + 1

def build_delay_input(generated: list[torch.Tensor], t: int, n_codebooks: int) -> torch.Tensor:
    """
    At generation step t, build the input token for each codebook head.
    Codebook k reads the token generated at step t - k.
    Returns: (n_codebooks,) int tensor of input tokens.
    """
    x = []
    for k in range(n_codebooks):
        src_step = t - k
        if src_step < 0:
            x.append(PAD_TOKEN)           # not yet generated
        elif src_step < len(generated):
            x.append(generated[src_step][k].item())
        else:
            x.append(PAD_TOKEN)
    return torch.tensor(x, dtype=torch.long)  # (n_codebooks,)


def apply_delay_pattern_to_tokens(
    tokens: torch.Tensor,   # (batch, n_codebooks, time)
    n_codebooks: int,
) -> torch.Tensor:
    """
    Shift codebook k forward by k steps, padding with PAD_TOKEN.
    Returns: (batch, n_codebooks, time + n_codebooks - 1)
    Used to preprocess ground-truth tokens for training.
    """
    B, K, T = tokens.shape
    out = torch.full((B, K, T + K - 1), fill_value=PAD_TOKEN, dtype=tokens.dtype)
    for k in range(K):
        out[:, k, k : k + T] = tokens[:, k, :]
    return out


@torch.no_grad()
def generate(model, prompt_tokens, max_frames: int, n_codebooks: int = 8):
    """
    Autoregressive generation using delay pattern.
    model: takes (seq_len, n_codebooks) input, returns (n_codebooks, vocab_size) logits.
    Returns: (max_frames, n_codebooks) generated token tensor.
    """
    generated: list[torch.Tensor] = []   # each entry: (n_codebooks,)
    context: list[torch.Tensor]   = list(prompt_tokens)  # seed context

    for t in range(max_frames):
        x = build_delay_input(context + generated, len(context) + t, n_codebooks)
        # x shape: (n_codebooks,) — the input token for each head at this step

        input_seq = torch.stack(context + generated)   # (seq_len, n_codebooks)
        logits = model(input_seq)                       # (n_codebooks, vocab_size)

        # Greedy decoding (replace with sampling + top-k/top-p in practice)
        new_tokens = logits.argmax(dim=-1)              # (n_codebooks,)
        generated.append(new_tokens)

        if new_tokens[0].item() == EOS_TOKEN:
            break

    return torch.stack(generated)  # (frames, n_codebooks)
</code></pre>

<p><strong>Key points:</strong></p>
<ul>
  <li>At step t, codebook k uses the token from step <code>t - k</code> as input. All K heads run in one forward pass.</li>
  <li>The <em>effective latency</em> for the first output frame is K−1 steps (7 frames for 8 codebooks = ~93 ms at 75 Hz).</li>
  <li>During training, apply <code>apply_delay_pattern_to_tokens</code> to the ground-truth before computing the cross-entropy loss, masking PAD positions.</li>
  <li>In production, combine this with KV-caching: only the new token at position t needs a forward pass; the rest of the context is cached.</li>
</ul>

<hr>

<h4>6.4 EnCodec vs. DAC vs. Mimi — Which to Use?</h4>

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
      <td><strong>Acoustic token rate</strong></td>
      <td>600 tok/s</td>
      <td>1,032 tok/s</td>
      <td>100 tok/s</td>
    </tr>
    <tr>
      <td><strong>Key innovation</strong></td>
      <td>First practical neural codec for AR; causal mode for streaming</td>
      <td>Improved VQ training (better codebook utilization); higher fidelity at high sample rates</td>
      <td>12.5 Hz frame rate + first codebook distilled from WavLM to be semantic-like</td>
    </tr>
    <tr>
      <td><strong>Codebook collapse</strong></td>
      <td>Moderate; uses EMA + reinit</td>
      <td>Better; additional tricks (factorized VQ, L2-norm)</td>
      <td>Not publicly detailed</td>
    </tr>
    <tr>
      <td><strong>Best for</strong></td>
      <td>TTS research baseline; maximum ecosystem support</td>
      <td>High-fidelity speech and music generation</td>
      <td>Real-time dialogue systems; low-cost AR generation</td>
    </tr>
  </tbody>
</table>

<p><strong>Choosing a codec for your project:</strong></p>

<ul>
  <li>
    <strong>EnCodec</strong> — choose it when you need maximum compatibility. VALL-E, VoiceCraft, AudioLM-style systems, and most public TTS codebases are built on EnCodec. Pretrained checkpoints, training recipes, and evaluation tooling are all widely available. Start here unless you have a specific reason not to.
  </li>
  <li>
    <strong>DAC</strong> — choose it when reconstruction quality is the top priority, especially for music or high-fidelity studio speech. DAC's improved VQ training (factorized codebooks, L2-normalized codes) yields meaningfully better perceptual quality at the same bitrate. The higher token rate (1,032 tok/s) makes end-to-end AR generation expensive, so pair it with a non-autoregressive acoustic decoder.
  </li>
  <li>
    <strong>Mimi</strong> — choose it when you are building a real-time or streaming speech dialogue system. The 12.5 Hz frame rate means only 100 tokens/second — an 8× reduction over EnCodec — making AR generation fast enough for interactive latency. Mimi's first codebook is distilled to be semantically grounded (via WavLM supervision), collapsing the usual two-stage semantic→acoustic pipeline into a single codec. The trade-off is less community tooling and fewer off-the-shelf TTS models.
  </li>
</ul>

<div class="note">
  <strong>Practical rule of thumb:</strong> For a new TTS research project in 2025–2026, start with <strong>EnCodec</strong> to leverage existing baselines. If quality is a bottleneck, switch the codec to <strong>DAC</strong>. If you need real-time dialogue, evaluate <strong>Mimi</strong> early — its frame-rate reduction pays off significantly at inference.
</div>

</div>
