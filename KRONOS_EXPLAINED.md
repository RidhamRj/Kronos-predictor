# Kronos: A Foundation Model for the Language of Financial Markets
*A Comprehensive Technical Architecture & Operational Guide*

---

## 1. Executive Summary

**Kronos** (accepted at **AAAI 2026**, arXiv:2508.02739, developed by researchers at Tsinghua University) is an open-source foundation model family specifically engineered for financial candlestick (K-line) time-series data.

Unlike general-purpose Time Series Foundation Models (TSFMs) such as TimesFM, Chronos, and MOIRAI—which are trained predominantly on non-financial metrics (weather, energy, traffic) and treat financial data as generic numerical curves—Kronos treats financial market dynamics as a **formal, discrete "language"**.

### Why General TSFMs Fail on Financial Data
1. **Extremely Low Signal-to-Noise Ratio (SNR)**: High-frequency market moves are dominated by microstructure noise, bid-ask bounce, and transient order imbalances. Continuous regression models tend to overfit this noise.
2. **Heavy/Fat-Tailed Distributions & Extreme Volatility**: Financial returns exhibit non-Gaussian fat tails and abrupt regime shifts (e.g., flash crashes, macro announcements). Continuous MSE losses are easily distorted by outliers.
3. **Intricate Multi-Attribute Dependencies (OHLCVA)**: A candlestick is not a single scalar; it represents Open, High, Low, Close, Volume, and Amount. The internal geometry of the bar (wicks, body length, volume confirmation) conveys essential market sentiment.
4. **Data Starvation in Generic Models**: Generic foundation models dedicate less than 1% of their training corpora to finance. Kronos is trained on **over 12 billion K-line records** sourced from **45+ global exchanges across 7 sampling frequencies**.

---

## 2. Architectural Overview: The Two-Phase Framework

Kronos operates on a **Discretize-and-Autoregress** paradigm composed of two primary engines:

```
Raw K-line Sequences (OHLCVA)
           │
           ▼
┌────────────────────────────────────────┐
│  Phase 1: Specialized K-line Tokenizer │
│  - Z-Score Normalization & Clipping    │
│  - 5 Cyclical Temporal Embeddings      │
│  - Transformer Autoencoder + BSQ       │
│  - Hierarchical Subtoken Factorization │
└────────────────────────────────────────┘
           │
           ▼  Hierarchical Discrete Tokens: [Coarse (s1), Fine (s2)]
┌────────────────────────────────────────┐
│  Phase 2: Autoregressive Decoder LLM   │
│  - Causal Attention with RoPE          │
│  - RMSNorm & Pre-LN Architecture       │
│  - Sequential Chain-Rule Generation    │
│  - Test-Time Monte Carlo Sampling      │
└────────────────────────────────────────┘
           │
           ▼
Multi-Task Financial Forecasts & Synthetics
(Price Forecasting, Volatility, Synthetic Generation, Alpha Strategies)
```

---

## 3. Phase 1: Specialized K-line Tokenization

The first phase maps continuous, 6-dimensional financial data $\mathbf{x}_t \in \mathbb{R}^6$ (Open, High, Low, Close, Volume, Amount) into discrete symbolic tokens.

### 3.1 Input Preprocessing & Temporal Features
1. **Per-Dimension Normalization**: Each feature dimension is independently z-score normalized and clipped to the range $[-5, 5]$ to neutralize extreme data feed anomalies.
2. **5-Fold Cyclical Temporal Embeddings**: Financial markets exhibit distinct intraday and calendar seasonalities. Kronos extracts:
   - Minute of day
   - Hour of day
   - Day of week
   - Day of month
   - Month of year
   
   Each is mapped to a dense embedding vector, summed, and injected into the token representations.

### 3.2 Binary Spherical Quantization (BSQ)
Instead of Euclidean Vector Quantization (VQ-VAE), which suffers from codebook collapse and unbounded error when faced with market spikes, Kronos adopts **Binary Spherical Quantization (BSQ)**:
- Continuous latent vectors $\bm{\xi}_t$ output by the encoder $E_{\text{enc}}$ are projected onto a **unit hypersphere**:
  $$\tilde{\bm{\xi}}_t = \frac{\bm{\xi}_t}{\|\bm{\xi}_t\|_2}$$
- Binarization is applied using the sign function to generate a $k$-bit binary code:
  $$b_t = \text{sign}(\tilde{\bm{\xi}}_t) \in \{-1, +1\}^k$$

#### Why BSQ Outperforms Euclidean Quantization in Finance:
- **Strictly Bounded Distortion**: The maximum quantization error is strictly bounded on the sphere ($\le \pi / \sqrt{L}$), preventing anomalous price jumps or flash crashes from destabilizing the codebook.
- **Hyperspherical Geometry & Angular Sensitivity**: Abrupt changes in market microstructure (e.g., sudden volume surges or price rejections) manifest as directional shifts in the feature space. Spherical projection captures angular differences with high fidelity.
- **Zero Codebook Collapse**: BSQ achieves **97.66%** coarse codebook utilization and **85.25%** fine codebook utilization without dead clusters.

### 3.3 Hierarchical Subtoken Factorization ($n=2$)
A $k$-bit codebook produces an intractable vocabulary size of $2^k$ (e.g., $k=20 \implies 2^{20} \approx 1,048,576$ tokens), which would make Transformer embedding tables prohibitively massive.

Kronos factorizes the $k$-bit code into **$n=2$ subtokens** of length $k/2$:
$$b_t = \bigl[b_t^c, \; b_t^f\bigr]$$
- **$b_t^c$ (Coarse Subtoken)**: Captures macro price action, bar direction (bullish/bearish), and general magnitude.
- **$b_t^f$ (Fine Subtoken)**: Captures micro details, candle wicks, shadow lengths, and intra-bar volume subtleties.

> **Efficiency Win**: Factorizing into $n=2$ shrinks vocabulary-dependent parameters by **over 99.8%** (from ~1.7B to 3.4M params in Kronos-base), while avoiding the latency penalties of higher factorizations ($n \ge 3$).

### 3.4 Hierarchical Reconstruction Loss
To enforce that $b_t^c$ and $b_t^f$ truly represent coarse and fine information, the autoencoder is trained using a composite hierarchical objective:
$$\mathcal{L}_{\text{tokenizer}} = \mathcal{L}_{\text{coarse}} + \mathcal{L}_{\text{fine}} + \lambda \mathcal{L}_{\text{quant}}$$
- **$\mathcal{L}_{\text{coarse}} = \mathbb{E}\left[\|\mathbf{x} - E_{\text{dec}}(b^c)\|^2\right]$**: Decodes using *only* the coarse token, forcing $b^c$ to reconstruct the main structure.
- **$\mathcal{L}_{\text{fine}} = \mathbb{E}\left[\|\mathbf{x} - E_{\text{dec}}([b^c, b^f])\|^2\right]$**: Decodes using both tokens, compelling $b^f$ to learn the residual detail.
- **$\mathcal{L}_{\text{quant}}$**: BSQ commitment and entropy regularization penalty.

---

## 4. Phase 2: Hierarchical Autoregressive Modeling

Once the market time series is discretized into hierarchical token sequences $\mathbf{b}_{1:T} = (b_1, b_2, \dots, b_T)$, the forecasting task is framed as next-token prediction using a decoder-only Transformer.

### 4.1 Chain-Rule Probability Decomposition
Given historical tokens $\mathbf{b}_{<t}$, the joint probability of the next token $b_t = [b_t^c, b_t^f]$ is factored via the probability chain rule:
$$p(b_t \mid \mathbf{b}_{<t}) = \underbrace{p(b_t^c \mid \mathbf{b}_{<t})}_{\text{Step 1: Coarse Prediction}} \times \underbrace{p(b_t^f \mid \mathbf{b}_{<t}, \; b_t^c)}_{\text{Step 2: Fine Prediction}}$$

### 4.2 Forward Propagation & Generation Flow
1. **Fused Input Representation**: At time step $i$, embeddings of coarse subtoken $e_c(b_i^c)$ and fine subtoken $e_f(b_i^f)$ are concatenated and linearly projected:
   $$\mathbf{v}_i = \mathbf{W}_{\text{fuse}} \left[ e_c(b_i^c) \,;\, e_f(b_i^f) \right]$$
2. **Contextual History State**: The sequence $\{\mathbf{v}_1, \dots, \mathbf{v}_{t-1}\}$ passes through causal Transformer blocks equipped with **Rotary Position Embeddings (RoPE)** and **RMSNorm (Pre-LN)**, yielding hidden state $\mathbf{h}_t$.
3. **Coarse Logit Head**:
   $$p(b_t^c \mid \mathbf{b}_{<t}) = \text{Softmax}(\mathbf{W}_c \mathbf{h}_t)$$
4. **Cross-Attention Fine Head**:
   To predict the fine token, the sampled coarse prediction $\hat{b}_t^c$ acts as a **Query**, while the historical state $\mathbf{h}_t$ acts as **Key** and **Value** in a cross-attention layer:
   $$p(b_t^f \mid \mathbf{b}_{<t}, \hat{b}_t^c) = \text{Softmax}\left(\mathbf{W}_f \, \text{CrossAttn}(e_c(\hat{b}_t^c), \mathbf{h}_t)\right)$$
5. **Mitigating Exposure Bias**: During training, $\hat{b}_t^c$ is sampled from the model's own predicted distribution rather than using teacher-forcing ground truth, aligning training dynamics with multi-step autoregressive rollout inference.

---

## 5. Pre-training Data & Curated Curation Pipeline

### 5.1 Dataset Scale & Scope
- **Records**: 12+ billion individual K-lines
- **Exchanges**: 45+ global exchanges across 30+ countries
- **Asset Classes**: Equities, Cryptocurrencies, Foreign Exchange (Forex), Futures
- **Frequencies**: 1m, 5m, 15m, 30m, 1h, 4h, 1d
- **Rebalancing**: Oversampling weights applied to crypto, forex, and futures to counteract equity dominance.

### 5.2 Two-Stage Cleaning Pipeline
Financial data is notorious for bad ticks, illiquidity, and artificial gaps. Kronos employs:
1. **Price Boundary Splitting**: Missing price values (NaN/Inf) are treated as sequence boundaries; sequences are partitioned into strictly contiguous valid blocks without fake imputation.
2. **Volume Zero-Imputation with Dropout**: Missing volumes/amounts are filled with 0. Furthermore, volume/amount is randomly dropped to 0 for **5%** of training batches, forcing the model to infer dynamics purely from price action if needed.
3. **Structural Break Segmentation**: Splits the sequence when relative price jumps exceed thresholds ($|\text{open}_t / \text{close}_{t-1} - 1| > \theta$), isolating corporate actions, stock splits, or contract rollovers.
4. **Illiquidity & Price Stagnation Filtering**: Removes sequences with sustained zero volume or identical consecutive closing prices.

---

## 6. Model Zoo & Specifications

Kronos models are hosted publicly on Hugging Face under the `NeoQuasar` organization:

| Model Name | Tokenizer | Context Window | Parameters | Public Status |
| :--- | :--- | :---: | :---: | :---: |
| **Kronos-mini** | `Kronos-Tokenizer-2k` | 2048 | 4.1M | Open-source |
| **Kronos-small** | `Kronos-Tokenizer-base` | 512 | 24.7M | Open-source |
| **Kronos-base** | `Kronos-Tokenizer-base` | 512 | 102.3M | Open-source |
| **Kronos-large** | `Kronos-Tokenizer-base` | 512 | 499.2M | AAAI Research Benchmark |

---

## 7. Inference & Test-Time Scaling

Kronos supports probabilistic autoregressive rollout at inference time:

### 7.1 Sampling Controls
- **Temperature ($T$)**:
  - **$T \approx 0.6$ (Low)**: Best for point price and return forecasting. Sharpens next-token probabilities to maximize directional accuracy (IC/RankIC).
  - **$T \approx 1.0$ (High)**: Best for realized volatility and synthetic K-line generation, where preserving distributional dispersion is vital.
- **Top-$p$ (Nucleus Sampling)**: Restricts token candidates to cumulative probability $p$.

### 7.2 Monte Carlo Ensembling (Test-Time Scaling)
Because Kronos generates candidate futures probabilistically, users can generate $N$ independent trajectories (e.g., $N=10$ or $N=50$) and average their decoded continuous values. As demonstrated empirically:
- Rollout ensembling significantly suppresses generation variance.
- Information Coefficient (IC) and RankIC monotonically increase with sample count $N$.

---

## 8. Benchmark Highlights

When evaluated across global markets against 25 leading baselines (including Chronos, TimesFM, PatchTST, iTransformer, TimeMOE, and GARCH):

| Downstream Task | Metric | Kronos vs. Baselines |
| :--- | :--- | :--- |
| **Price Forecasting** | RankIC | **+93%** over leading TSFMs; **+87%** over best non-pre-trained deep model |
| **Volatility Forecasting** | MAE | **9% lower error** than specialized econometric models (GARCH/ARCH) |
| **Synthetic K-Line Fidelity** | Discriminative Score | Outperforms DiffusionTS and TimeGAN |
| **Backtesting (CSI 300 / CSI 800)** | AER & IR | Highest Annualized Excess Return & Information Ratio |

---

## 9. Practical Python Quickstart

```python
import torch
import pandas as pd
# pip install huggingface_hub torch
from model import Kronos, KronosTokenizer, KronosPredictor

# 1. Load Tokenizer and Pretrained Model
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

# 2. Instantiate Predictor
predictor = KronosPredictor(
    model=model, 
    tokenizer=tokenizer, 
    max_context=512, 
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 3. Prepare Input OHLCVA DataFrame (Lookback <= 512 bars)
# df requires columns: ['open', 'high', 'low', 'close', 'volume', 'amount']
# and datetime index or timestamps

# 4. Generate Probabilistic Forecast
forecast_horizon = 24  # e.g., next 24 bars
predictions = predictor.predict(
    df=df,
    horizon=forecast_horizon,
    temperature=0.6,
    top_p=0.9,
    samples=10  # Monte Carlo ensemble rollouts
)

print("Forecasted OHLCVA sequence:")
print(predictions)
```

---

## 10. Disambiguation: Which "Kronos"?

In technical contexts, the name "Kronos" can refer to different systems:
1. **Kronos (This Model)**: The AAAI 2026 financial candlestick foundation model by Tsinghua researchers (`shiyu-coder/Kronos`).
2. **Amazon Chronos**: General-purpose time series LLM based on T5 architecture (`amazon-science/chronos-forecasting`).
3. **UKG Kronos (Workforce Central / Dimensions)**: Commercial enterprise workforce, employee scheduling, and payroll software suite.
4. **Khronos Group**: Industry consortium managing standards like Vulkan, OpenGL, and glTF.
