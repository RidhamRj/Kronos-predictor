# Kronos Documentation

This workspace contains technical documentation and architecture breakdowns for **Kronos: A Foundation Model for the Language of Financial Markets** (AAAI 2026).

👉 **[Read the Full Explanation in KRONOS_EXPLAINED.md](./KRONOS_EXPLAINED.md)**

---

### Quick Summary
- **Domain**: Financial time series & candlestick (K-line) foundation model.
- **Key Idea**: Treats OHLCVA financial candlestick sequences as a discrete "market language".
- **Tokenization**: Uses Binary Spherical Quantization (BSQ) with $n=2$ hierarchical coarse/fine subtoken factorization.
- **Prediction**: Autoregressive decoder-only Transformer with causal RoPE self-attention.
- **Dataset**: Pre-trained on 12+ billion records from 45+ global exchanges across 7 frequencies.
- **Official GitHub**: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- **Hugging Face Models**: [NeoQuasar on Hugging Face](https://huggingface.co/NeoQuasar)
