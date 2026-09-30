# From Bigrams to GPT — Building Language Models from Scratch

A public log of my work implementing neural language models from first principles, following Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) and extending it toward GPT training and evaluation.

**The rule:** every line here is typed by hand. No copy-paste, no Copilot, no autocomplete. AI is used as an examiner (probing my code, checking my reasoning), never as the author.

---

## Why this repo exists

- **Who:** ~10 years in quantitative finance (factor models, risk), CQF, former nuclear engineer. Strong on the math, deliberately closing the gap on implementation fluency.
- **Goal:** read, modify, and train modern transformer code with confidence — then apply it to model evaluation.
- **Angle:** *evals are backtests.* Quant finance learned the hard way about overfitting, multiple testing, and out-of-sample discipline. LLM benchmarks face the same failure modes. This repo builds the foundation to work on that problem properly.

---

## Credits

- Andrej Karpathy — [Zero to Hero](https://karpathy.ai/zero-to-hero.html), [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)
- Bengio et al. (2003), *A Neural Probabilistic Language Model*
- Vaswani et al. (2017), *Attention Is All You Need*
