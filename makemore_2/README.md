# makemore 2 — Neural Bigram Language Model

Second module of Karpathy's *Zero to Hero* series. Same task as makemore 1 (predict the next character of a name), but the counting table is replaced by a one-layer neural network trained with gradient descent.

**Result:** the neural net converges to a similar loss as the counting model (~2.45 NLL). That's the point: two routes to the same answer, and the gradient-based one scales to bigger models.

---

## What I built

- **Dataset:** 32K names (`names.txt`) → 228K bigram training pairs (input char → next char)
- **Encoding:** one-hot vectors over 27 characters (26 letters + `.` start/end token)
- **Model:** a single 27×27 weight matrix `W`, no bias, no hidden layer
- **Forward pass:** `logits = xenc @ W` → `exp` → normalize rows → probabilities (softmax by hand)
- **Loss:** average negative log-likelihood of the correct next character
- **Training:** manual loop: forward, `loss.backward()`, `W.data -= lr * W.grad`, zero grads
- **Regularization:** L2 penalty on `W` (equivalent to label smoothing / fake counts in the counting model)
- **Sampling:** generate new names from the trained model

---

## Results

| Model | Loss (NLL) |
|-------|-----------|
| Uniform baseline (1/27) | 3.30 |
| Counting bigram (makemore 1) | ~2.45 |
| Neural bigram (this module) | ~2.14|

---

## Key insights

- **`xenc @ W` is a table lookup.** Multiplying a one-hot vector by `W` just selects one row, so the trained `W` *is* the log-count table from makemore 1.
- **Logits are log-counts.** `exp(logits)` gives counts, and normalizing gives probabilities. Softmax is the counting model rewritten as differentiable operations.
- **Regularization = smoothing.** Pushing `W` toward zero pushes probabilities toward uniform, the same effect as adding fake counts.
- **Why bother, if the loss is identical?** Counting doesn't scale past bigrams (the table explodes). Gradient descent works unchanged when `W` becomes an MLP or a transformer.
---

## Run it

Open `makemore_2.ipynb` and run all cells. Requires `torch` and `matplotlib`; `names.txt` must be in `../data/` (or update the path in the first cell).

**Next:** makemore 3, an MLP with character embeddings (Bengio et al. 2003).
