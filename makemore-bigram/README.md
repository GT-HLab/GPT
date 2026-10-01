# makemore — Part 1: Bigram Character-Level Language Model

My implementation of Part 1 of Andrej Karpathy's *Neural Networks: Zero to Hero* series ("The spelled-out intro to language modeling: building makemore").

## What it does
Trains a model on ~32K first names to generate new, name-like strings, one character at a time. Each next character is predicted only from the previous one (bigram).

## Approach
Two equivalent models, built and compared:

1. **Counting model**: builds a 27×27 table of character-pair frequencies, normalizes the rows into probabilities, and samples from them.
2. **Neural-net model**: one linear layer (27×27 weights) on one-hot inputs, then softmax, trained by gradient descent on the negative log-likelihood. It converges to roughly the same loss as the counting model, which shows that both approaches learn the same distribution.

## Key concepts covered
- Tokenization at character level, with a `.` start/end token
- Maximum likelihood estimation and average negative log-likelihood as the loss
- Model smoothing (count smoothing ↔ L2 regularization on the weights)
- One-hot encoding, logits, softmax
- PyTorch tensors, broadcasting, `torch.multinomial` sampling
- Forward pass, `loss.backward()`, manual gradient-descent updates

## Results
| Model | Avg. NLL |
|---|---|
| Counting (with smoothing) | ~2.45 |
| Neural net (single layer) | ~2.45 |

Requires `names.txt` (from [karpathy/makemore](https://github.com/karpathy/makemore)) in the same folder.

## Next
Part 2: MLP with character embeddings (Bengio et al. 2003).

## Credit
Based on Andrej Karpathy's lecture: https://www.youtube.com/watch?v=PaCmpygFfXo
