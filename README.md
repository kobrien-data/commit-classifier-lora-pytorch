# Commit Message Classifier — From Scratch vs. Hand-Written LoRA

A code-commit classifier built **twice**: once as a from-scratch PyTorch network with every layer and the training loop written by hand, and once by adapting a pretrained DistilBERT with a **hand-implemented LoRA layer** — then compared head-to-head on trainable parameters, memory, training time, and accuracy, and served behind a FastAPI endpoint.

The point of the project isn't the task (it's a modest 6-class text classifier). The point is to show two things end to end: that I can write PyTorch training mechanics without a high-level trainer hiding them, and that I understand LoRA at the matrix level rather than as a one-line library call.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![License](https://img.shields.io/badge/License-MIT-green)

---

## The task

Given a commit message, predict its [Conventional Commit](https://www.conventionalcommits.org/) type:

`feat` · `fix` · `docs` · `refactor` · `test` · `chore`

Short inputs, six classes, trains fast on a laptop or a cheap GPU — small enough to iterate quickly, real enough to be a meaningful comparison.

---

## Results

> **Note:** the table below is populated by `src/compare.py`. Replace the placeholder cells with your own measured values after running it — these are not pre-filled, by design.

| Model | Trainable params | Peak memory | Time / epoch | Val accuracy | Macro-F1 |
|---|---|---|---|---|---|
| From scratch (embedding + MLP) | _all_ — `__` | `__` | `__` | `__` | `__` |
| DistilBERT + hand-written LoRA | `~__` (≈1%) | `__` | `__` | `__` | `__` |

**What to expect:** the LoRA model trains a tiny fraction of the parameters (typically ~99% fewer than full fine-tuning) yet beats the from-scratch baseline on accuracy — because it stands on DistilBERT's pretrained language knowledge, while the from-scratch model learns everything from random initialization. That contrast *is* the result; the interpretation matters more than the absolute scores.

---

## How it works

### Phase A — from-scratch classifier

A deliberately simple architecture so the **training loop** is the star, not the model:

```
Embedding(vocab, dim) → mean-pool → Linear → ReLU → Dropout → Linear → logits
CrossEntropyLoss, Adam
```

Vocabulary is built from the training set; everything trainable; trained from random init. Mediocre accuracy here is expected and sets up the comparison.

### Phase B — hand-written LoRA on a frozen DistilBERT

LoRA freezes the large pretrained weight matrix `W` and learns a small low-rank correction `W + (α/r)·BA` instead. The adapter is the centrepiece of the repo and is implemented from scratch in [`src/lora.py`](src/lora.py):

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base                          # pretrained, frozen
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)
```

Two details worth knowing:

- **`B` is initialised to zero**, so at step 0 the correction is exactly zero and the model behaves identically to pretrained DistilBERT — training starts from known-good behaviour and only diverges as it learns. `tests/test_lora.py` asserts exactly this.
- **Only `A`, `B`, and a small classification head carry gradients**; DistilBERT itself stays frozen. That's why the trainable-parameter count collapses.

The adapters are injected into the attention query/value projections; both phases reuse the **same training loop** in [`src/train.py`](src/train.py).

> On the **Q** in QLoRA: the LoRA math is hand-written here; the 4-bit quantization (NF4) is the genuinely hard part and is left to `bitsandbytes` if you load the base model in 4-bit. That's the standard division of labour.

---

## Project structure

```
commit-classifier-lora/
├── data/
│   ├── build_dataset.py     # GitHub API → labelled commit CSV
│   └── commits.csv          # generated
├── src/
│   ├── dataset.py           # PyTorch Dataset + DataLoader
│   ├── tokenizer.py         # vocab for Phase A; HF tokenizer for Phase B
│   ├── model_scratch.py     # from-scratch nn.Module
│   ├── lora.py              # hand-written LoRALinear  ← the centrepiece
│   ├── model_lora.py        # DistilBERT + injected LoRA + head
│   ├── train.py             # one training loop, both models
│   ├── evaluate.py          # accuracy, macro-F1, confusion matrix
│   └── compare.py           # runs both, prints the results table
├── serve/
│   ├── app.py               # FastAPI: POST /classify
│   └── Dockerfile
├── tests/
│   └── test_lora.py         # LoRA output == base output at init (B=0)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/<your-username>/commit-classifier-lora.git
cd commit-classifier-lora
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

**1. Build the dataset** (pulls commit messages from a set of public repos via the GitHub API, keeps messages with a conventional-commit prefix, and uses the prefix as the label):

```bash
export GITHUB_TOKEN=<your_token>   # avoids low rate limits
python data/build_dataset.py --out data/commits.csv
```

**2. Train each model:**

```bash
python src/train.py --model scratch
python src/train.py --model lora
```

**3. Run the comparison** (trains both if needed and prints the results table):

```bash
python src/compare.py
```

**4. Serve the trained model:**

```bash
uvicorn serve.app:app --reload
# or
docker build -t commit-classifier serve/ && docker run -p 8000:8000 commit-classifier
```

---

## API

`POST /classify`

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"message": "fix race condition in connection pool"}'
```

```json
{
  "type": "fix",
  "probabilities": {
    "fix": 0.91, "feat": 0.04, "refactor": 0.03,
    "chore": 0.01, "test": 0.01, "docs": 0.00
  }
}
```

---

## What I implemented by hand

These are written from scratch rather than delegated to a high-level library, because they're the point of the project:

- The `LoRALinear` adapter and its injection into DistilBERT's attention layers
- The PyTorch `Dataset` / `DataLoader` wiring
- The from-scratch classifier (`nn.Module`)
- The training loop (`forward → loss → backward → optimizer.step → zero_grad`, epoch loop, no-grad eval pass)
- The evaluation metrics and the comparison harness

Pretrained-model loading, the HF tokenizer, and the FastAPI plumbing use libraries — the standard division of labour.

---

## Tech stack

PyTorch · Hugging Face Transformers (DistilBERT base + tokenizer) · FastAPI · Docker · pytest + flake8 (CI). MLflow optional for experiment tracking.

---

## What I'd do with more time

- Load the base model in 4-bit (`bitsandbytes`) to make it true QLoRA and measure the memory difference.
- Expand beyond six commit types and handle multi-label messages.
- Add an LLM-as-judge evaluation to catch borderline misclassifications the accuracy metric hides.
- Sweep the LoRA rank `r` and report the accuracy/parameter trade-off curve.

---

## License

MIT — see [LICENSE](LICENSE).