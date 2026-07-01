# Commit Message Classifier — From Scratch vs. Hand-Written LoRA

A code-commit classifier built **twice**: once as a from-scratch PyTorch network with every layer and the training loop written by hand, and once by adapting a pretrained DistilBERT with a **hand-implemented LoRA layer** which is then compared head-to-head on trainable parameters, accuracy, and macro-F1.

The point of the project isn't the task (it's a modest 6-class text classifier). The point is to show two things end to end: that I can write PyTorch training mechanics without a high-level trainer hiding them, and that I understand LoRA at the matrix level rather than as a one-line library call.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![License](https://img.shields.io/badge/License-MIT-green)

---

## The task

Given a commit message, predict its [Conventional Commit](https://www.conventionalcommits.org/) type:

`feat` · `fix` · `docs` · `refactor` · `test` · `chore`

Short inputs, six classes, trains fast on a laptop or a cheap GPU — small enough to iterate quickly, real enough to be a meaningful comparison.

Data comes from the [`0x404/ccs_dataset`](https://huggingface.co/datasets/0x404/ccs_dataset) dataset on the Hugging Face Hub (commit messages labelled by Conventional Commit type, with a human-annotated split). It downloads automatically on first run — no scraping or API keys required.

---

## Results

Both models were trained and evaluated on the same six-class dataset (a fold of
the original ten Conventional Commit types. See note below) and scored on the
same held-out test set.

| Model | Trainable params | Test accuracy | Macro-F1 |
|---|---|---|---|
| From-scratch (embedding + mean-pool + MLP) | ~2.0M (all) | 0.5350 | 0.3735 |
| DistilBERT + hand-written LoRA | 155K / 66.5M (0.23%) | **0.6425** | **0.5829** |

The LoRA-adapted model outperforms the from-scratch baseline by ~11 points of
accuracy and ~21 points of macro-F1, while training only **0.23%** of its
parameters. The macro-F1 gap is the wider of the two because the from-scratch
model leans on the majority class and struggles with the finer distinctions,
whereas the pretrained model generalises more evenly across all six classes.

**On the six classes:** the dataset labels commits with ten Conventional Commit types. An initial ten-class run plateaued near 50% for *both* models. A confusion matrix showed the errors were concentrated in semantically overlapping types — `ci`, `build`, and `style` bled into `chore`, and `perf` into `refactor` — the same categories human annotators disagree on. These were folded into six classes reflecting distinctions the models (and people) can make reliably. See the write-up below.

---

## What I found
 
The interesting part of this project was the investigation, not just the final numbers:
 
- **Ten classes plateaued.** Both models stalled around 50% accuracy no matter how they were tuned.
- **A validation split made the problem visible.** Adding a held-out validation slice (carved from the training set, never the test set) and measuring accuracy each epoch showed the LoRA model's training loss falling to ~0.05 while validation accuracy flatlined. This showed clear overfitting. Early stopping (saving the best-validation epoch) was added in response.
- **A confusion matrix explained the ceiling.** The errors weren't random: they clustered among semantically similar commit types. `chore`, which is essentially a catch-all, had the worst F1 and scattered across every other class. This is inter-annotator ambiguity, an upper bound no model can beat.
- **Folding to six classes was an evidence-based decision.** Collapsing the overlapping types lifted both models and, more importantly, opened a clean gap between them where the ten-class versions had been nearly tied.
The takeaway: the from-scratch baseline learns the easy, distinctive classes but leans on the majority class (hence its low macro-F1); the pretrained LoRA model generalises across all six, which is exactly where pretrained language understanding earns its place.
 
---


## How it works

### Phase A — from-scratch classifier

A deliberately simple architecture so the **training loop** is the star, not the model:

```
Embedding(vocab, dim) → mean-pool → Linear → ReLU → Dropout → Linear → logits
CrossEntropyLoss, Adam
```

The token vocabulary comes from the DistilBERT tokenizer; the embeddings are trained from random init. Everything is trainable. A modest accuracy here is expected, it's the baseline the LoRA model is measured against.

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

- **`B` is initialised to zero**, so at step 0 the correction is exactly zero and the model behaves identically to pretrained DistilBERT. Training starts from known-good behaviour and only diverges as it learns.
- **Only `A`, `B`, and a small classification head carry gradients**; DistilBERT itself stays frozen. That's why the trainable-parameter count collapses to 0.23%.

The adapters are injected into the attention query/value projections; both phases reuse the **same training loop** in [`src/train.py`](src/train.py).

> On the **Q** in QLoRA: the LoRA math is hand-written here; the 4-bit quantization (NF4) is the genuinely hard part and is left to `bitsandbytes` if you load the base model in 4-bit. That's the standard division of labour.

---

## Project structure

```
commit-classifier-lora/

├── src/
│   ├── dataset.py           # PyTorch Dataset: tokenizes + maps labels (with six-class fold)
│   ├── model_scratch.py     # from-scratch nn.Module
│   ├── lora.py              # hand-written LoRALinear  ← the centrepiece
│   ├── model_lora.py        # DistilBERT + injected LoRA + classifiction head
│   ├── train.py             # training loop with validation + early stopping
│   ├── evaluate.py          # accuracy, macro-F1 on the test set
├── serve/
│   ├── app.py               # FastAPI: POST /classify
│   └── Dockerfile
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

**1. Load the dataset** 

The dataset downloads automatically from the Hugging Face Hub on first run.

**2. Train each model:**

```bash
python src/train.py
```

**3. Run the evaluation**:

```bash
python src/evaluate.py
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
- The PyTorch `Dataset` (tokenization, label mapping, and the six-class fold)
- The from-scratch classifier (`nn.Module`)
- The training loop (`forward -> loss -> zero_grad -> backward -> optimizer.step`, epoch loop, and a no-grad validation pass with early stopping)
- The evaluation metrics and confusion-matrix diagnostics
Pretrained-model loading and the tokenizer come from Hugging Face `transformers` — the standard division of labour.

---

## Tech stack

PyTorch · Hugging Face `transformers` (DistilBERT base + tokenizer) · `datasets` · scikit-learn (metrics).

---

## What I'd do with more time

- Build a FastAPI endpoint + Dockerfile to serve the trained model as a `POST /classify` API.
- Load the base model in 4-bit (`bitsandbytes`) to make it true QLoRA and measure the memory difference.
- Sweep the LoRA rank `r` and report the accuracy/parameter trade-off curve.
- Add a `test_lora.py` asserting the adapter's output equals the base layer's at initialisation (the zero-init `B` property).
- Expand back to the full ten classes with a larger training set to see whether more data closes the ambiguity gap.

---

## License

MIT — see [LICENSE](LICENSE).