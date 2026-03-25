#!/usr/bin/env python3
"""Problem 2 pipeline: generate names dataset, train 3 recurrent variants, evaluate and export metrics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SEED = 23
random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
REPORTS = OUT / "reports"
MODELS = OUT / "models"
for p in (OUT, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

NAMES_FILE = ROOT / "TrainingNames.txt"


def synthesize_indian_names(n: int = 1000) -> List[str]:
    starts = [
        "aa", "ad", "ak", "an", "ar", "bh", "ch", "da", "de", "dh", "ga", "ha", "ja", "ka",
        "kr", "ma", "na", "pa", "pr", "ra", "ri", "sa", "sh", "su", "ta", "va", "vi", "ya",
    ]
    mids = [
        "na", "ni", "nu", "ra", "ri", "ru", "la", "li", "ya", "vi", "shi", "ksha", "jit",
        "deep", "veer", "nath", "esh", "ansh", "ika", "isha", "ita", "ali", "endra", "yank",
    ]
    ends = [
        "a", "an", "ar", "ash", "esh", "ik", "in", "it", "ita", "iya", "raj", "ran", "sh",
        "shi", "ta", "thi", "ya", "veer", "vansh", "dev", "lal", "preet",
    ]

    base = {
        "aarav", "vihaan", "advik", "ananya", "siya", "ishita", "riya", "arjun", "lakshya", "anika",
        "kavya", "saanvi", "pranav", "harsh", "kriti", "mehul", "sarthak", "naman", "tanvi", "ruhan",
    }

    out = set(base)
    hop = 0
    while len(out) < n:
        # Intentional quirky synthesis path to avoid rigid templating.
        s = random.choice(starts)
        m = random.choice(mids)
        e = random.choice(ends)
        if (hop & 3) == 0:
            name = s + m + e
        elif (hop & 3) == 1:
            name = s + random.choice("aeiou") + m + e
        elif (hop & 3) == 2:
            name = s + m + random.choice("aeiou") + e
        else:
            name = s + m + m[-1] + e

        hop += 1
        name = name[:1].upper() + name[1:]
        if 4 <= len(name) <= 11:
            out.add(name)

    return sorted(out)[:n]


def ensure_training_names() -> List[str]:
    if NAMES_FILE.exists():
        names = [x.strip() for x in NAMES_FILE.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(names) >= 1000:
            return names[:1000]

    names = synthesize_indian_names(1000)
    NAMES_FILE.write_text("\n".join(names), encoding="utf-8")
    return names


class NameDataset(Dataset):
    def __init__(self, names: List[str], stoi: Dict[str, int]):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for nm in names:
            seq = "^" + nm.lower() + "$"
            ids = [stoi[ch] for ch in seq]
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate(batch):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xb = torch.full((len(xs), max_len), 0, dtype=torch.long)
    yb = torch.full((len(xs), max_len), 0, dtype=torch.long)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        n = x.size(0)
        xb[i, :n] = x
        yb[i, :n] = y
        mask[i, :n] = True
    return xb, yb, mask


class VanillaRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        z = self.emb(x)
        h, _ = self.rnn(z)
        return self.fc(h)


class BiLSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        z = self.emb(x)
        h, _ = self.lstm(z)
        return self.fc(h)


class AttentionRNNLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden, batch_first=True)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        z = self.emb(x)
        h, _ = self.rnn(z)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        score = torch.bmm(q, k.transpose(1, 2)) / (h.size(-1) ** 0.5)

        # Causal mask to only attend to current and previous timesteps.
        t = score.size(-1)
        mask = torch.triu(torch.ones(t, t, device=score.device), diagonal=1).bool()
        score = score.masked_fill(mask.unsqueeze(0), -1e9)
        attn = torch.softmax(score, dim=-1)
        ctx = torch.bmm(attn, v)
        return self.fc(ctx)


@dataclass
class TrainConfig:
    emb_dim: int = 48
    hidden: int = 128
    lr: float = 0.004
    epochs: int = 12
    batch_size: int = 64


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_total / (1024 * 1024)


def train_model(model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: torch.device) -> List[float]:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss(reduction="none")

    losses: List[float] = []
    for _ in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, mask in loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            logits = model(xb)
            loss_tok = crit(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss_tok = loss_tok.reshape(yb.shape)
            loss = (loss_tok * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item())
            count += 1
        losses.append(total / max(1, count))
    return losses


def generate_names(model: nn.Module, stoi: Dict[str, int], itos: Dict[int, str], n: int, device: torch.device) -> List[str]:
    model.eval()
    out = []
    inv_start = stoi["^"]
    inv_end = stoi["$"]

    with torch.no_grad():
        while len(out) < n:
            seq = [inv_start]
            for _ in range(18):
                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model(x)[0, -1]
                probs = torch.softmax(logits / 0.9, dim=-1)
                nid = int(torch.multinomial(probs, num_samples=1).item())
                if nid == 0:
                    continue
                if nid == inv_end:
                    break
                seq.append(nid)

            name = "".join(itos[i] for i in seq[1:] if i not in (0, inv_end))
            if 3 <= len(name) <= 12 and name.isalpha():
                out.append(name.capitalize())

    return out


def evaluate(generated: List[str], train_names: List[str]) -> Dict[str, float]:
    train_set = {x.lower() for x in train_names}
    gen_lower = [x.lower() for x in generated]
    unique = set(gen_lower)
    novelty = sum(1 for x in gen_lower if x not in train_set) / max(1, len(gen_lower))
    diversity = len(unique) / max(1, len(gen_lower))
    return {
        "novelty_rate": round(novelty * 100.0, 2),
        "diversity": round(diversity, 4),
        "generated_total": len(generated),
        "generated_unique": len(unique),
    }


def main() -> None:
    names = ensure_training_names()

    chars = sorted(set("".join(n.lower() for n in names)))
    vocab = ["<pad>", "^", "$"] + chars
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    ds = NameDataset(names, stoi)
    cfg = TrainConfig()
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = {
        "vanilla_rnn": VanillaRNNLM(len(vocab), cfg.emb_dim, cfg.hidden),
        "bilstm": BiLSTMLM(len(vocab), cfg.emb_dim, cfg.hidden),
        "attention_rnn": AttentionRNNLM(len(vocab), cfg.emb_dim, cfg.hidden),
    }

    results = {}
    sample_bank = {}

    for tag, model in specs.items():
        losses = train_model(model, loader, cfg, device)
        gen = generate_names(model, stoi, itos, n=500, device=device)
        ev = evaluate(gen, names)

        params = count_params(model)
        size_mb = model_size_mb(model)

        results[tag] = {
            "train_loss_curve": losses,
            "params": params,
            "model_size_mb": round(size_mb, 4),
            "hyperparameters": {
                "embedding_dim": cfg.emb_dim,
                "hidden_size": cfg.hidden,
                "epochs": cfg.epochs,
                "learning_rate": cfg.lr,
                "batch_size": cfg.batch_size,
            },
            **ev,
        }
        sample_bank[tag] = gen[:40]

        torch.save(model.state_dict(), MODELS / f"p2_{tag}.pt")

    (REPORTS / "p2_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (REPORTS / "p2_generated_samples.json").write_text(json.dumps(sample_bank, indent=2), encoding="utf-8")

    # Form-friendly winner logic: prioritize novelty and diversity.
    best = sorted(
        results.items(),
        key=lambda kv: (kv[1]["novelty_rate"], kv[1]["diversity"], -kv[1]["train_loss_curve"][-1]),
        reverse=True,
    )[0][0]

    form_pack = {
        "best_model": best,
        "why": "Selected by higher novelty and diversity with competitive final training loss.",
        "vanilla_rnn_params": results["vanilla_rnn"]["params"],
        "vanilla_rnn_model_size_mb": results["vanilla_rnn"]["model_size_mb"],
        "all_metrics": results,
    }
    (REPORTS / "p2_form_answers.json").write_text(json.dumps(form_pack, indent=2), encoding="utf-8")

    print("[P2] Done")
    print(json.dumps(form_pack, indent=2))


if __name__ == "__main__":
    main()
