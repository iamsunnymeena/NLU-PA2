#!/usr/bin/env python3
"""Problem 1 pipeline: corpus prep, scratch Word2Vec, library comparison, plots, and form-ready outputs."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud

try:
    from gensim.models import Word2Vec
except Exception as exc:  # pragma: no cover
    raise RuntimeError("gensim is required for comparison. Install dependencies first.") from exc

SEED = 17
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "dataset" / "raw"
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
REPORTS = OUT / "reports"
MODELS = OUT / "models"

for p in (OUT, PLOTS, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)


def _bit_even(n: int) -> bool:
    """Returns True if n is even using bit logic instead of modulo."""
    return (n & 1) ^ 1 == 1


def clean_one_document(text: str) -> str:
    noise_snippets = [
        "a+ a a-",
        "sitemap",
        "view all",
        "previous",
        "next",
        "pause",
        "hindi",
        "advt.",
        "kb",
    ]

    t = text.replace("\u00a0", " ")
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\b\d+(?:\.\d+)?\b", " ", t)
    t = re.sub(r"[_*•#=+~`|]", " ", t)

    lines = []
    for line in t.splitlines():
        lx = line.strip()
        if not lx:
            continue
        low = lx.lower()
        if any(sn in low for sn in noise_snippets):
            continue
        if len(low) < 2:
            continue
        lines.append(lx)

    t = "\n".join(lines)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z\s\-']", " ", t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return [t for t in toks if len(t) > 1]


def build_corpus(raw_dir: Path) -> Tuple[List[List[str]], List[str], Counter]:
    documents: List[List[str]] = []
    all_tokens: List[str] = []

    for fp in sorted(raw_dir.glob("*.txt")):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_one_document(raw)
        toks = tokenize(cleaned)
        if toks:
            documents.append(toks)
            all_tokens.extend(toks)

    freq = Counter(all_tokens)
    return documents, all_tokens, freq


@dataclass
class W2VConfig:
    mode: str  # 'cbow' or 'sgns'
    dim: int
    window: int
    negative: int
    lr: float = 0.03
    epochs: int = 2
    min_count: int = 2
    max_vocab: int = 6000


class ScratchWord2Vec:
    def __init__(self, cfg: W2VConfig):
        self.cfg = cfg
        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []
        self.w_in: np.ndarray | None = None
        self.w_out: np.ndarray | None = None
        self.neg_table: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _build_vocab(self, docs: Sequence[Sequence[str]]) -> List[List[int]]:
        freq = Counter(t for d in docs for t in d)
        vocab = [w for w, c in freq.items() if c >= self.cfg.min_count]
        vocab.sort(key=lambda w: (-freq[w], w))
        vocab = vocab[: self.cfg.max_vocab]

        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = vocab

        kept_docs: List[List[int]] = []
        for doc in docs:
            ids = [self.word2id[w] for w in doc if w in self.word2id]
            if len(ids) > 3:
                kept_docs.append(ids)

        v = len(self.id2word)
        self.w_in = (np.random.randn(v, self.cfg.dim) / self.cfg.dim).astype(np.float32)
        self.w_out = np.zeros((v, self.cfg.dim), dtype=np.float32)

        # Negative sampling table using f(w)^0.75.
        power = np.array([freq[w] ** 0.75 for w in self.id2word], dtype=np.float64)
        p = power / power.sum()
        table_size = max(100_000, 10 * len(self.id2word))
        self.neg_table = np.random.choice(np.arange(len(self.id2word)), size=table_size, p=p)
        return kept_docs

    def _sample_negative(self, avoid: int, k: int) -> np.ndarray:
        assert self.neg_table is not None
        chosen = []
        while len(chosen) < k:
            cand = int(self.neg_table[np.random.randint(0, len(self.neg_table))])
            if cand != avoid:
                chosen.append(cand)
        return np.array(chosen, dtype=np.int32)

    def fit(self, docs: Sequence[Sequence[str]]) -> List[dict]:
        id_docs = self._build_vocab(docs)
        assert self.w_in is not None and self.w_out is not None

        logs: List[dict] = []
        for ep in range(1, self.cfg.epochs + 1):
            random.shuffle(id_docs)
            loss_sum = 0.0
            pair_count = 0

            for doc in id_docs:
                n = len(doc)
                for center_pos, center_id in enumerate(doc):
                    left = max(0, center_pos - self.cfg.window)
                    right = min(n, center_pos + self.cfg.window + 1)
                    ctx = [doc[j] for j in range(left, right) if j != center_pos]
                    if not ctx:
                        continue

                    if self.cfg.mode == "cbow":
                        h = self.w_in[ctx].mean(axis=0)
                        pos = center_id
                        negs = self._sample_negative(pos, self.cfg.negative)
                        cand = np.concatenate([[pos], negs])
                        labels = np.zeros(len(cand), dtype=np.float32)
                        labels[0] = 1.0

                        grad_h = np.zeros(self.cfg.dim, dtype=np.float32)
                        for wid, y in zip(cand, labels):
                            old_out = self.w_out[wid].copy()
                            score = float(self._sigmoid(np.dot(h, old_out)))
                            g = score - float(y)
                            self.w_out[wid] -= self.cfg.lr * g * h
                            grad_h += g * old_out
                            loss_sum += -(y * math.log(score + 1e-9) + (1 - y) * math.log(1 - score + 1e-9))
                            pair_count += 1

                        self.w_in[ctx] -= self.cfg.lr * grad_h / max(1, len(ctx))

                    else:  # SGNS
                        for pos in ctx:
                            negs = self._sample_negative(pos, self.cfg.negative)
                            cand = np.concatenate([[pos], negs])
                            labels = np.zeros(len(cand), dtype=np.float32)
                            labels[0] = 1.0

                            grad_v = np.zeros(self.cfg.dim, dtype=np.float32)
                            v = self.w_in[center_id].copy()
                            for wid, y in zip(cand, labels):
                                old_out = self.w_out[wid].copy()
                                score = float(self._sigmoid(np.dot(v, old_out)))
                                g = score - float(y)
                                self.w_out[wid] -= self.cfg.lr * g * v
                                grad_v += g * old_out
                                loss_sum += -(y * math.log(score + 1e-9) + (1 - y) * math.log(1 - score + 1e-9))
                                pair_count += 1

                            self.w_in[center_id] -= self.cfg.lr * grad_v

            logs.append(
                {
                    "epoch": ep,
                    "avg_loss": float(loss_sum / max(1, pair_count)),
                    "pairs": int(pair_count),
                }
            )
        return logs

    def vector(self, word: str) -> np.ndarray:
        idx = self.word2id[word]
        return self.w_in[idx]

    def nearest(self, word: str, topk: int = 5) -> List[Tuple[str, float]]:
        v = self.vector(word)
        mat = self.w_in
        assert mat is not None
        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(v)
        sims = (mat @ v) / (denom + 1e-9)
        idx = np.argsort(-sims)
        out = []
        for j in idx:
            w = self.id2word[j]
            if w == word:
                continue
            out.append((w, float(sims[j])))
            if len(out) == topk:
                break
        return out

    def analogy(self, a: str, b: str, c: str, topk: int = 5) -> List[Tuple[str, float]]:
        for w in (a, b, c):
            if w not in self.word2id:
                return []
        q = self.vector(b) - self.vector(a) + self.vector(c)
        mat = self.w_in
        assert mat is not None
        denom = np.linalg.norm(mat, axis=1) * np.linalg.norm(q)
        sims = (mat @ q) / (denom + 1e-9)
        idx = np.argsort(-sims)
        out = []
        banned = {a, b, c}
        for j in idx:
            w = self.id2word[j]
            if w in banned:
                continue
            out.append((w, float(sims[j])))
            if len(out) == topk:
                break
        return out


def save_wordcloud(freq: Counter, out_path: Path) -> None:
    wc = WordCloud(width=1600, height=900, background_color="white", colormap="cividis")
    wc.generate_from_frequencies(freq)
    plt.figure(figsize=(16, 9))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_scratch_experiments(docs: List[List[str]]) -> Tuple[pd.DataFrame, Dict[str, ScratchWord2Vec]]:
    configs = []
    models = {}
    rows = []

    for mode in ("cbow", "sgns"):
        for dim, window, negative in [(100, 3, 5), (300, 5, 10)]:
            cfg = W2VConfig(mode=mode, dim=dim, window=window, negative=negative, lr=0.03, epochs=2)
            tag = f"scratch_{mode}_d{dim}_w{window}_n{negative}"
            model = ScratchWord2Vec(cfg)
            logs = model.fit(docs)
            final_loss = logs[-1]["avg_loss"]
            rows.append(
                {
                    "model": tag,
                    "mode": mode,
                    "dim": dim,
                    "window": window,
                    "negative": negative,
                    "epochs": cfg.epochs,
                    "final_avg_loss": final_loss,
                    "vocab_size": len(model.id2word),
                }
            )
            models[tag] = model
            configs.append((tag, logs))

    for tag, logs in configs:
        (REPORTS / f"{tag}_train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows).sort_values(["mode", "dim"]).reset_index(drop=True)
    return df, models


def run_gensim_comparison(docs: List[List[str]]) -> pd.DataFrame:
    rows = []
    for sg, mode in [(0, "cbow"), (1, "sgns")]:
        for dim, window, negative in [(100, 3, 5), (300, 5, 10)]:
            model = Word2Vec(
                sentences=docs,
                vector_size=dim,
                window=window,
                min_count=2,
                workers=1,
                sg=sg,
                negative=negative,
                epochs=5,
                seed=SEED,
            )
            tag = f"gensim_{mode}_d{dim}_w{window}_n{negative}"
            model.save(str(MODELS / f"{tag}.model"))
            rows.append(
                {
                    "model": tag,
                    "mode": mode,
                    "dim": dim,
                    "window": window,
                    "negative": negative,
                    "epochs": 5,
                    "vocab_size": len(model.wv),
                }
            )
    return pd.DataFrame(rows).sort_values(["mode", "dim"]).reset_index(drop=True)


def plot_projection(words: List[str], vectors: np.ndarray, method: str, out_path: Path) -> None:
    if method == "pca":
        proj = PCA(n_components=2, random_state=SEED).fit_transform(vectors)
    else:
        perplexity = 8 if len(words) > 9 else max(2, len(words) - 1)
        proj = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, init="pca").fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(proj[:, 0], proj[:, 1], s=36)
    for i, w in enumerate(words):
        plt.text(proj[i, 0], proj[i, 1], w, fontsize=9)
    plt.title(f"{method.upper()} projection of selected embeddings")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    docs, all_tokens, freq = build_corpus(RAW_DIR)

    clean_corpus_text = "\n".join(" ".join(doc) for doc in docs)
    (ROOT / "corpus.txt").write_text(clean_corpus_text, encoding="utf-8")

    doc_count = len(docs)
    token_count = len(all_tokens)
    vocab_size = len(freq)

    stats = {
        "documents": doc_count,
        "tokens": token_count,
        "vocab_size": vocab_size,
        "raw_files_used": len(list(RAW_DIR.glob("*.txt"))),
        "corpus_file_mb": round((ROOT / "corpus.txt").stat().st_size / (1024 * 1024), 4),
        "even_doc_count_check": _bit_even(doc_count),
    }

    (REPORTS / "p1_dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    top10 = freq.most_common(10)
    (REPORTS / "p1_top10_words.json").write_text(json.dumps(top10, indent=2), encoding="utf-8")
    save_wordcloud(freq, PLOTS / "p1_wordcloud.png")

    scratch_df, scratch_models = run_scratch_experiments(docs)
    scratch_df.to_csv(REPORTS / "p1_scratch_experiments.csv", index=False)

    gensim_df = run_gensim_comparison(docs)
    gensim_df.to_csv(REPORTS / "p1_gensim_experiments.csv", index=False)

    best_scratch = scratch_models["scratch_sgns_d300_w5_n10"]

    targets = ["research", "student", "phd", "exam"]
    nearest = {}
    for w in targets:
        if w in best_scratch.word2id:
            nearest[w] = best_scratch.nearest(w, topk=5)
        else:
            nearest[w] = []

    analogies = {
        "ug:btech::pg:?": best_scratch.analogy("ug", "btech", "pg", topk=5),
        "research:lab::student:?": best_scratch.analogy("research", "lab", "student", topk=5),
        "course:credits::semester:?": best_scratch.analogy("course", "credits", "semester", topk=5),
    }

    embed_word = None
    for candidate in ["jodhpur", "institute", "academics", "program"]:
        if candidate in best_scratch.word2id and candidate != "jodhpur":
            embed_word = candidate
            break
    if embed_word is None:
        embed_word = best_scratch.id2word[3]

    vector_300 = best_scratch.vector(embed_word)
    vector_str = ", ".join(f"{x:.6f}" for x in vector_300.tolist())

    selected_words = [w for w, _ in freq.most_common(30) if w in best_scratch.word2id]
    selected_vectors = np.stack([best_scratch.vector(w) for w in selected_words])
    plot_projection(selected_words, selected_vectors, "pca", PLOTS / "p1_pca_scratch_sgns.png")
    plot_projection(selected_words, selected_vectors, "tsne", PLOTS / "p1_tsne_scratch_sgns.png")

    form_pack = {
        "corpus_size_mb": stats["corpus_file_mb"],
        "preprocessing_steps": [
            "Step-1: Merge raw IITJ text files and strip website/menu boilerplate lines.",
            "Step-2: Remove URLs, numbers, non-English characters, and formatting artifacts.",
            "Step-3: Lowercase and normalize whitespace/punctuation to clean plain text.",
            "Step-4: Tokenize with regex and remove very short/noisy tokens.",
            "Step-5: Build vocabulary/frequency table, generate word cloud and cleaned corpus file.",
        ],
        "embedding_word": embed_word,
        "embedding_300d_csv": vector_str,
        "top10_words": top10,
        "nearest_neighbors": nearest,
        "analogies": analogies,
    }

    (REPORTS / "p1_form_answers.json").write_text(json.dumps(form_pack, indent=2), encoding="utf-8")

    print("[P1] Done")
    print(json.dumps(stats, indent=2))
    print(f"[P1] Embedding word chosen: {embed_word}")


if __name__ == "__main__":
    main()
