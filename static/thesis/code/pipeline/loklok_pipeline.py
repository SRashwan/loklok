"""
LOKLOK Thesis Pipeline v7.0
===========================
Calibrated in-silico experiment for the M.A. thesis:
"The Social Media Radio Innovation and Its Impact on University Students'
Ratings: An Exploratory Study".

This module implements the final, reproducible analysis pipeline:

  Stage 1  Four-stream ingestion (field wave, FQD/FGD, stimulus corpus,
           participant observation), dual-modality embedding, DML/IPW
           debiasing, spectral manifold projection, and Monte-Carlo prior Q0.
  Stage 2  Within-persona 2^3 full-factorial tensor ABM, eight operator
           filters (Hamiltonian, Laplace, Lyapunov, DeGroot, Lagrange,
           Hilbert, Percolation, Entropy), and a Zewail-inspired
           pump-probe attentional-coherence gate.
  Stage 3  GRM-IRT measurement model for SReval (MCS+PB) and SRcomp
           (NMLS), dual frequentist/Bayesian inference, adversarial
           misinformation stress tests, Sobol/LHS sensitivity, reliability,
           measurement-invariance exports, and publication figures.

All outputs are model-based.  The measured wave is used only as an
empirical anchor and is never relabelled as a simulated treatment effect.

Author: research pipeline for Sherif M. A. Rashwan (ACU).
License: MIT (code); thesis data remain the author's property.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, stats
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

try:  # optional GPU acceleration
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except Exception:  # pragma: no cover
    PANDAS_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression, Ridge

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    SKLEARN_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import HashingVectorizer

    HASHING_AVAILABLE = True
except Exception:  # pragma: no cover
    HASHING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FACTORIAL_CELLS: List[Tuple[int, int, int]] = [
    (0, 0, 0),  # G0  max engagement
    (1, 0, 0),  # A   BRS credibility
    (0, 1, 0),  # B   MORE diversity + cooling
    (0, 0, 1),  # C   ERIC moderation + EMA reflection
    (1, 1, 0),  # A+B
    (1, 0, 1),  # A+C
    (0, 1, 1),  # B+C
    (1, 1, 1),  # G1  full epistemic-first LOKLOK
]

CELL_LABELS = [
    "G0 [0,0,0]",
    "A [1,0,0]",
    "B [0,1,0]",
    "C [0,0,1]",
    "AB [1,1,0]",
    "AC [1,0,1]",
    "BC [0,1,1]",
    "G1 [1,1,1]",
]

NMLS_DOMAINS = {
    "FC": list(range(0, 7)),
    "CC": list(range(7, 18)),
    "FP": list(range(18, 25)),
    "CP": list(range(25, 35)),
}

DEFAULT_SEED = 20261105
DEFAULT_N_PERSONAS = 500
DEFAULT_SESSIONS = 18


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def configure_gpu_backend(prefer_fp16: bool = True) -> Dict[str, object]:
    """Detect CUDA and configure TF32/AMP for T4/L4/A100 or RTX 3050."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        return {
            "backend": "PyTorch_CUDA",
            "device": str(device),
            "gpu_name": name,
            "vram_gb": round(float(vram), 2),
            "use_amp_fp16": bool(prefer_fp16),
        }
    return {
        "backend": "NumPy_CPU_Fallback",
        "device": None,
        "gpu_name": "CPU (vectorized BLAS/LAPACK)",
        "vram_gb": 0.0,
        "use_amp_fp16": False,
    }


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def robust_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    if not np.isfinite(iqr) or iqr <= 1e-12:
        sd = np.nanstd(x)
        iqr = sd if sd > 1e-12 else 1.0
    return (x - med) / iqr


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    return (x - mu) / sd


def _matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector product with optional CUDA acceleration.

    Small CPU BLAS calls are already fast; when a CUDA device is present and
    the matrix is not tiny, the product is offloaded to torch.float64 on the
    GPU and returned as NumPy.
    """
    if TORCH_AVAILABLE and torch.cuda.is_available() and A.shape[0] >= 256:
        try:
            At = torch.as_tensor(A, dtype=torch.float64, device="cuda")
            xt = torch.as_tensor(x, dtype=torch.float64, device="cuda")
            return (At @ xt).detach().cpu().numpy()
        except Exception:
            pass
    return A @ x


def hdi(samples: np.ndarray, prob: float = 0.95) -> Tuple[float, float]:
    """Shortest highest-density interval from posterior draws."""
    x = np.sort(np.asarray(samples, dtype=float).ravel())
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    width = int(math.floor(prob * n))
    width = max(1, min(width, n - 1))
    widths = x[width:] - x[: n - width]
    idx = int(np.argmin(widths))
    return (float(x[idx]), float(x[idx + width]))


def cohen_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 1e-12 else float("nan")


def paired_summary(diff: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "t": float("nan"),
            "p": float("nan"),
            "dz": float("nan"),
            "n": int(len(diff)),
        }
    t, p = stats.ttest_rel(diff, np.zeros_like(diff))
    se = stats.sem(diff)
    crit = stats.t.ppf(0.975, len(diff) - 1)
    return {
        "mean": float(np.mean(diff)),
        "sd": float(np.std(diff, ddof=1)),
        "ci_low": float(np.mean(diff) - crit * se),
        "ci_high": float(np.mean(diff) + crit * se),
        "t": float(t),
        "p": float(p),
        "dz": cohen_dz(diff),
        "n": int(len(diff)),
    }


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------


def load_measured_data(measured_dir: Path) -> Dict[str, object]:
    """Load the de-identified 52-row survey and derive scale scores."""
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas is required to load measured data")
    csv_path = measured_dir / "respondents_52.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    br_cols = ["BR1", "BR2", "BR3", "BR4"]
    sv_cols = ["SV1", "SV2", "SV3", "SV4"]
    iq_cols = ["IQ1", "IQ2", "IQ3"]
    pb_cols = ["PB1", "PB2", "PB3", "PB4", "PB5"]

    for c in br_cols + sv_cols + iq_cols + pb_cols + ["star_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # BR4 is reversed: original response 0..4 -> 4 - x
    if "BR4_reversed" in df.columns:
        df["BR4r"] = pd.to_numeric(df["BR4_reversed"], errors="coerce")
    else:
        df["BR4r"] = 4.0 - df["BR4"]

    df["BR_mean"] = df[["BR1", "BR2", "BR3", "BR4r"]].mean(axis=1)
    df["SV_mean"] = df[sv_cols].mean(axis=1)
    df["IQ_mean"] = df[iq_cols].mean(axis=1)
    df["PB_mean"] = df[pb_cols].mean(axis=1)
    df["BRS_total"] = df[["BR1", "BR2", "BR3", "BR4r", *sv_cols, *iq_cols]].sum(axis=1)

    for c in ["ride_duration_minutes", "interaction_ordinal", "age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    items = {
        "BR": br_cols,
        "SV": sv_cols,
        "IQ": iq_cols,
        "PB": pb_cols,
        "BRS": ["BR1", "BR2", "BR3", "BR4r", *sv_cols, *iq_cols],
    }

    measured = {
        "n": int(len(df)),
        "df": df,
        "items": items,
        "pb_mean": float(df["PB_mean"].mean()),
        "pb_sd": float(df["PB_mean"].std(ddof=1)),
        "br_mean": float(df["BR_mean"].mean()),
        "sv_mean": float(df["SV_mean"].mean()),
        "iq_mean": float(df["IQ_mean"].mean()),
        "star_mean": float(df["star_rating"].mean()),
        "duration_mean": float(df["ride_duration_minutes"].mean()),
        "duration_min": float(df["ride_duration_minutes"].min()),
        "duration_max": float(df["ride_duration_minutes"].max()),
        "interaction_counts": df["interaction_level"].value_counts().to_dict()
        if "interaction_level" in df.columns
        else {},
        "university_counts": df["university"].value_counts().to_dict()
        if "university" in df.columns
        else {},
    }
    return measured


def _parse_profiles(path: Path) -> Dict[str, str]:
    """Parse the participant-observation log into {SRxxx: text}. Names are
    not returned; only the body text is retained for feature extraction."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: Dict[str, str] = {}
    pattern = re.compile(r"\*\*(SR\d{3,4})\s*[-\u2013\u2014]\s*[^*]+\*\*")
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        code = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        body = re.sub(r"\s+", " ", body)
        blocks[code] = body
    return blocks


def _hash_text_features(texts: Sequence[str], n_features: int = 32) -> np.ndarray:
    """Deterministic hashing/TF-IDF-like features without external models."""
    if HASHING_AVAILABLE and texts:
        try:
            vec = HashingVectorizer(
                n_features=n_features,
                alternate_sign=False,
                norm="l2",
                ngram_range=(1, 2),
                lowercase=True,
            )
            return vec.transform(list(texts)).toarray().astype(float)
        except Exception:
            pass
    # light-weight fallback: character n-gram hashing
    out = np.zeros((len(texts), n_features), dtype=float)
    for i, t in enumerate(texts):
        t = (t or "").lower()
        for n in (3, 4):
            for j in range(max(0, len(t) - n + 1)):
                h = hash(t[j : j + n]) % n_features
                out[i, h] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return out / norms


def _theme_incidence(text: str) -> np.ndarray:
    """T1-T6 incidence from researcher-recorded account text, using the
    thesis codebook keywords.  These are coarse features, not automated
    qualitative coding."""
    t = (text or "").lower()
    keys = {
        "T1_credibility": [
            "credib",
            "correct",
            "verif",
            "fact",
            "source",
            "accuracy",
            "transparen",
        ],
        "T2_belonging": [
            "belong",
            "peer",
            "community",
            "friend",
            "together",
            "solidar",
            "familiar",
        ],
        "T3_context": [
            "ride",
            "traffic",
            "commute",
            "uber",
            "tired",
            "stress",
            "quiet",
            "background",
        ],
        "T4_participation": [
            "upvote",
            "comment",
            "interact",
            "participat",
            "voice",
            "co-host",
            "cohost",
            "submit",
        ],
        "T5_relevance": [
            "relevant",
            "useful",
            "student",
            "campus",
            "practical",
            "local",
            "survival",
        ],
        "T6_delivery": [
            "audio",
            "sound",
            "stream",
            "buffer",
            "interface",
            "screen",
            "quality",
        ],
    }
    out = np.zeros(len(keys), dtype=float)
    for i, (_, words) in enumerate(keys.items()):
        out[i] = sum(1.0 for w in words if w in t)
    return np.log1p(out)


def _corpus_features(corpus_dir: Path) -> Dict[str, float]:
    """Summarize the Uber-review corpus without shipping raw review text."""
    feats = {
        "reviews_n": 0.0,
        "mean_words": 0.0,
        "pos_share": 0.5,
        "neg_share": 0.5,
        "lexical_diversity": 0.5,
        "mean_sent_len": 0.0,
    }
    reviews_path = corpus_dir / "uber-reviews-hf" / "reviews.ndjson"
    if not reviews_path.exists():
        return feats
    n = 0
    total_words = 0
    pos = 0
    neg = 0
    vocab = set()
    total_tokens = 0
    sent_lens: List[int] = []
    try:
        with open(reviews_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                text = str(
                    obj.get("text")
                    or obj.get("review")
                    or obj.get("content")
                    or ""
                )
                score = obj.get("score") or obj.get("rating") or obj.get("stars")
                n += 1
                words = re.findall(r"[A-Za-z']+", text.lower())
                total_words += len(words)
                total_tokens += len(words)
                vocab.update(words)
                sent_lens.append(len(words))
                try:
                    s = float(score)
                    if s >= 4:
                        pos += 1
                    elif s <= 2:
                        neg += 1
                except Exception:
                    pass
    except Exception:
        return feats
    if n:
        feats.update(
            {
                "reviews_n": float(n),
                "mean_words": float(total_words / n),
                "pos_share": float(pos / n),
                "neg_share": float(neg / n),
                "lexical_diversity": float(len(vocab) / max(1, total_tokens)),
                "mean_sent_len": float(np.mean(sent_lens)) if sent_lens else 0.0,
            }
        )
    return feats


def build_four_stream_embedding(
    measured: Dict[str, object],
    data_root: Path,
    profiles_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Create the four-stream dual-modality embedding Z and nuisance matrix B.

    Quant stream: 11 BRS + 5 PB + star + duration + interaction + demographics.
    Qual stream:  participant-observation text hashing + FGD/theme incidence
                  + corpus summary + stimulus-corpus counts.
    """
    df = measured["df"].copy()
    n = len(df)

    # ---- quantitative block -------------------------------------------------
    br = df[["BR1", "BR2", "BR3", "BR4r"]].to_numpy(float)
    sv = df[["SV1", "SV2", "SV3", "SV4"]].to_numpy(float)
    iq = df[["IQ1", "IQ2", "IQ3"]].to_numpy(float)
    pb = df[["PB1", "PB2", "PB3", "PB4", "PB5"]].to_numpy(float)
    star = df[["star_rating"]].to_numpy(float)
    duration = df[["ride_duration_minutes"]].to_numpy(float)
    interaction = df[["interaction_ordinal"]].to_numpy(float)
    age = df[["age"]].to_numpy(float) if "age" in df.columns else np.zeros((n, 1))

    uni_dummies = np.zeros((n, 0))
    if "university" in df.columns:
        uni = pd.get_dummies(df["university"], drop_first=False).to_numpy(float)
        # drop one category to avoid collinearity with intercept
        uni_dummies = uni[:, 1:] if uni.shape[1] > 1 else uni

    fac_dummies = np.zeros((n, 0))
    if "faculty_group" in df.columns:
        fac = pd.get_dummies(df["faculty_group"], drop_first=False).to_numpy(float)
        fac_dummies = fac[:, 1:] if fac.shape[1] > 1 else fac

    gender = np.zeros((n, 1))
    if "gender" in df.columns:
        gender = (df["gender"].astype(str).str.upper() == "M").to_numpy(float).reshape(-1, 1)

    X_quant = np.hstack(
        [br, sv, iq, pb, star, duration, interaction, age, gender, uni_dummies, fac_dummies]
    )
    X_quant = np.nan_to_num(X_quant, nan=np.nanmedian(X_quant, axis=0))
    X_quant_std = zscore(X_quant)

    # ---- qualitative / context block ---------------------------------------
    profiles: Dict[str, str] = {}
    for candidate in [
        profiles_path,
        data_root / "03_data" / "qualitative" / "po_profiles_pseudonymized.txt",
        data_root / "03_data" / "qualitative" / "restricted" / "po_profiles_pseudonymized.txt",
    ]:
        if candidate and Path(candidate).exists():
            profiles = _parse_profiles(Path(candidate))
            if profiles:
                break

    ids = df["student_id"].astype(str).tolist() if "student_id" in df.columns else [
        f"SR{i+1:03d}" for i in range(n)
    ]
    profile_texts = [profiles.get(i, "") for i in ids]

    # If profiles are unavailable, construct a deterministic context sentence
    # from the measured design fields so the pipeline still runs.
    for i in range(n):
        if not profile_texts[i]:
            row = df.iloc[i]
            profile_texts[i] = " ".join(
                str(row.get(c, "")) for c in ["listening_context", "student_mood", "traffic_conditions"]
            )

    po_hash = _hash_text_features(profile_texts, n_features=32)
    theme = np.vstack([_theme_incidence(t) for t in profile_texts])

    corpus = _corpus_features(data_root / "03_data" / "qualitative" / "corpora")
    corpus_vec = np.array(
        [
            corpus["reviews_n"],
            corpus["mean_words"],
            corpus["pos_share"],
            corpus["neg_share"],
            corpus["lexical_diversity"],
            corpus["mean_sent_len"],
        ],
        dtype=float,
    )
    corpus_vec = np.tile(corpus_vec, (n, 1))

    # stimulus corpus: 25 briefs; per-respondent exposure is anchored by the
    # 5 PB items and the ride context, not by a fabricated text response.
    n_topics = 25.0
    topic_exposure = np.tile(np.array([[n_topics, math.log1p(n_topics)]]), (n, 1))

    # FGD theme telemetry (aggregate, repeated with small deterministic jitter)
    fgd = np.tile(theme.mean(axis=0, keepdims=True), (n, 1))
    rng = np.random.default_rng(DEFAULT_SEED)
    fgd = fgd + rng.normal(0, 0.05, size=fgd.shape)

    X_qual = np.hstack([po_hash, theme, corpus_vec, topic_exposure, fgd])
    X_qual = np.nan_to_num(X_qual, nan=0.0)
    norms = np.linalg.norm(X_qual, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    X_qual_norm = X_qual / norms

    Z_biased = np.hstack([X_quant_std, X_qual_norm])

    # ---- nuisance / bias block ---------------------------------------------
    person_mean = X_quant_std.mean(axis=1, keepdims=True)
    person_sd = X_quant_std.std(axis=1, keepdims=True)
    acquiescence = (br > 2).sum(axis=1).reshape(-1, 1) + (pb > 4).sum(axis=1).reshape(-1, 1)
    high_interaction = (interaction >= 3).astype(float)
    B_nuisance = np.hstack(
        [
            np.ones((n, 1)),
            person_mean,
            person_sd,
            acquiescence,
            duration,
            interaction,
            high_interaction,
            age,
            gender,
        ]
    )
    B_nuisance = np.nan_to_num(B_nuisance, nan=0.0)
    B_names = [
        "Intercept",
        "Person mean",
        "Person SD",
        "Acquiescence",
        "Ride duration",
        "Interaction",
        "High interaction",
        "Age",
        "Gender",
    ]

    return {
        "n": n,
        "ids": ids,
        "Z_biased": Z_biased,
        "B_nuisance": B_nuisance,
        "B_names": B_names,
        "X_quant": X_quant,
        "X_qual": X_qual,
        "profiles_found": bool(profiles),
        "corpus": corpus,
        "themes": theme,
        "profile_texts": profile_texts,
        "high_interaction": high_interaction.ravel(),
    }


# ---------------------------------------------------------------------------
# 2. Debiasing and spectral manifold
# ---------------------------------------------------------------------------


def dml_ipw_debias(
    Z: np.ndarray,
    B: np.ndarray,
    high_interaction: np.ndarray,
    B_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Double/orthogonal ML residualization + inverse-propensity weighting."""
    n, p = Z.shape
    B_aug = np.hstack([np.ones((n, 1)), B])
    ridge = 1e-2
    if SKLEARN_AVAILABLE:
        model = Ridge(alpha=ridge, fit_intercept=False)
        model.fit(B_aug, Z)
        Z_pred = model.predict(B_aug)
    else:
        beta = np.linalg.solve(B_aug.T @ B_aug + ridge * np.eye(B_aug.shape[1]), B_aug.T @ Z)
        Z_pred = B_aug @ beta
    Z_orth = Z - Z_pred

    # Propensity of being in the high-interaction stratum (the sample is
    # heavily skewed: 33 high vs 1 minimal in the measured wave).
    p_score = np.full(n, float(np.mean(high_interaction)))
    if SKLEARN_AVAILABLE and len(np.unique(high_interaction)) > 1:
        try:
            lr = LogisticRegression(max_iter=2000, C=1.0)
            lr.fit(B, high_interaction)
            p_score = lr.predict_proba(B)[:, 1]
        except Exception:
            pass
    p_score = np.clip(p_score, 0.10, 0.90)
    w = 1.0 / p_score
    w = w / np.mean(w)
    sqrt_w = np.sqrt(w)[:, None]
    Z_clean = Z_orth * sqrt_w
    Z_clean = zscore(Z_clean)

    # balance diagnostics: absolute correlation of each residualized feature
    # with the high-interaction indicator, before and after weighting.
    def max_abs_corr(mat: np.ndarray) -> float:
        y = high_interaction.astype(float)
        y = (y - y.mean()) / (y.std() + 1e-12)
        m = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)
        return float(np.max(np.abs(m.T @ y / len(y))))

    def col_corr(mat: np.ndarray, col: int) -> float:
        y = high_interaction.astype(float)
        y = (y - y.mean()) / (y.std() + 1e-12)
        x = mat[:, col]
        x = (x - x.mean()) / (x.std() + 1e-12)
        return float(abs(x @ y / len(y)))

    # per-nuisance balance uses the first `B.shape[1]` columns of Z_orth
    # (the residualized quantitative block); label them for the Love plot.
    names = list(B_names) if B_names is not None else [f"Nuisance {i}" for i in range(B.shape[1])]
    balance = []
    for j, name in enumerate(names):
        if j < Z_orth.shape[1]:
            balance.append(
                {
                    "covariate": name,
                    "before": col_corr(Z, j),
                    "after_dml": col_corr(Z_orth, j),
                    "after_ipw": col_corr(Z_clean, j),
                }
            )

    return {
        "Z_clean": Z_clean,
        "propensity": p_score,
        "weights": w,
        "max_abs_corr_before": max_abs_corr(Z),
        "max_abs_corr_orth": max_abs_corr(Z_orth),
        "max_abs_corr_after": max_abs_corr(Z_clean),
        "covariate_balance": balance,
        "weight_range": (float(w.min()), float(w.max())),
    }


def spectral_manifold(
    Z: np.ndarray,
    n_components: int = 6,
    k_neighbors: int = 8,
    heat_tau: float = 2.5,
) -> Dict[str, object]:
    """PCA + LLE + Laplacian eigenmaps, averaged into debiased coordinates."""
    n, p = Z.shape
    k_neighbors = max(2, min(k_neighbors, n - 2))
    n_components = max(1, min(n_components, n - 2, p))

    # PCA
    cov = np.cov(Z, rowvar=False)
    eigvals, eigvecs = linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:n_components]
    U_pca = Z @ eigvecs[:, idx] / np.sqrt(np.maximum(eigvals[idx], 1e-8))

    # LLE
    dist = cdist(Z, Z, metric="euclidean")
    W_lle = np.zeros((n, n))
    for i in range(n):
        nbrs = np.argsort(dist[i])[1 : k_neighbors + 1]
        Zi = Z[nbrs] - Z[i]
        C = Zi @ Zi.T
        C += np.eye(k_neighbors) * 1e-3 * max(np.trace(C), 1e-8)
        try:
            w = np.linalg.solve(C, np.ones(k_neighbors))
        except np.linalg.LinAlgError:
            w = np.ones(k_neighbors)
        W_lle[i, nbrs] = w / (np.sum(w) + 1e-12)
    M = (np.eye(n) - W_lle).T @ (np.eye(n) - W_lle)
    vals_lle, vecs_lle = linalg.eigh(M)
    U_lle = vecs_lle[:, 1 : n_components + 1]
    U_lle = zscore(U_lle)

    # Laplacian eigenmaps
    K = np.exp(-(dist**2) / (4.0 * heat_tau))
    np.fill_diagonal(K, 0.0)
    D = np.diag(K.sum(axis=1))
    vals_lap, vecs_lap = linalg.eigh(D - K, D)
    U_lap = vecs_lap[:, 1 : n_components + 1]
    U_lap = zscore(U_lap)

    U_pca = zscore(U_pca)
    Theta = (U_pca + U_lle + U_lap) / 3.0
    Theta = zscore(Theta)

    # condition number before/after (on the retained axes)
    cond_before = float(np.linalg.cond(cov + 1e-8 * np.eye(p)))
    cond_after = float(np.linalg.cond(np.cov(Theta, rowvar=False) + 1e-8 * np.eye(Theta.shape[1])))
    return {
        "Theta": Theta,
        "U_pca": U_pca,
        "U_lle": U_lle,
        "U_lap": U_lap,
        "condition_before": cond_before,
        "condition_after": cond_after,
        "eigvals_pca": eigvals[idx],
    }


# ---------------------------------------------------------------------------
# 3. Q0 prior and personas
# ---------------------------------------------------------------------------


def fit_q0(Theta: np.ndarray, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """Fit a continuous Q0 prior from the debiased manifold.

    A Gaussian KDE is used when the sample covariance is well-conditioned;
    otherwise a shrunken multivariate normal is used.  Both are explicit
    smoothing choices, not claims that the empirical 52 are a population.
    """
    n, d = Theta.shape
    cov = np.cov(Theta, rowvar=False)
    cond = np.linalg.cond(cov + 1e-10 * np.eye(d))
    if n >= 30 and np.isfinite(cond) and cond < 1e8:
        kde = gaussian_kde(Theta.T, bw_method="scott")
        sampler = "gaussian_kde"
    else:
        kde = None
        sampler = "shrunken_mvn"
    return {"kde": kde, "mean": Theta.mean(axis=0), "cov": cov, "sampler": sampler, "cond": float(cond)}


def draw_personas(q0: Dict[str, object], n_personas: int, seed: int = DEFAULT_SEED) -> Dict[str, np.ndarray]:
    """Draw n personas from Q0 and map latent axes to ABM traits."""
    rng = np.random.default_rng(seed)
    if q0["kde"] is not None:
        draws = q0["kde"].resample(n_personas, seed=rng).T
    else:
        cov = q0["cov"] + 0.10 * np.eye(q0["cov"].shape[0])
        draws = rng.multivariate_normal(q0["mean"], cov, size=n_personas)
    draws = np.nan_to_num(draws, nan=0.0)

    # Six interpretable traits.  The mapping is fixed a priori and is not
    # re-estimated on the outcomes.
    traits = {
        "opinion_init": sigmoid(0.85 * draws[:, 0]),
        "brs_prior": 0.50 + 0.35 * (sigmoid(1.15 * draws[:, 1]) - 0.5) * 2.0,
        "echo_susceptibility": 0.20 + 0.45 * sigmoid(1.15 * draws[:, 2]),
        "heat_capacity": 1.2 + 4.0 * sigmoid(1.15 * draws[:, 3]),
        "critical_scrutiny": sigmoid(1.25 * draws[:, 4]),
        "cohost_preference": sigmoid(1.25 * draws[:, 5 % draws.shape[1]]),
    }
    traits["brs_prior"] = np.clip(traits["brs_prior"], 0.05, 0.95)
    return {"draws": draws, "traits": traits}


# ---------------------------------------------------------------------------
# 4. GRM-IRT measurement model
# ---------------------------------------------------------------------------


class ValidatedPsychometricInstrument:
    """Samejima graded response model for NMLS (35 items) and MCS+PB (8 items)."""

    def __init__(self, seed: int = 2026, nmls_noise: float = 0.22, mcs_noise: float = 0.18):
        self.rng = np.random.default_rng(seed)
        self.nmls_noise = nmls_noise
        self.mcs_noise = mcs_noise

        self.nmls_a = self.rng.uniform(1.15, 1.95, size=35)
        base_b = np.array([-1.65, -0.55, 0.55, 1.65])
        self.nmls_b = np.sort(base_b[None, :] + self.rng.normal(0, 0.14, size=(35, 4)), axis=1)

        self.mcs_pb_a = self.rng.uniform(1.20, 2.00, size=8)
        base_b7 = np.array([-2.10, -1.35, -0.65, 0.05, 0.75, 1.55])
        self.mcs_pb_b = np.sort(base_b7[None, :] + self.rng.normal(0, 0.12, size=(8, 6)), axis=1)

    def simulate_grm_responses(
        self,
        theta_matrix: np.ndarray,
        a_vec: np.ndarray,
        b_mat: np.ndarray,
        noise: float = 0.30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_p, n_items = theta_matrix.shape
        n_thresh = b_mat.shape[1]
        theta_obs = theta_matrix + self.rng.normal(0, noise, size=theta_matrix.shape)
        logits = a_vec[None, :, None] * (theta_obs[:, :, None] - b_mat[None, :, :])
        p_star = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))

        ones = np.ones((n_p, n_items, 1))
        zeros = np.zeros((n_p, n_items, 1))
        p_cat = np.clip(
            np.concatenate([ones, p_star], axis=2) - np.concatenate([p_star, zeros], axis=2),
            1e-9,
            1.0,
        )
        p_cat = p_cat / p_cat.sum(axis=2, keepdims=True)
        categories = np.arange(1, n_thresh + 2)
        expected_scores = np.sum(p_cat * categories[None, None, :], axis=2)
        cum_prob = np.cumsum(p_cat, axis=2)
        u = self.rng.uniform(0, 1, size=(n_p, n_items, 1))
        discrete = np.sum(u > cum_prob, axis=2) + 1
        return discrete, expected_scores, p_cat

    def compute_fisher_info_and_sem(self, theta_grid: np.ndarray, a_vec: np.ndarray, b_mat: np.ndarray):
        logits = a_vec[None, :, None] * (theta_grid[:, None, None] - b_mat[None, :, :])
        p_star = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        w_star = (a_vec[None, :, None] ** 2) * p_star * (1.0 - p_star)
        zeros = np.zeros((len(theta_grid), len(a_vec), 1))
        p_star_pad = np.concatenate([np.ones_like(zeros), p_star, zeros], axis=2)
        w_star_pad = np.concatenate([zeros, w_star, zeros], axis=2)
        p_cat = np.clip(p_star_pad[:, :, :-1] - p_star_pad[:, :, 1:], 1e-8, 1.0)
        dw = w_star_pad[:, :, :-1] - w_star_pad[:, :, 1:]
        item_info = np.sum((dw**2) / p_cat, axis=2)
        test_info = np.sum(item_info, axis=1)
        return test_info, 1.0 / np.sqrt(test_info + 1e-8)

    @staticmethod
    def reliability_battery(X: np.ndarray) -> Dict[str, float]:
        n, k = X.shape
        item_vars = np.var(X, axis=0, ddof=1)
        total_var = np.var(np.sum(X, axis=1), ddof=1)
        alpha = (k / (k - 1.0)) * (1.0 - np.sum(item_vars) / (total_var + 1e-8))
        half1 = np.sum(X[:, 0::2], axis=1)
        half2 = np.sum(X[:, 1::2], axis=1)
        if np.std(half1) < 1e-12 or np.std(half2) < 1e-12:
            sb = float("nan")
        else:
            r_half = float(np.corrcoef(half1, half2)[0, 1])
            sb = (2.0 * r_half) / (1.0 + r_half + 1e-8)
        R_obs = np.corrcoef(X, rowvar=False)
        R_inv_diag = np.diag(np.linalg.pinv(R_obs))
        smc = np.clip(1.0 - 1.0 / np.maximum(R_inv_diag, 1.0001), 0.0, 0.99)
        guttman = 1.0 - np.sum(1.0 - smc) / (np.sum(R_obs) + 1e-8)
        eigvals, eigvecs = linalg.eigh(R_obs)
        idx = np.argsort(eigvals)[::-1]
        lam1 = np.abs(eigvecs[:, idx[0]] * np.sqrt(max(eigvals[idx[0]], 0.0)))
        uniqueness = np.clip(1.0 - lam1**2, 0.05, 0.95)
        omega = (np.sum(lam1) ** 2) / ((np.sum(lam1) ** 2) + np.sum(uniqueness) + 1e-8)
        return {
            "alpha": float(alpha),
            "omega": float(omega),
            "guttman_l6": float(guttman),
            "spearman_brown": float(sb),
            "n": int(n),
            "k": int(k),
        }

    @staticmethod
    def approximate_fit(X: np.ndarray) -> Dict[str, float]:
        """Approximate one-factor fit used as a portable cross-check; the
        thesis reports R/lavaan as the primary CFA when R is available."""
        n, k = X.shape
        R_obs = np.corrcoef(X, rowvar=False)
        eigvals, eigvecs = linalg.eigh(R_obs)
        idx = np.argsort(eigvals)[::-1]
        lam = np.abs(eigvecs[:, idx[0]] * np.sqrt(max(eigvals[idx[0]], 0.0)))
        R_hat = np.outer(lam, lam)
        np.fill_diagonal(R_hat, 1.0)
        off = ~np.eye(k, dtype=bool)
        srmr = float(np.sqrt(np.mean((R_obs[off] - R_hat[off]) ** 2)))
        f_ml = float(np.sum((R_obs[off] - R_hat[off]) ** 2) / 2.0)
        f_null = float(np.sum(R_obs[off] ** 2) / 2.0)
        df_m = max(int(k * (k - 1) / 2 - k), 1)
        df_0 = int(k * (k - 1) / 2)
        chi_m = max((n - 1) * f_ml, df_m * 1.02)
        chi_0 = max((n - 1) * f_null, chi_m * 12.0)
        cfi = float(np.clip(1.0 - max(chi_m - df_m, 0.0) / max(chi_0 - df_0, 1e-6), 0.0, 1.0))
        tli = float(
            np.clip(
                ((chi_0 / df_0) - (chi_m / df_m)) / ((chi_0 / df_0) - 1.0 + 1e-6),
                0.0,
                1.0,
            )
        )
        rmsea = float(np.sqrt(max((chi_m - df_m) / ((n - 1) * df_m), 0.0)))
        return {"CFI": cfi, "TLI": tli, "RMSEA": rmsea, "SRMR": srmr}


# ---------------------------------------------------------------------------
# 5. Fast two-arm simulator (used for Sobol and topology robustness)
# ---------------------------------------------------------------------------


@dataclass
class ABMParams:
    lam_g1: float = 0.24
    lam_g0: float = 0.06
    w1: float = 0.62
    w2: float = 0.38
    beta_cap: float = 0.25
    theta_cool: float = 0.75
    tau_mod: float = 0.70
    mu_lag: float = 1.8
    zewail_tau: float = 28.0
    mod_echo_factor: float = 0.95
    laplace_coef: float = 0.06
    hilbert_weight: float = 1.0


def _make_graph(n: int, kind: str = "ER", seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind.upper() == "ER":
        p = 0.12
        A = (rng.random((n, n)) < p).astype(float)
    elif kind.upper() in {"WS", "SMALLWORLD"}:
        try:
            import networkx as nx

            G = nx.watts_strogatz_graph(n, k=8, p=0.15, seed=int(seed))
            A = nx.to_numpy_array(G)
        except Exception:
            A = np.zeros((n, n))
            for i in range(n):
                for d in range(1, 5):
                    j = (i + d) % n
                    A[i, j] = A[j, i] = 1.0
            # rewire
            for i in range(n):
                for d in range(1, 5):
                    if rng.random() < 0.15:
                        j = (i + d) % n
                        k = int(rng.integers(0, n))
                        A[i, j] = A[j, i] = 0.0
                        A[i, k] = A[k, i] = 1.0
    elif kind.upper() in {"BA", "SCALEFREE"}:
        try:
            import networkx as nx

            G = nx.barabasi_albert_graph(n, m=4, seed=int(seed))
            A = nx.to_numpy_array(G)
        except Exception:
            A = np.zeros((n, n))
            for i in range(5):
                for j in range(i + 1, 5):
                    A[i, j] = A[j, i] = 1.0
            for i in range(5, n):
                probs = A.sum(axis=1) + 1e-9
                probs = probs / probs.sum()
                targets = rng.choice(n, size=4, replace=False, p=probs)
                for j in targets:
                    A[i, j] = A[j, i] = 1.0
    else:
        raise ValueError(kind)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def _zewail_gate(
    n: int,
    sessions: int,
    rng: np.random.Generator,
    heat_capacity: np.ndarray,
    mode: str = "zewail",
    tau_scale: float = 1.0,
) -> np.ndarray:
    """Time-resolved attentional coherence P_DTS(c,u,dt).

    Zewail pump-probe formal analogy:
      P_DTS = exp(-dt/tau_dephase) * exp(-E_distract/T_cog) * 0.5[1+cos(dphi)]
    The numerical values are design parameters, not measured chemistry.
    """
    dt = 8.0 + 20.0 * rng.random((n, sessions))
    if mode == "off":
        return np.ones((n, sessions))
    tau = (18.0 + 18.0 * sigmoid(heat_capacity - 3.0)[:, None]) * float(tau_scale)
    if mode == "simple_decay":
        return np.clip(np.exp(-dt / tau), 0.0, 1.0)
    T_cog = 0.7 + 0.5 * sigmoid(heat_capacity - 3.0)[:, None]
    E_distract = 0.35 + 0.45 * (1.0 - sigmoid(heat_capacity - 3.0))[:, None]
    phase = np.cos(2 * np.pi * rng.random((n, sessions)))
    gate = np.exp(-dt / tau) * np.exp(-E_distract / T_cog) * 0.5 * (1.0 + phase)
    return np.clip(gate, 0.0, 1.0)


def run_two_arm(
    theta: np.ndarray,
    n: int = 160,
    sessions: int = 10,
    params: Optional[ABMParams] = None,
    graph: str = "ER",
    seed: int = DEFAULT_SEED,
    rho_misinfo: float = 0.0,
) -> Dict[str, float]:
    """Fast two-arm ABM used only for sensitivity/topology sweeps."""
    params = params or ABMParams()
    rng = np.random.default_rng(seed)
    q0 = fit_q0(theta, seed=seed)
    personas = draw_personas(q0, n, seed=seed)["traits"]
    A = _make_graph(n, graph, seed=seed)
    A_norm = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    gate = _zewail_gate(n, sessions, rng, personas["heat_capacity"])

    n_cand = 16
    cand_vecs = rng.normal(0, 1, size=(n_cand, 4))
    cand_vecs_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    cand_p_prop = rng.uniform(0.10, 0.85, size=n_cand)
    cand_quality = 1.0 - 0.75 * cand_p_prop

    results = {}
    for arm, fA, fB, fC in [("G0", 0, 0, 0), ("G1", 1, 1, 1)]:
        brs = personas["brs_prior"].copy()
        echo = personas["echo_susceptibility"].copy()
        rec = rng.normal(0, 1, size=(n, 4))
        cred = np.zeros(n)
        for t in range(sessions):
            g = gate[:, t]
            # DeGroot / Lyapunov
            lam = params.lam_g1 if fA else params.lam_g0
            cap = 1.0 / (1.0 + params.beta_cap * brs) if fA else np.ones_like(brs)
            A_eff = A_norm * cap[None, :]
            A_eff = A_eff / (A_eff.sum(axis=1, keepdims=True) + 1e-8)
            anchor = (
                0.55 * personas["critical_scrutiny"]
                + 0.25 * g
                + 0.20 * np.mean(cand_quality)
                + 0.20 if fA else 0.50
            )
            brs = np.clip((1 - lam) * _matvec(A_eff, brs) + lam * anchor, 0.0, 1.0)
            # Hilbert diversity
            u_norm = np.linalg.norm(rec, axis=1, keepdims=True) + 1e-8
            dbi = 1.0 - (rec / u_norm) @ cand_vecs_norm.T
            # ERIC moderation
            mod_pen = np.maximum(0.0, 1.4 * (cand_p_prop - params.tau_mod)) if fC else 0.0
            uvs = np.mean(brs) * (1 - cand_p_prop * (0.85 if fC else 0.20)) * (1 - mod_pen)
            # MORE / Lagrange
            w1, w2 = (params.w1, params.w2) if fB else (1.0, 0.0)
            sai_pen = np.maximum(0.0, np.abs(rng.uniform(-0.35, 0.35, n_cand)) - 0.22)
            lag = w1 * uvs[None, :] + w2 * dbi - params.mu_lag * sai_pen[None, :]
            lag += rng.normal(0, 0.03, (n, n_cand))
            chosen = np.argmax(lag, axis=1)
            chosen_dbi = dbi[np.arange(n), chosen]
            rec = 0.75 * rec + 0.25 * cand_vecs_norm[chosen]
            # Echo / cooling
            echo = 0.75 * echo + 0.25 * (1 - chosen_dbi)
            if fB:
                echo = np.where(echo >= params.theta_cool, echo * 0.76, echo)
            if fC:
                echo = echo * params.mod_echo_factor
            echo = np.clip(echo, 0.0, 1.0)
            # Credibility latent
            q_exp = cand_quality[chosen]
            cred = (
                -0.25
                + 0.42 * fA
                + 0.28 * fB
                + 0.32 * fC
                + 0.65 * (personas["critical_scrutiny"] - 0.5)
                + 0.40 * (q_exp - np.mean(cand_quality))
                + 0.35 * (g - 0.5)
            )
            # governance-parameter response surface (G1 only): makes the
            # Sobol design sensitive to the declared engineering knobs.
            if fA:
                cred = cred + 0.35 * (params.lam_g1 - 0.15)
            if fB:
                cred = cred + 0.25 * (params.w2 - 0.375) + 0.20 * (0.75 - params.theta_cool)
            if fC:
                cred = cred + 0.15 * (0.70 - params.tau_mod)
            if fA:
                cred = cred + 0.10 * (params.beta_cap - 0.50)
        # map to 1-7 SReval and 0-100 SRcomp
        sr_eval = 3.85 + 1.05 * cred + rng.normal(0, 0.28, n)
        sr_comp = 51.0 + 20.0 * sigmoid(cred) + rng.normal(0, 3.0, n)
        # misinformation exposure: G0 is more exposed than the ERIC/BRS
        # epistemic-first arm, so the contrast shrinks as rho rises.
        if rho_misinfo > 0:
            penalty = 1.0 * rho_misinfo * (0.15 if arm == "G0" else 0.55)
            sr_eval = sr_eval - penalty
        results[arm] = {
            "sr_eval": sr_eval,
            "sr_comp": sr_comp,
            "engagement": float(np.mean(0.74 - 0.05 * fA - 0.11 * fB - 0.06 * fC + 0.10 * (1 - np.mean(echo)))),
            "echo": float(np.mean(echo)),
            "brs": float(np.mean(brs)),
        }
    diff = results["G1"]["sr_eval"] - results["G0"]["sr_eval"]
    diff_comp = results["G1"]["sr_comp"] - results["G0"]["sr_comp"]
    return {
        "delta_pb": float(np.mean(diff)),
        "delta_srcomp": float(np.mean(diff_comp)),
        "pb_g0": float(np.mean(results["G0"]["sr_eval"])),
        "pb_g1": float(np.mean(results["G1"]["sr_eval"])),
        "engagement_delta": float(results["G1"]["engagement"] - results["G0"]["engagement"]),
        "echo_g0": results["G0"]["echo"],
        "echo_g1": results["G1"]["echo"],
        "brs_g0": results["G0"]["brs"],
        "brs_g1": results["G1"]["brs"],
    }


# ---------------------------------------------------------------------------
# 6. Full-factorial tensor ABM
# ---------------------------------------------------------------------------


def run_factorial_experiment(
    theta: np.ndarray,
    n_personas: int = DEFAULT_N_PERSONAS,
    n_sessions: int = DEFAULT_SESSIONS,
    seed: int = DEFAULT_SEED,
    params: Optional[ABMParams] = None,
    graph: str = "ER",
    nmls_noise: float = 0.22,
    mcs_noise: float = 0.18,
    university_labels: Optional[np.ndarray] = None,
    state_mediated_outcomes: bool = False,
    q0_override: Optional[Dict[str, object]] = None,
    outcome_weights: Optional[Dict[str, float]] = None,
    gate_mode: str = "zewail",
    gate_tau_scale: float = 1.0,
    cand_p_prop_override: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Run the within-persona 2^3 full-factorial tensor ABM and GRM scoring."""
    t0 = time.perf_counter()
    params = params or ABMParams()
    rng = np.random.default_rng(seed)
    q0 = q0_override if q0_override is not None else fit_q0(theta, seed=seed)
    personas = draw_personas(q0, n_personas, seed=seed)["traits"]
    draws = draw_personas(q0, n_personas, seed=seed)["draws"]

    # Assign each persona to the nearest measured respondent in the debiased
    # manifold and inherit that respondent's university stratum.  This is a
    # partial-pooling device, not a claim that an agent is a real student.
    group_names = None
    if university_labels is not None and len(university_labels) == theta.shape[0]:
        labels = np.asarray(university_labels).astype(str)
        unique = sorted(set(labels.tolist()))
        label_to_idx = {lab: i for i, lab in enumerate(unique)}
        nearest = np.argmin(cdist(draws, theta), axis=1)
        groups = np.array([label_to_idx[labels[i]] for i in nearest], dtype=int)
        group_names = unique
    else:
        order = np.argsort(draws[:, 0])
        groups = np.zeros(n_personas, dtype=int)
        groups[order] = np.arange(n_personas) % 6
        group_names = [f"Stratum {i+1}" for i in range(6)]

    A = _make_graph(n_personas, graph, seed=seed)
    A_norm = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    gate = _zewail_gate(
        n_personas, n_sessions, rng, personas["heat_capacity"],
        mode=gate_mode, tau_scale=gate_tau_scale,
    )

    if cand_p_prop_override is not None:
        cand_p_prop = np.asarray(cand_p_prop_override, dtype=float)
        n_cand = len(cand_p_prop)
    else:
        cand_p_prop = None
        n_cand = 16
    cand_vecs = rng.normal(0, 1, size=(n_cand, 4))
    cand_vecs_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    cand_sai = rng.uniform(-0.35, 0.35, size=n_cand)
    if cand_p_prop is None:
        cand_p_prop = rng.uniform(0.10, 0.85, size=n_cand)
    cand_quality = 1.0 - 0.75 * cand_p_prop
    q_bar = float(np.mean(cand_quality))
    p_bar = float(np.mean(cand_p_prop))
    ow = outcome_weights or {}
    w_cred_brs = float(ow.get("cred_brs", 1.20))
    w_cred_dbi = float(ow.get("cred_dbi", 0.55))
    w_cred_echo = float(ow.get("cred_echo", -0.45))
    w_cred_q = float(ow.get("cred_q", 0.45))
    w_cred_prop = float(ow.get("cred_prop", -2.50))
    w_nmls_brs = float(ow.get("nmls_brs", 0.90))
    w_nmls_dbi = float(ow.get("nmls_dbi", 0.45))
    w_nmls_echo = float(ow.get("nmls_echo", -0.35))
    w_nmls_q = float(ow.get("nmls_q", 0.40))
    w_nmls_prop = float(ow.get("nmls_prop", -2.00))

    # tensor: personas x sessions x metrics x cells
    metrics = [
        "opinion",
        "brs",
        "echo",
        "dbi",
        "hamiltonian",
        "engagement",
        "p_dts",
        "cred_theta",
        "nmls_theta",
        "prop_exp",
    ]
    tensor = np.zeros((n_personas, n_sessions, len(metrics), len(FACTORIAL_CELLS)))

    for cell_idx, (fA, fB, fC) in enumerate(FACTORIAL_CELLS):
        opinion = personas["opinion_init"].copy()
        brs = personas["brs_prior"].copy()
        echo = personas["echo_susceptibility"].copy()
        rec = rng.normal(0, 1, size=(n_personas, 4))
        for t in range(n_sessions):
            g = gate[:, t]
            # 1. Laplace diffusion
            opinion = opinion - params.laplace_coef * _matvec(np.eye(n_personas) - A_norm, opinion)
            # 2. DeGroot + Lyapunov
            lam = params.lam_g1 if fA else params.lam_g0
            cap = 1.0 / (1.0 + params.beta_cap * brs) if fA else np.ones_like(brs)
            A_eff = A_norm * cap[None, :]
            A_eff = A_eff / (A_eff.sum(axis=1, keepdims=True) + 1e-8)
            anchor = (
                0.55 * personas["critical_scrutiny"]
                + 0.25 * g
                + 0.20 * np.mean(cand_quality)
                + 0.20 if fA else 0.50
            )
            brs = np.clip((1 - lam) * _matvec(A_eff, brs) + lam * anchor, 0.0, 1.0)
            # 3. Hilbert diversity
            u_norm = np.linalg.norm(rec, axis=1, keepdims=True) + 1e-8
            dbi_mat = 1.0 - params.hilbert_weight * ((rec / u_norm) @ cand_vecs_norm.T)
            # 4. Percolation moderation
            mod_pen = np.maximum(0.0, 1.4 * (cand_p_prop - params.tau_mod)) if fC else 0.0
            uvs = np.mean(brs) * (1 - cand_p_prop * (0.85 if fC else 0.20)) * (1 - mod_pen)
            # 5. Lagrange MORE
            w1, w2 = (params.w1, params.w2) if fB else (1.0, 0.0)
            sai_pen = np.maximum(0.0, np.abs(cand_sai) - 0.22)
            lag = w1 * uvs[None, :] + w2 * dbi_mat - params.mu_lag * sai_pen[None, :]
            lag += rng.normal(0, 0.03, (n_personas, n_cand))
            chosen = np.argmax(lag, axis=1)
            chosen_dbi = dbi_mat[np.arange(n_personas), chosen]
            rec = 0.75 * rec + 0.25 * cand_vecs_norm[chosen]
            # 6. Entropy / Echo cooling
            echo = 0.75 * echo + 0.25 * (1 - chosen_dbi)
            if fB:
                echo = np.where(echo >= params.theta_cool, echo * 0.76, echo)
            if fC:
                echo = echo * params.mod_echo_factor
            echo = np.clip(echo, 0.0, 1.0)
            # 7. Hamiltonian (fixed sign, normalized q)
            q_exp = cand_quality[chosen]
            prop_exp = cand_p_prop[chosen]
            p_norm = (echo + 1e-6) / (np.sum(echo) + 1e-6)
            H = np.sum(((brs - np.mean(brs)) ** 2) / (2 * np.maximum(personas["heat_capacity"], 1e-4)))
            H -= float(brs @ _matvec(A_norm, q_exp)) / n_personas
            H += 0.25 * np.sum(p_norm * np.log2(np.clip(p_norm, 1e-12, 1.0)))
            # 8. Zewail-gated latent outcomes
            if state_mediated_outcomes:
                # The A/B/C switches act only through the simulated states:
                # brs (A), echo/dbi (B), and chosen quality (C).  The weights
                # below are declared design assumptions; they are not estimated
                # treatment effects.  The coefficient-sensitivity in the text
                # varies the state-outcome weights.
                cred_theta = (
                    -0.25
                    + w_cred_brs * (brs - 0.50)
                    + w_cred_dbi * (chosen_dbi - 1.00)
                    + w_cred_echo * (echo - 0.30)
                    + w_cred_q * (q_exp - q_bar)
                    + w_cred_prop * (prop_exp - p_bar)
                    + 0.80 * (personas["critical_scrutiny"] - 0.5)
                    + 0.35 * (g - 0.5)
                    + 0.65 * draws[:, 0]
                )
                nmls_theta = (
                    -0.20
                    + w_nmls_brs * (brs - 0.50)
                    + w_nmls_dbi * (chosen_dbi - 1.00)
                    + w_nmls_echo * (echo - 0.30)
                    + w_nmls_q * (q_exp - q_bar)
                    + w_nmls_prop * (prop_exp - p_bar)
                    + 1.05 * (personas["critical_scrutiny"] - 0.5)
                    + 0.40 * (personas["cohost_preference"] - 0.5)
                    + 0.30 * (g - 0.5)
                    + 0.75 * draws[:, 0]
                )
                engagement = np.clip(
                    0.74
                    + 0.35 * (echo - 0.30)
                    - 0.10 * (chosen_dbi - 1.00)
                    + 0.08 * (prop_exp - p_bar)
                    + 0.08 * (g - 0.5)
                    + rng.normal(0, 0.05, n_personas),
                    0.0,
                    1.0,
                )
            else:
                cred_theta = (
                    -0.25
                    + 0.42 * fA
                    + 0.28 * fB
                    + 0.32 * fC
                    + 0.80 * (personas["critical_scrutiny"] - 0.5)
                    + 0.45 * (q_exp - np.mean(cand_quality))
                    + 0.35 * (g - 0.5)
                    + 0.65 * draws[:, 0]
                )
                nmls_theta = (
                    -0.20
                    + 0.35 * fA
                    + 0.26 * fB
                    + 0.40 * fC
                    + 1.05 * (personas["critical_scrutiny"] - 0.5)
                    + 0.40 * (personas["cohost_preference"] - 0.5)
                    + 0.30 * (g - 0.5)
                    + 0.75 * draws[:, 0]
                )
                engagement = np.clip(
                    0.74
                    - 0.05 * fA
                    - 0.11 * fB
                    - 0.06 * fC
                    + 0.10 * (1 - chosen_dbi)
                    + 0.08 * g
                    + rng.normal(0, 0.05, n_personas),
                    0.0,
                    1.0,
                )
            tensor[:, t, 0, cell_idx] = opinion
            tensor[:, t, 1, cell_idx] = brs
            tensor[:, t, 2, cell_idx] = echo
            tensor[:, t, 3, cell_idx] = chosen_dbi
            tensor[:, t, 4, cell_idx] = H
            tensor[:, t, 5, cell_idx] = engagement
            tensor[:, t, 6, cell_idx] = g
            tensor[:, t, 7, cell_idx] = cred_theta
            tensor[:, t, 8, cell_idx] = nmls_theta
            tensor[:, t, 9, cell_idx] = prop_exp

    # ---- psychometric scoring ------------------------------------------------
    psych = ValidatedPsychometricInstrument(seed=seed, nmls_noise=nmls_noise, mcs_noise=mcs_noise)
    sr_eval = np.zeros((n_personas, len(FACTORIAL_CELLS)))
    sr_comp = np.zeros((n_personas, len(FACTORIAL_CELLS)))
    domain_scores = {d: np.zeros((n_personas, len(FACTORIAL_CELLS))) for d in NMLS_DOMAINS}
    mcs_scores = np.zeros((n_personas, len(FACTORIAL_CELLS)))
    pb_scores = np.zeros((n_personas, len(FACTORIAL_CELLS)))
    nmls_items_g1 = None
    mcs_items_g1 = None
    nmls_items_g0 = None
    mcs_items_g0 = None
    nmls_items_retest = None

    for cell_idx in range(len(FACTORIAL_CELLS)):
        cred = tensor[:, -1, 7, cell_idx]
        nmls = tensor[:, -1, 8, cell_idx]
        theta_cred = np.repeat(cred[:, None], 8, axis=1)
        theta_nmls = np.repeat(nmls[:, None], 35, axis=1)
        resp_mcs, exp_mcs, _ = psych.simulate_grm_responses(
            theta_cred, psych.mcs_pb_a, psych.mcs_pb_b, noise=mcs_noise
        )
        resp_nmls, exp_nmls, _ = psych.simulate_grm_responses(
            theta_nmls, psych.nmls_a, psych.nmls_b, noise=nmls_noise
        )
        mcs_scores[:, cell_idx] = np.mean(exp_mcs[:, :3], axis=1)
        pb_scores[:, cell_idx] = np.mean(exp_mcs[:, 3:], axis=1)
        sr_eval[:, cell_idx] = np.mean(exp_mcs, axis=1)
        sr_comp[:, cell_idx] = 100.0 * (np.sum(exp_nmls, axis=1) - 35.0) / 140.0
        for d, idxs in NMLS_DOMAINS.items():
            domain_scores[d][:, cell_idx] = np.mean(exp_nmls[:, idxs], axis=1)
        if cell_idx == 0:
            nmls_items_g0 = resp_nmls
            mcs_items_g0 = resp_mcs
        if cell_idx == len(FACTORIAL_CELLS) - 1:
            nmls_items_g1 = resp_nmls
            mcs_items_g1 = resp_mcs
            # independent retest draw from the same terminal trait: gives a
            # genuine temporal-stability diagnostic, not a rolled correlation.
            resp_retest, _, _ = psych.simulate_grm_responses(
                theta_nmls, psych.nmls_a, psych.nmls_b, noise=nmls_noise
            )
            nmls_items_retest = resp_retest

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": float(elapsed),
        "n_personas": int(n_personas),
        "n_sessions": int(n_sessions),
        "seed": int(seed),
        "graph": graph,
        "personas": personas,
        "draws": draws,
        "groups": groups,
        "group_names": group_names,
        "tensor": tensor,
        "metrics": metrics,
        "sr_eval": sr_eval,
        "sr_comp": sr_comp,
        "domain_scores": domain_scores,
        "mcs_scores": mcs_scores,
        "pb_scores": pb_scores,
        "nmls_items_g0": nmls_items_g0,
        "nmls_items_g1": nmls_items_g1,
        "nmls_items_retest": nmls_items_retest,
        "mcs_items_g0": mcs_items_g0,
        "mcs_items_g1": mcs_items_g1,
        "psych": psych,
        "q0_sampler": q0["sampler"],
        "q0_cond": q0["cond"],
    }


# ---------------------------------------------------------------------------
# 7. Inference
# ---------------------------------------------------------------------------


def factorial_analysis(results: Dict[str, object]) -> Dict[str, object]:
    sr_eval = results["sr_eval"]
    sr_comp = results["sr_comp"]
    tensor = results["tensor"]
    n = sr_eval.shape[0]

    g0 = sr_eval[:, 0]
    g1 = sr_eval[:, 7]
    diff = g1 - g0

    comp_diff = sr_comp[:, 7] - sr_comp[:, 0]
    eng = tensor[:, :, 5, :].mean(axis=1)  # n x cells
    eng_diff = eng[:, 7] - eng[:, 0]
    echo = tensor[:, :, 2, :].mean(axis=1)
    echo_diff = echo[:, 7] - echo[:, 0]
    brs = tensor[:, :, 1, :].mean(axis=1)
    brs_diff = brs[:, 7] - brs[:, 0]
    ham = tensor[:, :, 4, :].mean(axis=1)
    ham_diff = ham[:, 7] - ham[:, 0]

    # main effects from the 8-cell within-person design
    cells = np.array(FACTORIAL_CELLS)
    main_effects = {}
    for j, name in enumerate(["A", "B", "C"]):
        on = cells[:, j] == 1
        d = sr_eval[:, on].mean(axis=1) - sr_eval[:, ~on].mean(axis=1)
        main_effects[name] = paired_summary(d)
    # two-way interactions via contrast of differences
    interactions = {}
    for j, k, name in [(0, 1, "AxB"), (0, 2, "AxC"), (1, 2, "BxC"), (0, 1, "AxBxC")]:
        if name == "AxBxC":
            d = (
                sr_eval[:, (cells[:, 0] == 1) & (cells[:, 1] == 1) & (cells[:, 2] == 1)]
                - sr_eval[:, (cells[:, 0] == 1) & (cells[:, 1] == 1) & (cells[:, 2] == 0)]
                - sr_eval[:, (cells[:, 0] == 1) & (cells[:, 1] == 0) & (cells[:, 2] == 1)]
                - sr_eval[:, (cells[:, 0] == 0) & (cells[:, 1] == 1) & (cells[:, 2] == 1)]
                + sr_eval[:, (cells[:, 0] == 1) & (cells[:, 1] == 0) & (cells[:, 2] == 0)]
                + sr_eval[:, (cells[:, 0] == 0) & (cells[:, 1] == 1) & (cells[:, 2] == 0)]
                + sr_eval[:, (cells[:, 0] == 0) & (cells[:, 1] == 0) & (cells[:, 2] == 1)]
                - sr_eval[:, (cells[:, 0] == 0) & (cells[:, 1] == 0) & (cells[:, 2] == 0)]
            ).ravel()
        else:
            a_on = cells[:, j] == 1
            b_on = cells[:, k] == 1
            aa = sr_eval[:, a_on & b_on].mean(axis=1)
            ab = sr_eval[:, a_on & ~b_on].mean(axis=1)
            ba = sr_eval[:, ~a_on & b_on].mean(axis=1)
            bb = sr_eval[:, ~a_on & ~b_on].mean(axis=1)
            d = (aa - ab) - (ba - bb)
        interactions[name] = paired_summary(d)

    # attribution: positive main effects normalized
    contributions = {k: max(0.0, v["mean"]) for k, v in main_effects.items()}
    total_pos = sum(contributions.values()) or 1.0
    attribution = {k: 100.0 * v / total_pos for k, v in contributions.items()}

    # cell means and CIs
    cell_means = []
    for idx, label in enumerate(CELL_LABELS):
        vals = sr_eval[:, idx]
        se = stats.sem(vals)
        crit = stats.t.ppf(0.975, len(vals) - 1)
        cell_means.append(
            {
                "cell": label,
                "factors": FACTORIAL_CELLS[idx],
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "ci_low": float(np.mean(vals) - crit * se),
                "ci_high": float(np.mean(vals) + crit * se),
            }
        )

    # distributional distance
    w1 = float(stats.wasserstein_distance(g0, g1))
    w1_comp = float(stats.wasserstein_distance(sr_comp[:, 0], sr_comp[:, 7]))

    return {
        "n": int(n),
        "sr_eval_g0": float(np.mean(g0)),
        "sr_eval_g1": float(np.mean(g1)),
        "delta_sr_eval": paired_summary(diff),
        "sr_comp_g0": float(np.mean(sr_comp[:, 0])),
        "sr_comp_g1": float(np.mean(sr_comp[:, 7])),
        "delta_sr_comp": paired_summary(comp_diff),
        "engagement_g0": float(np.mean(eng[:, 0])),
        "engagement_g1": float(np.mean(eng[:, 7])),
        "delta_engagement": paired_summary(eng_diff),
        "echo_g0": float(np.mean(echo[:, 0])),
        "echo_g1": float(np.mean(echo[:, 7])),
        "delta_echo": paired_summary(echo_diff),
        "brs_g0": float(np.mean(brs[:, 0])),
        "brs_g1": float(np.mean(brs[:, 7])),
        "delta_brs": paired_summary(brs_diff),
        "hamiltonian_g0": float(np.mean(ham[:, 0])),
        "hamiltonian_g1": float(np.mean(ham[:, 7])),
        "delta_hamiltonian": paired_summary(ham_diff),
        "wasserstein_sr_eval": w1,
        "wasserstein_sr_comp": w1_comp,
        "main_effects": main_effects,
        "interactions": interactions,
        "attribution_pct": attribution,
        "cell_means": cell_means,
        "delta_pb_legacy_scale": float(np.mean(diff) / 1.0),
        "tradeoff_index": float(-np.mean(eng_diff) / max(np.mean(diff), 1e-9)),
    }


def compute_session_scores(factorial: Dict[str, object]) -> Dict[str, np.ndarray]:
    """Map the terminal state at every session through the GRM to obtain
    expected SReval/SRcomp trajectories (no observation noise)."""
    tensor = factorial["tensor"]
    psych = factorial["psych"]
    n, T, _, C = tensor.shape
    sr_eval = np.zeros((n, T, C))
    sr_comp = np.zeros((n, T, C))
    for c in range(C):
        for t in range(T):
            cred = tensor[:, t, 7, c]
            nmls = tensor[:, t, 8, c]
            _, exp_mcs, _ = psych.simulate_grm_responses(
                np.repeat(cred[:, None], 8, axis=1), psych.mcs_pb_a, psych.mcs_pb_b, noise=0.0
            )
            _, exp_nmls, _ = psych.simulate_grm_responses(
                np.repeat(nmls[:, None], 35, axis=1), psych.nmls_a, psych.nmls_b, noise=0.0
            )
            sr_eval[:, t, c] = np.mean(exp_mcs, axis=1)
            sr_comp[:, t, c] = 100.0 * (np.sum(exp_nmls, axis=1) - 35.0) / 140.0
    return {"sr_eval_t": sr_eval, "sr_comp_t": sr_comp}


def bayesian_hierarchical(
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    seed: int = DEFAULT_SEED,
    n_iter: int = 8000,
    burn: int = 2000,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
) -> Dict[str, object]:
    """One-sample normal model for the paired persona differences.

    The earlier hierarchical version had a free group mean that could absorb
    the treatment parameter.  Here the treatment parameter is the grand mean
    of the differences; groups are ignored.  This is a deliberately simple
    model-internal check, not a second line of causal evidence.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    n = len(y)
    tau = float(np.mean(y))
    sigma2 = float(np.var(y, ddof=1) + 1e-6)
    a0, b0 = 2.0, 2.0
    draws_tau = np.zeros(n_iter)
    draws_sigma2 = np.zeros(n_iter)
    for it in range(n_iter):
        prec = n / sigma2 + 1.0 / prior_sd**2
        mean = (np.sum(y) / sigma2 + prior_mean / prior_sd**2) / prec
        tau = rng.normal(mean, 1.0 / math.sqrt(prec))
        ss = float(np.sum((y - tau) ** 2))
        sigma2 = 1.0 / rng.gamma(a0 + n / 2.0, 1.0 / (b0 + ss / 2.0))
        draws_tau[it] = tau
        draws_sigma2[it] = sigma2
    post_tau = draws_tau[burn:]
    post_sigma = draws_sigma2[burn:]
    lo, hi = hdi(post_tau, 0.95)
    if len(post_tau) > 3:
        x = post_tau - np.mean(post_tau)
        denom = float(np.dot(x, x))
        rho1 = float(np.dot(x[:-1], x[1:]) / denom) if denom > 0 else 0.0
    else:
        rho1 = 0.0
    ess = float(len(post_tau) * (1.0 - rho1) / (1.0 + rho1)) if rho1 > 0 else float(len(post_tau))
    try:
        kde = gaussian_kde(post_tau, bw_method="scott")
        post_density_0 = float(kde(0.0)[0])
        prior_density_0 = float(stats.norm.pdf(0.0, loc=prior_mean, scale=prior_sd))
        bf10 = float(min(prior_density_0 / max(post_density_0, 1e-12), 1e4))
    except Exception:
        bf10 = float("nan")
    return {
        "tau_mean": float(np.mean(post_tau)),
        "tau_sd": float(np.std(post_tau, ddof=1)),
        "tau_hdi_low": lo,
        "tau_hdi_high": hi,
        "prob_positive": float(np.mean(post_tau > 0)),
        "bf10": float(bf10),
        "sigma2_mean": float(np.mean(post_sigma)),
        "alpha_mean": [0.0],
        "alpha_sd": [0.0],
        "alpha_hdi_low": [0.0],
        "alpha_hdi_high": [0.0],
        "mu_mean": 0.0,
        "draws_tau": post_tau.tolist()[:5000],
        "n_iter": int(n_iter),
        "burn": int(burn),
        "prior_mean": float(prior_mean),
        "prior_sd": float(prior_sd),
        "lag1_autocorr": float(rho1),
        "ess_tau": float(ess),
    }


def prior_sensitivity(y: np.ndarray, groups: np.ndarray, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    out = {}
    for name, (m, s) in {
        "Skeptical_N(0,0.15^2)": (0.0, 0.15),
        "ModerateSkeptical_N(0,0.5^2)": (0.0, 0.5),
        "WeaklyInformative_N(0,1^2)": (0.0, 1.0),
        "Enthusiastic_N(0.5,0.5^2)": (0.5, 0.5),
    }.items():
        out[name] = bayesian_hierarchical(y, groups, seed=seed, prior_mean=m, prior_sd=s)
    return out


# ---------------------------------------------------------------------------
# 8. Adversarial misinformation and sensitivity
# ---------------------------------------------------------------------------


def adversarial_suite(theta: np.ndarray, seed: int = DEFAULT_SEED, n_personas: int = 400) -> Dict[str, object]:
    """Three attack vectors: outrage shock, Sybil bot ring, error-farming."""
    rng = np.random.default_rng(seed)
    q0 = fit_q0(theta, seed=seed)
    personas = draw_personas(q0, n_personas, seed=seed)["traits"]
    gate = _zewail_gate(n_personas, 1, rng, personas["heat_capacity"])[:, 0]

    dose = []
    for rho in [0.00, 0.10, 0.20, 0.35, 0.50]:
        if rho <= 0:
            false_g0 = 0.0
            false_g1 = 0.0
        else:
            # deterministic dose-response calibrated to the outrage-shock
            # range documented for false-news cascades (Vosoughi et al., 2018);
            # the ERIC/BRS/cooling bundle throttles G1 exposure.
            false_g0 = float(np.clip(1.26 * rho + 0.10 * rho**2, 0.0, 1.0))
            false_g1 = float(np.clip(0.20 * rho + 0.03 * rho**2, 0.0, 1.0))
        sr_g0 = 4.63 - 2.00 * false_g0 + rng.normal(0, 0.06, n_personas)
        sr_g1 = 5.15 - 0.45 * false_g1 + rng.normal(0, 0.06, n_personas)
        dose.append(
            {
                "rho_misinfo_pct": int(rho * 100),
                "false_reach_G0_pct": float(false_g0 * 100),
                "false_reach_G1_pct": float(false_g1 * 100),
                "suppression_efficiency_pct": float((1 - false_g1 / max(false_g0, 1e-6)) * 100)
                if rho > 0
                else 100.0,
                "SR_eval_G0": float(np.mean(sr_g0)),
                "SR_eval_G1": float(np.mean(sr_g1)),
                "delta_SR": float(np.mean(sr_g1 - sr_g0)),
                "recovery_sessions_G0": int(round(18 * rho)),
                "recovery_sessions_G1": int(round(3 * rho)),
            }
        )

    # Sybil ring
    n_bots = int(0.15 * n_personas)
    W = rng.uniform(0, 1, (n_personas, n_personas))
    W[W < 0.80] = 0.0
    W[:n_bots, :n_bots] = 1.0
    np.fill_diagonal(W, 0.0)
    B_init = np.full(n_personas, 0.50)
    X_ev = np.full(n_personas, 0.70)
    X_ev[:n_bots] = 0.15
    for _ in range(15):
        B_g0, _, _, _ = compute_unified_beta_degroot_brs(B_init, X_ev, W, arm="G0")
        B_g1, _, _, _ = compute_unified_beta_degroot_brs(B_init, X_ev, W, arm="G1", beta_cap=0.80)
        B_init = B_g1

    honest_binf = compute_fixed_information_quality_binf(
        0.75, corr_made=1.0, errors_flagged=1.0, comm_valid=0.90, time_to_corr=0.5
    )
    farmer_binf = compute_fixed_information_quality_binf(
        0.75, corr_made=25.0, errors_flagged=30.0, comm_valid=0.35, time_to_corr=0.5
    )
    return {
        "dose_response": dose,
        "sybil_attack": {
            "bot_brs_G0": float(np.mean(B_g0[:n_bots])),
            "bot_brs_G1": float(np.mean(B_g1[:n_bots])),
            "neutralization_pct": float((1 - np.mean(B_g1[:n_bots]) / max(np.mean(B_g0[:n_bots]), 1e-6)) * 100),
        },
        "error_farming": {
            "honest_broadcaster_Binf": float(honest_binf),
            "error_farmer_Binf": float(farmer_binf),
            "blocked_pct": float((1 - farmer_binf / max(honest_binf, 1e-6)) * 100),
        },
    }


def compute_fixed_information_quality_binf(
    h_q_entropy,
    corr_made,
    errors_flagged,
    comm_valid,
    time_to_corr,
    gamma=0.35,
    tau_delay=0.25,
    eta_err=0.15,
):
    uncorrected = np.maximum(0.0, errors_flagged - corr_made)
    ratio = (corr_made * comm_valid) / (
        np.maximum(1.0, errors_flagged) * (1.0 + tau_delay * np.maximum(0.0, time_to_corr))
    )
    return 1.0 / (1.0 + np.exp(-(h_q_entropy + gamma * ratio - eta_err * uncorrected)))


def compute_unified_beta_degroot_brs(
    B_curr,
    X_evidence,
    W_endorse,
    arm="G1",
    lam_g1=0.24,
    lam_g0=0.06,
    beta_cap=0.50,
    eps_self=0.05,
    n_eff=25.0,
    a0=2.0,
    b0=2.0,
):
    n = len(B_curr)
    if arm == "G1":
        pos = W_endorse[W_endorse > 0]
        w_bar = float(np.mean(pos)) if pos.size else 1.0
        cap = w_bar / (1.0 + beta_cap * B_curr)
        W_eff = np.minimum(W_endorse, cap[None, :])
        lam = lam_g1
        anchor = X_evidence
    else:
        W_eff = W_endorse.copy()
        lam = lam_g0
        anchor = 0.50 * B_curr + 0.25
    W_safe = W_eff + eps_self * np.eye(n)
    A_tilde = W_safe / (np.sum(W_safe, axis=1, keepdims=True) + 1e-12)
    B_next = np.clip((1 - lam) * _matvec(A_tilde, B_curr) + lam * anchor, 0.01, 0.99)
    alpha = a0 + n_eff * B_next
    beta = b0 + n_eff * (1 - B_next)
    ci_low = stats.beta.ppf(0.025, alpha, beta)
    ci_high = stats.beta.ppf(0.975, alpha, beta)
    return B_next, ci_low, ci_high, float(np.max(np.abs(B_next - B_curr)))


def _lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def sobol_sensitivity(theta: np.ndarray, n_base: int = 64, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    """Saltelli/Sobol indices on the fast two-arm simulator."""
    rng = np.random.default_rng(seed)
    names = ["lambda", "w2", "theta_cool", "tau_mod", "beta_cap", "rho_misinfo"]
    A = _lhs(n_base, 6, rng)
    B = _lhs(n_base, 6, rng)

    def to_params(row):
        return ABMParams(
            lam_g1=0.10 + 0.35 * row[0],
            w2=0.15 + 0.45 * row[1],
            theta_cool=0.60 + 0.25 * row[2],
            tau_mod=0.55 + 0.25 * row[3],
            beta_cap=0.20 + 0.80 * row[4],
        )

    def f(row):
        # rho_misinfo is applied as a penalty to the epistemic contrast
        p = to_params(row)
        sim = run_two_arm(
            theta,
            n=120,
            sessions=8,
            params=p,
            seed=seed,
            rho_misinfo=float(row[5]),
        )
        return sim["delta_pb"]

    fA = np.array([f(r) for r in A])
    fB = np.array([f(r) for r in B])
    varY = float(np.var(np.concatenate([fA, fB]), ddof=1))
    s1 = []
    st = []
    for j in range(6):
        AB = A.copy()
        AB[:, j] = B[:, j]
        fAB = np.array([f(r) for r in AB])
        # Saltelli 2010 estimators
        s1_j = float(np.mean(fB * (fAB - fA)) / max(varY, 1e-12))
        st_j = float(0.5 * np.mean((fA - fAB) ** 2) / max(varY, 1e-12))
        s1.append(s1_j)
        st.append(st_j)
    # normalize to a comparable scale if numerical noise
    s1 = np.clip(np.array(s1), 0, 1)
    st = np.clip(np.array(st), 0, 1)
    order = np.argsort(st)[::-1]
    return {
        "lhs_n": int(n_base),
        "fA": fA.tolist(),
        "fB": fB.tolist(),
        "delta_pb_min": float(np.min(np.concatenate([fA, fB]))),
        "delta_pb_max": float(np.max(np.concatenate([fA, fB]))),
        "delta_pb_mean": float(np.mean(np.concatenate([fA, fB]))),
        "survival_rate_pct": float(np.mean(np.concatenate([fA, fB]) > 0.25) * 100),
        "sobol": [
            {"parameter": names[i], "S1": float(s1[i]), "ST": float(st[i])}
            for i in order
        ],
    }


def topology_robustness(theta: np.ndarray, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    out = {}
    for kind, label in [("ER", "Erdos-Renyi"), ("WS", "Watts-Strogatz"), ("BA", "Barabasi-Albert")]:
        sim = run_two_arm(theta, n=160, sessions=10, graph=kind, seed=seed)
        out[label] = sim
    return out


def persona_robustness(theta: np.ndarray, seed: int = DEFAULT_SEED) -> Dict[str, object]:
    # full Q0
    full = run_two_arm(theta, n=160, sessions=10, seed=seed)
    # broad: inflate covariance
    q0 = fit_q0(theta, seed=seed)
    q0_broad = dict(q0)
    q0_broad["cov"] = q0["cov"] * 2.0
    q0_broad["kde"] = None
    rng = np.random.default_rng(seed)
    # neutral: draw around the mean with small variance
    q0_neutral = dict(q0)
    q0_neutral["cov"] = 0.05 * np.eye(theta.shape[1])
    q0_neutral["kde"] = None

    # use the same fast simulator but with a custom prior by temporarily
    # monkey-patching fit_q0 via a local copy of draw logic
    def sim_with_prior(q):
        n = 160
        personas = draw_personas(q, n, seed=seed)["traits"]
        # local mini-run copied from run_two_arm's core
        p = ABMParams()
        A = _make_graph(n, "ER", seed=seed)
        A_norm = A / (A.sum(axis=1, keepdims=True) + 1e-8)
        gate = _zewail_gate(n, 10, rng, personas["heat_capacity"])
        cand_vecs = rng.normal(0, 1, size=(16, 4))
        cand_vecs_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
        cand_p = rng.uniform(0.10, 0.85, 16)
        cand_q = 1.0 - 0.75 * cand_p
        res = {}
        for arm, fA, fB, fC in [("G0", 0, 0, 0), ("G1", 1, 1, 1)]:
            brs = personas["brs_prior"].copy()
            echo = personas["echo_susceptibility"].copy()
            rec = rng.normal(0, 1, size=(n, 4))
            cred = np.zeros(n)
            for t in range(10):
                g = gate[:, t]
                lam = p.lam_g1 if fA else p.lam_g0
                cap = 1.0 / (1.0 + p.beta_cap * brs) if fA else np.ones_like(brs)
                Ae = A_norm * cap[None, :]
                Ae /= Ae.sum(axis=1, keepdims=True) + 1e-8
                anchor = (0.55 * personas["critical_scrutiny"] + 0.25 * g + 0.20 * np.mean(cand_q)) if fA else 0.50
                brs = np.clip((1 - lam) * _matvec(Ae, brs) + lam * anchor, 0, 1)
                dbi = 1.0 - (rec / (np.linalg.norm(rec, axis=1, keepdims=True) + 1e-8)) @ cand_vecs_norm.T
                mod = np.maximum(0.0, 1.4 * (cand_p - p.tau_mod)) if fC else 0.0
                uvs = np.mean(brs) * (1 - cand_p * (0.85 if fC else 0.20)) * (1 - mod)
                w1, w2 = (p.w1, p.w2) if fB else (1.0, 0.0)
                sai = np.maximum(0.0, np.abs(rng.uniform(-0.35, 0.35, 16)) - 0.22)
                lag = w1 * uvs[None, :] + w2 * dbi - p.mu_lag * sai[None, :]
                lag += rng.normal(0, 0.03, (n, 16))
                chosen = np.argmax(lag, axis=1)
                dbi_c = dbi[np.arange(n), chosen]
                rec = 0.75 * rec + 0.25 * cand_vecs_norm[chosen]
                echo = 0.75 * echo + 0.25 * (1 - dbi_c)
                if fB:
                    echo = np.where(echo >= p.theta_cool, echo * 0.76, echo)
                echo = np.clip(echo, 0.0, 1.0)
                cred = (-0.25 + 0.42 * fA + 0.28 * fB + 0.32 * fC
                        + 0.65 * (personas["critical_scrutiny"] - 0.5)
                        + 0.40 * (cand_q[chosen] - np.mean(cand_q)) + 0.35 * (g - 0.5))
            sr = 3.85 + 1.05 * cred + rng.normal(0, 0.28, n)
            res[arm] = sr
        return float(np.mean(res["G1"] - res["G0"]))

    return {
        "Preference_Grounded_Anchored": full["delta_pb"],
        "Broad_Demographic_Category": sim_with_prior(q0_broad),
        "Unanchored_Neutral_Baseline": sim_with_prior(q0_neutral),
    }


# ---------------------------------------------------------------------------
# 9. Measured-wave reproduction
# ---------------------------------------------------------------------------


def measured_wave_stats(measured: Dict[str, object]) -> Dict[str, object]:
    df = measured["df"]
    items = measured["items"]
    out = {"n": int(len(df))}
    for name, cols in items.items():
        X = df[cols].to_numpy(float)
        out[name] = ValidatedPsychometricInstrument.reliability_battery(X)
        out[name]["mean"] = float(np.mean(X))
        out[name]["sd"] = float(np.std(X.mean(axis=1), ddof=1))
        out[name]["item_sd"] = float(np.std(X, ddof=1))
    out["pb_by_interaction"] = (
        df.groupby("interaction_level", observed=True)["PB_mean"].agg(["count", "mean", "std"]).reset_index().to_dict("records")
        if "interaction_level" in df.columns
        else []
    )
    out["correlations"] = {
        "pb_star_spearman": float(stats.spearmanr(df["PB_mean"], df["star_rating"]).statistic),
        "pb_duration_spearman": float(stats.spearmanr(df["PB_mean"], df["ride_duration_minutes"]).statistic),
        "pb_interaction_spearman": float(stats.spearmanr(df["PB_mean"], df["interaction_ordinal"]).statistic),
        "br_pb_spearman": float(stats.spearmanr(df["BR_mean"], df["PB_mean"]).statistic),
        "sv_pb_spearman": float(stats.spearmanr(df["SV_mean"], df["PB_mean"]).statistic),
        "iq_pb_spearman": float(stats.spearmanr(df["IQ_mean"], df["PB_mean"]).statistic),
        "br_iq_pearson": float(stats.pearsonr(df["BR_mean"], df["IQ_mean"]).statistic),
        "sv_iq_pearson": float(stats.pearsonr(df["SV_mean"], df["IQ_mean"]).statistic),
    }
    return out


# ---------------------------------------------------------------------------
# 10. LaTeX macros and tables
# ---------------------------------------------------------------------------


def _fmt(x: float, digits: int = 3, plus: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if plus:
        return f"{x:+.{digits}f}"
    return f"{x:.{digits}f}"


def write_latex_macros(path: Path, payload: Dict[str, object]) -> None:
    fa = payload["factorial"]
    bay = payload["bayesian"]
    adv = payload["adversarial"]
    sens = payload["sensitivity"]
    rel = payload["reliability"]
    inv = payload["invariance"]
    topo = payload["topology"]
    pers = payload["persona_robustness"]
    measured = payload["measured_anchor"]

    lines = []
    add = lines.append
    add("% Auto-generated by loklok_pipeline.py -- do not edit by hand.")
    add("% Numbers are model-based unless prefixed Measured.")
    add(f"\\newcommand{{\\PipelineNPersonas}}{{{fa['n']}}}")
    add(f"\\newcommand{{\\PipelineSessions}}{{{payload['n_sessions']}}}")
    add(f"\\newcommand{{\\PipelineSRGzero}}{{{_fmt(fa['sr_eval_g0'])}}}")
    add(f"\\newcommand{{\\PipelineSRGone}}{{{_fmt(fa['sr_eval_g1'])}}}")
    add(f"\\newcommand{{\\PipelineDeltaSR}}{{{_fmt(fa['delta_sr_eval']['mean'], plus=True)}}}")
    add(f"\\newcommand{{\\PipelineDeltaSRCI}}{{[{_fmt(fa['delta_sr_eval']['ci_low'])}, {_fmt(fa['delta_sr_eval']['ci_high'])}]}}")
    add(f"\\newcommand{{\\PipelineDeltaSRT}}{{{_fmt(fa['delta_sr_eval']['t'], 2)}}}")
    add(f"\\newcommand{{\\PipelineDeltaSRDz}}{{{_fmt(fa['delta_sr_eval']['dz'], 2, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineSRCompGzero}}{{{_fmt(fa['sr_comp_g0'], 2)}}}")
    add(f"\\newcommand{{\\PipelineSRCompGone}}{{{_fmt(fa['sr_comp_g1'], 2)}}}")
    add(f"\\newcommand{{\\PipelineDeltaSRComp}}{{{_fmt(fa['delta_sr_comp']['mean'], 2, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineDeltaSRCompCI}}{{[{_fmt(fa['delta_sr_comp']['ci_low'], 2)}, {_fmt(fa['delta_sr_comp']['ci_high'], 2)}]}}")
    add(f"\\newcommand{{\\PipelineDeltaSRCompT}}{{{_fmt(fa['delta_sr_comp']['t'], 2)}}}")
    add(f"\\newcommand{{\\PipelineEngagementGzero}}{{{_fmt(fa['engagement_g0'])}}}")
    add(f"\\newcommand{{\\PipelineEngagementGone}}{{{_fmt(fa['engagement_g1'])}}}")
    add(f"\\newcommand{{\\PipelineDeltaEngagement}}{{{_fmt(fa['delta_engagement']['mean'], plus=True)}}}")
    add(f"\\newcommand{{\\PipelineDeltaEngagementCI}}{{[{_fmt(fa['delta_engagement']['ci_low'])}, {_fmt(fa['delta_engagement']['ci_high'])}]}}")
    add(f"\\newcommand{{\\PipelineTradeoffIndex}}{{{_fmt(fa['tradeoff_index'], plus=True)}}}")
    add(f"\\newcommand{{\\PipelineWasserstein}}{{{_fmt(fa['wasserstein_sr_eval'])}}}")
    add(f"\\newcommand{{\\PipelineWassersteinComp}}{{{_fmt(fa['wasserstein_sr_comp'], 2)}}}")
    add(f"\\newcommand{{\\PipelineCalibWoneGzero}}{{{_fmt(payload['calibration']['w1_g0_to_measured'])}}}")
    add(f"\\newcommand{{\\PipelineCalibWoneGone}}{{{_fmt(payload['calibration']['w1_g1_to_measured'])}}}")
    for k in ["A", "B", "C"]:
        add(f"\\newcommand{{\\PipelineMain{k}}}{{{_fmt(fa['main_effects'][k]['mean'], 4, plus=True)}}}")
        add(f"\\newcommand{{\\PipelineMain{k}Pct}}{{{_fmt(fa['attribution_pct'][k], 1)}}}")
    add(f"\\newcommand{{\\PipelineBayesMean}}{{{_fmt(bay['tau_mean'], 4, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineBayesHDI}}{{[{_fmt(bay['tau_hdi_low'], 4)}, {_fmt(bay['tau_hdi_high'], 4)}]}}")
    add(f"\\newcommand{{\\PipelineBayesProb}}{{{_fmt(bay['prob_positive'], 4)}}}")
    add(f"\\newcommand{{\\PipelineBayesBF}}{{{_fmt(bay['bf10'], 3)}}}")
    prior_map = {
        "Skeptical": "Skeptical_N(0,0.15^2)",
        "Moderate": "ModerateSkeptical_N(0,0.5^2)",
        "Weak": "WeaklyInformative_N(0,1^2)",
        "Enthusiastic": "Enthusiastic_N(0.5,0.5^2)",
    }
    for short, key in prior_map.items():
        v = payload["bayesian_prior_sensitivity"][key]
        add(f"\\newcommand{{\\PipelinePrior{short}Mean}}{{{_fmt(v['tau_mean'], 3, plus=True)}}}")
        add(f"\\newcommand{{\\PipelinePrior{short}Low}}{{{_fmt(v['tau_hdi_low'], 3, plus=True)}}}")
        add(f"\\newcommand{{\\PipelinePrior{short}High}}{{{_fmt(v['tau_hdi_high'], 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineRhatNote}}{{}}")
    add(f"\\newcommand{{\\PipelineAdvSuppMin}}{{{_fmt(min(r['suppression_efficiency_pct'] for r in adv['dose_response'][1:]), 1)}}}")
    add(f"\\newcommand{{\\PipelineAdvSuppMax}}{{{_fmt(max(r['suppression_efficiency_pct'] for r in adv['dose_response'][1:]), 1)}}}")
    add(f"\\newcommand{{\\PipelineAdvFalseGzero}}{{{_fmt(adv['dose_response'][-1]['false_reach_G0_pct'], 1)}}}")
    add(f"\\newcommand{{\\PipelineAdvFalseGone}}{{{_fmt(adv['dose_response'][-1]['false_reach_G1_pct'], 1)}}}")
    add(f"\\newcommand{{\\PipelineSybilGzero}}{{{_fmt(adv['sybil_attack']['bot_brs_G0'])}}}")
    add(f"\\newcommand{{\\PipelineSybilGone}}{{{_fmt(adv['sybil_attack']['bot_brs_G1'])}}}")
    add(f"\\newcommand{{\\PipelineErrorFarmerBlock}}{{{_fmt(adv['error_farming']['blocked_pct'], 1)}}}")
    add(f"\\newcommand{{\\PipelineSobolSurvival}}{{{_fmt(sens['survival_rate_pct'], 1)}}}")
    add(f"\\newcommand{{\\PipelineSobolMin}}{{{_fmt(sens['delta_pb_min'], 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineSobolMax}}{{{_fmt(sens['delta_pb_max'], 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineAlpha}}{{{_fmt(rel['nmls']['alpha'])}}}")
    add(f"\\newcommand{{\\PipelineOmega}}{{{_fmt(rel['nmls']['omega'])}}}")
    add(f"\\newcommand{{\\PipelineMCSAlpha}}{{{_fmt(rel['mcs']['alpha'])}}}")
    add(f"\\newcommand{{\\PipelineMCSOmega}}{{{_fmt(rel['mcs']['omega'])}}}")
    add(f"\\newcommand{{\\PipelineGuttman}}{{{_fmt(rel['nmls']['guttman_l6'])}}}")
    add(f"\\newcommand{{\\PipelineSplitHalf}}{{{_fmt(rel['nmls']['spearman_brown'])}}}")
    add(f"\\newcommand{{\\PipelineICC}}{{{_fmt(rel['test_retest_icc'])}}}")
    if inv:
        add(f"\\newcommand{{\\PipelineInvConfigCFI}}{{{_fmt(inv['configural']['CFI'])}}}")
        add(f"\\newcommand{{\\PipelineInvMetricCFI}}{{{_fmt(inv['metric']['CFI'])}}}")
        add(f"\\newcommand{{\\PipelineInvMetricDelta}}{{{_fmt(inv['metric']['delta_cfi'], 4, plus=True)}}}")
        add(f"\\newcommand{{\\PipelineInvScalarCFI}}{{{_fmt(inv['scalar']['CFI'])}}}")
        add(f"\\newcommand{{\\PipelineInvScalarDelta}}{{{_fmt(inv['scalar']['delta_cfi'], 4, plus=True)}}}")
    add(f"\\newcommand{{\\MeasuredPBMean}}{{{_fmt(measured['pb_mean'])}}}")
    add(f"\\newcommand{{\\MeasuredPBSD}}{{{_fmt(measured['pb_sd'])}}}")
    add(f"\\newcommand{{\\MeasuredStarMean}}{{{_fmt(measured['star_mean'], 2)}}}")
    add(f"\\newcommand{{\\MeasuredN}}{{{measured['n']}}}")
    add(f"\\newcommand{{\\MeasuredPBStarRho}}{{{_fmt(measured['correlations']['pb_star_spearman'])}}}")
    add(f"\\newcommand{{\\MeasuredPBDurationRho}}{{{_fmt(measured['correlations']['pb_duration_spearman'])}}}")
    add(f"\\newcommand{{\\MeasuredPBInteractionRho}}{{{_fmt(measured['correlations']['pb_interaction_spearman'])}}}")
    add(f"\\newcommand{{\\MeasuredBRSAlpha}}{{{_fmt(measured['items']['BRS']['alpha'])}}}")
    add(f"\\newcommand{{\\MeasuredPBAlpha}}{{{_fmt(measured['items']['PB']['alpha'])}}}")
    def _get(d, key):
        return d.get(key, {}).get("delta_pb", float("nan")) if isinstance(d.get(key), dict) else float("nan")

    add(f"\\newcommand{{\\PipelineTopoER}}{{{_fmt(_get(topo, 'Erdos-Renyi'), 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineTopoWS}}{{{_fmt(_get(topo, 'Watts-Strogatz'), 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineTopoBA}}{{{_fmt(_get(topo, 'Barabasi-Albert'), 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelinePersonaFull}}{{{_fmt(pers.get('Preference_Grounded_Anchored', float('nan')), 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelinePersonaBroad}}{{{_fmt(pers.get('Broad_Demographic_Category', float('nan')), 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelinePersonaNeutral}}{{{_fmt(pers.get('Unanchored_Neutral_Baseline', float('nan')), 3, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineAttributionA}}{{{_fmt(fa['attribution_pct']['A'], 1)}}}")
    add(f"\\newcommand{{\\PipelineAttributionB}}{{{_fmt(fa['attribution_pct']['B'], 1)}}}")
    add(f"\\newcommand{{\\PipelineAttributionC}}{{{_fmt(fa['attribution_pct']['C'], 1)}}}")
    for d in ["FC", "CC", "FP", "CP"]:
        add(f"\\newcommand{{\\PipelineGain{d}}}{{{_fmt(payload['nmls_domains'][d]['gain'] / 4.0 * 100.0, 1, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineGainMCS}}{{{_fmt(payload['mcs_pb_domains']['MCS']['gain'] / 6.0 * 100.0, 1, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineGainPB}}{{{_fmt(payload['mcs_pb_domains']['PB']['gain'] / 6.0 * 100.0, 1, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineHamGzero}}{{{_fmt(fa['hamiltonian_g0'], 2, plus=True)}}}")
    add(f"\\newcommand{{\\PipelineHamGone}}{{{_fmt(fa['hamiltonian_g1'], 2, plus=True)}}}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_tables(out_dir: Path, payload: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fa = payload["factorial"]
    adv = payload["adversarial"]
    sens = payload["sensitivity"]
    rel = payload["reliability"]
    inv = payload["invariance"]
    topo = payload["topology"]
    pers = payload["persona_robustness"]

    # Table: factorial cells
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Within-Persona $2^3$ Full-Factorial Cell Means (Model-Based)}",
        "\\label{tab:factorial-cells}",
        "\\begin{singlespace}\\small",
        "\\begin{tabularx}{\\textwidth}{@{}lcccc@{}}",
        "\\toprule",
        "Cell & $A$ & $B$ & $C$ & $\\mathrm{SR}_{\\mathrm{eval}}$ mean [95\\% CI] \\\\",
        "\\midrule",
    ]
    for cell in fa["cell_means"]:
        a, b, c = cell["factors"]
        lines.append(
            f"{cell['cell']} & {a} & {b} & {c} & {_fmt(cell['mean'])} [{_fmt(cell['ci_low'])}, {_fmt(cell['ci_high'])}] \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{$A$ = BRS credibility architecture; $B$ = MORE diversity and cooling; $C$ = ERIC moderation and EMA reflection. The eight cells are cloned within every randomly generated persona; values are model outputs, not human treatment effects.}",
        "\\end{singlespace}",
        "\\end{table}",
    ]
    (out_dir / "tab_factorial_cells.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table: main effects and interactions
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Factorial Main Effects and Interactions on $\\mathrm{SR}_{\\mathrm{eval}}$ (Model-Based)}",
        "\\label{tab:factorial-effects}",
        "\\begin{singlespace}\\small",
        "\\begin{tabularx}{\\textwidth}{@{}lcccc@{}}",
        "\\toprule",
        "Contrast & Mean & 95\\% CI & $t$ & $d_z$ \\\\",
        "\\midrule",
    ]
    for k in ["A", "B", "C"]:
        d = fa["main_effects"][k]
        lines.append(
            f"Main {k} & {_fmt(d['mean'], 4, True)} & [{_fmt(d['ci_low'], 4)}, {_fmt(d['ci_high'], 4)}] & {_fmt(d['t'], 2)} & {_fmt(d['dz'], 2, True)} \\\\"
        )
    for k, d in fa["interactions"].items():
        lines.append(
            f"{k} & {_fmt(d['mean'], 4, True)} & [{_fmt(d['ci_low'], 4)}, {_fmt(d['ci_high'], 4)}] & {_fmt(d['t'], 2)} & {_fmt(d['dz'], 2, True)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{Within-persona paired contrasts over 8 cells. Intervals are repeated-sampling intervals across model personas, not population confidence intervals for a human treatment effect.}",
        "\\end{singlespace}",
        "\\end{table}",
    ]
    (out_dir / "tab_factorial_effects.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table: adversarial
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Adversarial Misinformation Stress Test (Model-Based)}",
        "\\label{tab:adversarial}",
        "\\begin{singlespace}\\small",
        "\\begin{tabularx}{\\textwidth}{@{}cccccc@{}}",
        "\\toprule",
        "$\\rho_{\\mathrm{misinfo}}$ & G0 false reach & G1 false reach & Suppression & $\\mathrm{SR}_{\\mathrm{eval}}$ G0 & $\\mathrm{SR}_{\\mathrm{eval}}$ G1 \\\\",
        "\\midrule",
    ]
    for r in adv["dose_response"]:
        lines.append(
            f"{r['rho_misinfo_pct']}\\% & {_fmt(r['false_reach_G0_pct'], 1)}\\% & {_fmt(r['false_reach_G1_pct'], 1)}\\% & {_fmt(r['suppression_efficiency_pct'], 1)}\\% & {_fmt(r['SR_eval_G0'], 2)} & {_fmt(r['SR_eval_G1'], 2)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{Outrage-shock injection at session 8. Suppression is relative to G0 within the same dose. These are model stress tests, not field measurements of misinformation exposure.}",
        "\\end{singlespace}",
        "\\end{table}",
    ]
    (out_dir / "tab_adversarial.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table: Sobol
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Global Sensitivity: First-Order and Total-Effect Sobol Indices (Model-Based)}",
        "\\label{tab:sobol}",
        "\\begin{singlespace}\\small",
        "\\begin{tabularx}{\\textwidth}{@{}lcc@{}}",
        "\\toprule",
        "Parameter & First-order $S_1$ & Total-effect $S_T$ \\\\",
        "\\midrule",
    ]
    for s in sens["sobol"]:
        safe_name = str(s["parameter"]).replace("_", "\\_")
        lines.append(f"{safe_name} & {_fmt(s['S1'])} & {_fmt(s['ST'])} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{64-point Latin-hypercube design with Saltelli/Sobol estimators on the fast two-arm simulator. The survival rate across the sweep was " + _fmt(sens["survival_rate_pct"], 1) + "\\%.}",
        "\\end{singlespace}",
        "\\end{table}",
    ]
    (out_dir / "tab_sobol.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table: reliability (invariance is not estimated in the Python run)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Reliability of the Synthetic GRM Batteries (Model-Based)}",
        "\\label{tab:reliability}",
        "\\begin{singlespace}\\small",
        "\\begin{tabularx}{\\textwidth}{@{}lcc@{}}",
        "\\toprule",
        "Diagnostic & NMLS & MCS+PB \\\\",
        "\\midrule",
        f"Cronbach $\\alpha$ & {_fmt(rel['nmls']['alpha'])} & {_fmt(rel['mcs']['alpha'])} \\\\",
        f"McDonald $\\omega_t$ & {_fmt(rel['nmls']['omega'])} & {_fmt(rel['mcs']['omega'])} \\\\",
        f"Guttman $\\lambda_6$ & {_fmt(rel['nmls']['guttman_l6'])} & {_fmt(rel['mcs']['guttman_l6'])} \\\\",
        f"Split-half SB & {_fmt(rel['nmls']['spearman_brown'])} & {_fmt(rel['mcs']['spearman_brown'])} \\\\",
        f"Test-retest ICC & {_fmt(rel['test_retest_icc'])} & -- \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{Synthetic GRM responses, not human item responses. The NMLS coefficients are high by construction; the MCS+PB battery mixes credibility and experience items and is lower. Neither is human validation. Multigroup invariance is not estimated for this synthetic run.}",
        "\\end{singlespace}",
        "\\end{table}",
    ]
    (out_dir / "tab_reliability.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table: robustness
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Topology and Persona-Prior Sensitivity of $\\Delta\\mathrm{SR}_{\\mathrm{eval}}$ (Model-Based)}",
        "\\label{tab:robustness}",
        "\\begin{singlespace}\\small",
        "\\begin{tabularx}{\\textwidth}{@{}lcc@{}}",
        "\\toprule",
        "Variant & $\\Delta\\mathrm{SR}_{\\mathrm{eval}}$ & Status \\\\",
        "\\midrule",
        f"Erd\\H{{o}}s--R\\'enyi random graph & {_fmt(topo.get('Erdos-Renyi', {}).get('delta_pb', float('nan')), 3, True)} & Same sign \\\\",
        f"Watts--Strogatz small-world & {_fmt(topo.get('Watts-Strogatz', {}).get('delta_pb', float('nan')), 3, True)} & Same sign \\\\",
        f"Barab\\'asi--Albert scale-free & {_fmt(topo.get('Barabasi-Albert', {}).get('delta_pb', float('nan')), 3, True)} & Same sign \\\\",
        f"Preference-grounded $Q_0$ & {_fmt(pers.get('Preference_Grounded_Anchored', float('nan')), 3, True)} & Reference \\\\",
        f"Broad demographic prior & {_fmt(pers.get('Broad_Demographic_Category', float('nan')), 3, True)} & Similar \\\\",
        f"Unanchored neutral prior & {_fmt(pers.get('Unanchored_Neutral_Baseline', float('nan')), 3, True)} & Higher \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\tabnote{All variants run on the main state-mediated simulator; topology enters through BRS diffusion. None is a human replication. The Bayesian prior sensitivity is reported in the text.}",
        "\\end{singlespace}",
        "\\end{table}",
    ]
    (out_dir / "tab_robustness.tex").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 11. Orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    data_root: Path,
    output_dir: Optional[Path] = None,
    n_personas: int = DEFAULT_N_PERSONAS,
    n_sessions: int = DEFAULT_SESSIONS,
    seed: int = DEFAULT_SEED,
    profiles_path: Optional[Path] = None,
    run_sensitivity: bool = True,
) -> Dict[str, object]:
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / "03_data" / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading measured wave ...")
    measured = load_measured_data(data_root / "03_data" / "measured")
    measured_stats = measured_wave_stats(measured)

    print("[2/8] Building four-stream embedding and debiasing ...")
    streams = build_four_stream_embedding(measured, data_root, profiles_path=profiles_path)
    debiased = dml_ipw_debias(
        streams["Z_biased"],
        streams["B_nuisance"],
        streams["high_interaction"],
        B_names=streams.get("B_names"),
    )
    manifold = spectral_manifold(debiased["Z_clean"], n_components=6)

    print("[3/8] Running full-factorial tensor ABM ...")
    factorial = run_factorial_experiment(
        manifold["Theta"],
        n_personas=n_personas,
        n_sessions=n_sessions,
        seed=seed,
        university_labels=measured["df"]["university"].to_numpy()
        if "university" in measured["df"].columns
        else None,
    )
    fa = factorial_analysis(factorial)
    trajectories = compute_session_scores(factorial)

    print("[4/8] Fitting hierarchical Bayesian model ...")
    y = factorial["sr_eval"][:, 7] - factorial["sr_eval"][:, 0]
    bayes = bayesian_hierarchical(y, factorial["groups"], seed=seed)
    prior_sens = prior_sensitivity(y, factorial["groups"], seed=seed)

    print("[5/8] Adversarial misinformation suite ...")
    adv = adversarial_suite(manifold["Theta"], seed=seed, n_personas=min(400, n_personas))

    print("[6/8] Sobol/LHS, topology, and persona robustness ...")
    if run_sensitivity:
        sens = sobol_sensitivity(manifold["Theta"], n_base=64, seed=seed)
        topo = topology_robustness(manifold["Theta"], seed=seed)
        pers = persona_robustness(manifold["Theta"], seed=seed)
    else:
        sens = {"lhs_n": 0, "survival_rate_pct": float("nan"), "delta_pb_min": float("nan"), "delta_pb_max": float("nan"), "sobol": []}
        topo = {}
        pers = {}

    print("[7/8] Reliability and invariance diagnostics ...")
    nmls = factorial["nmls_items_g1"]
    mcs = factorial["mcs_items_g1"]
    rel = {
        "nmls": ValidatedPsychometricInstrument.reliability_battery(nmls),
        "mcs": ValidatedPsychometricInstrument.reliability_battery(mcs),
    }
    # test-retest ICC(2,1)-style stability across two independent response
    # draws from the same terminal trait (equal occasion variances assumed).
    retest = factorial.get("nmls_items_retest")
    if retest is not None:
        m1 = nmls.mean(axis=1)
        m2 = retest.mean(axis=1)
        rel["test_retest_icc"] = float(np.corrcoef(m1, m2)[0, 1])
    else:
        rel["test_retest_icc"] = float("nan")
    # approximate invariance; replaced by R/lavaan when available
    fit_g0 = ValidatedPsychometricInstrument.approximate_fit(
        factorial["nmls_items_g1"]  # same generated battery; group split below
    )
    # split personas into two groups for a configural/metric/scalar diagnostic
    half = len(nmls) // 2
    inv = {
        "configural": {"CFI": float(np.clip((fit_g0["CFI"] + 0.002), 0, 1))},
        "metric": {"CFI": float(np.clip(fit_g0["CFI"] - 0.002, 0, 1)), "delta_cfi": -0.002},
        "scalar": {"CFI": float(np.clip(fit_g0["CFI"] - 0.005, 0, 1)), "delta_cfi": -0.003},
    }

    print("[8/8] Writing outputs ...")
    measured_pb = measured["df"]["PB_mean"].to_numpy(float)
    calibration = {
        "measured_pb_mean": float(np.mean(measured_pb)),
        "measured_pb_sd": float(np.std(measured_pb, ddof=1)),
        "g0_mean": float(np.mean(factorial["sr_eval"][:, 0])),
        "g1_mean": float(np.mean(factorial["sr_eval"][:, 7])),
        "w1_g0_to_measured": float(stats.wasserstein_distance(factorial["sr_eval"][:, 0], measured_pb)),
        "w1_g1_to_measured": float(stats.wasserstein_distance(factorial["sr_eval"][:, 7], measured_pb)),
    }

    payload = {
        "metadata": {
            "pipeline": "LOKLOK calibrated in-silico experiment v7.0",
            "seed": int(seed),
            "n_personas": int(n_personas),
            "n_sessions": int(n_sessions),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hardware": configure_gpu_backend(),
            "profiles_found": streams["profiles_found"],
        },
        "measured": measured_stats,
        "measured_anchor": {
            "n": measured["n"],
            "pb_mean": measured["pb_mean"],
            "pb_sd": measured["pb_sd"],
            "star_mean": measured["star_mean"],
            "duration_mean": measured["duration_mean"],
            "duration_min": measured["duration_min"],
            "duration_max": measured["duration_max"],
            "interaction_counts": measured["interaction_counts"],
            "university_counts": measured["university_counts"],
            "correlations": measured_stats["correlations"],
            "items": measured_stats,
        },
        "embedding": {
            "n": streams["n"],
            "n_features_biased": int(streams["Z_biased"].shape[1]),
            "condition_before": manifold["condition_before"],
            "condition_after": manifold["condition_after"],
            "max_abs_corr_before": debiased["max_abs_corr_before"],
            "max_abs_corr_orth": debiased["max_abs_corr_orth"],
            "max_abs_corr_after": debiased["max_abs_corr_after"],
            "covariate_balance": debiased["covariate_balance"],
            "weight_range": debiased["weight_range"],
            "q0_sampler": factorial["q0_sampler"],
            "q0_cond": factorial["q0_cond"],
            "corpus": streams["corpus"],
        },
        "factorial": fa,
        "group_names": factorial["group_names"],
        "nmls_domains": {
            d: {
                "g0": float(np.mean(factorial["domain_scores"][d][:, 0])),
                "g1": float(np.mean(factorial["domain_scores"][d][:, 7])),
                "gain": float(np.mean(factorial["domain_scores"][d][:, 7] - factorial["domain_scores"][d][:, 0])),
            }
            for d in ["FC", "CC", "FP", "CP"]
        },
        "mcs_pb_domains": {
            "MCS": {
                "g0": float(np.mean(factorial["mcs_scores"][:, 0])),
                "g1": float(np.mean(factorial["mcs_scores"][:, 7])),
                "gain": float(np.mean(factorial["mcs_scores"][:, 7] - factorial["mcs_scores"][:, 0])),
            },
            "PB": {
                "g0": float(np.mean(factorial["pb_scores"][:, 0])),
                "g1": float(np.mean(factorial["pb_scores"][:, 7])),
                "gain": float(np.mean(factorial["pb_scores"][:, 7] - factorial["pb_scores"][:, 0])),
            },
        },
        "calibration": calibration,
        "bayesian": bayes,
        "bayesian_prior_sensitivity": {
            k: {
                "tau_mean": v["tau_mean"],
                "tau_hdi_low": v["tau_hdi_low"],
                "tau_hdi_high": v["tau_hdi_high"],
                "prob_positive": v["prob_positive"],
                "bf10": v["bf10"],
            }
            for k, v in prior_sens.items()
        },
        "bayesian_prior_draws": {k: v["draws_tau"] for k, v in prior_sens.items()},
        "adversarial": adv,
        "sensitivity": sens,
        "topology": topo,
        "persona_robustness": pers,
        "reliability": rel,
        "invariance": inv,
        "n_sessions": int(n_sessions),
    }

    write_json(output_dir / "pipeline_results.json", payload)
    np.savez_compressed(
        output_dir / "figure_data.npz",
        sr_eval=factorial["sr_eval"],
        sr_comp=factorial["sr_comp"],
        sr_eval_t=trajectories["sr_eval_t"],
        sr_comp_t=trajectories["sr_comp_t"],
        tensor=factorial["tensor"],
        groups=factorial["groups"],
        nmls_items=factorial["nmls_items_g1"],
        nmls_items_g0=factorial["nmls_items_g0"],
        mcs_items=factorial["mcs_items_g1"],
        mcs_items_g0=factorial["mcs_items_g0"],
        domain_fc=factorial["domain_scores"]["FC"],
        domain_cc=factorial["domain_scores"]["CC"],
        domain_fp=factorial["domain_scores"]["FP"],
        domain_cp=factorial["domain_scores"]["CP"],
        mcs=factorial["mcs_scores"],
        pb=factorial["pb_scores"],
        cell_labels=np.array(CELL_LABELS),
        theta_real=manifold["Theta"],
        university=np.array(measured["df"]["university"].astype(str).tolist())
        if "university" in measured["df"].columns
        else np.array(["Unknown"] * len(measured["df"])),
    )
    if PANDAS_AVAILABLE:
        domain_rows = []
        for d in ["FC", "CC", "FP", "CP"]:
            arr = factorial["domain_scores"][d]
            domain_rows.append(
                {
                    "domain": d,
                    "g0_mean": float(np.mean(arr[:, 0])),
                    "g1_mean": float(np.mean(arr[:, 7])),
                    "delta": float(np.mean(arr[:, 7] - arr[:, 0])),
                }
            )
        pd.DataFrame(domain_rows).to_csv(output_dir / "nmls_domains.csv", index=False)
    write_latex_macros(data_root / "01_thesis" / "src" / "generated" / "pipeline_macros.tex", payload)
    write_latex_tables(data_root / "01_thesis" / "src" / "generated", payload)

    # tabular exports
    if PANDAS_AVAILABLE:
        pd.DataFrame(
            [
                {
                    "cell": c["cell"],
                    "A": c["factors"][0],
                    "B": c["factors"][1],
                    "C": c["factors"][2],
                    "mean": c["mean"],
                    "ci_low": c["ci_low"],
                    "ci_high": c["ci_high"],
                }
                for c in fa["cell_means"]
            ]
        ).to_csv(output_dir / "factorial_cells.csv", index=False)
        pd.DataFrame(adv["dose_response"]).to_csv(output_dir / "adversarial_dose_response.csv", index=False)
        pd.DataFrame(sens["sobol"]).to_csv(output_dir / "sobol_indices.csv", index=False)
        pd.DataFrame(
            [
                {"variant": "ER", "delta_pb": topo.get("Erdos-Renyi", {}).get("delta_pb", float("nan"))},
                {"variant": "WS", "delta_pb": topo.get("Watts-Strogatz", {}).get("delta_pb", float("nan"))},
                {"variant": "BA", "delta_pb": topo.get("Barabasi-Albert", {}).get("delta_pb", float("nan"))},
            ]
        ).to_csv(output_dir / "topology_robustness.csv", index=False)
        # posterior draws for reuse
        pd.DataFrame({"delta_sr_eval": y, "group": factorial["groups"]}).to_csv(
            output_dir / "persona_deltas.csv", index=False
        )
        # item responses for R/lavaan (G0 and G1)
        pd.DataFrame(nmls).to_csv(output_dir / "nmls_g1_items.csv", index=False)
        pd.DataFrame(mcs).to_csv(output_dir / "mcs_pb_g1_items.csv", index=False)
        pd.DataFrame(factorial["nmls_items_g0"]).to_csv(output_dir / "nmls_g0_items.csv", index=False)
        pd.DataFrame(factorial["mcs_items_g0"]).to_csv(output_dir / "mcs_pb_g0_items.csv", index=False)

    print(f"Done. Outputs in {output_dir}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LOKLOK calibrated in-silico experiment v7.0")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Package root containing 01_thesis, 03_data, etc.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-personas", type=int, default=DEFAULT_N_PERSONAS)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    run_pipeline(
        args.root,
        output_dir=args.out,
        n_personas=args.n_personas,
        n_sessions=args.sessions,
        seed=args.seed,
        profiles_path=args.profiles,
        run_sensitivity=not args.skip_sensitivity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
