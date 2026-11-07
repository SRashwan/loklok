"""Per-code human reliability and leave-one-out ceiling for the LOKLOK codebook.

Inputs
------
03_data/model_outputs/qual_validation/human_coder_labels.csv
    Long format: unit_id, coder, source, role, C01-C12, I01-I13.

Outputs
-------
03_data/model_outputs/qual_validation/per_code_reliability.csv
03_data/model_outputs/qual_validation/loo_human_reliability.csv
03_data/model_outputs/qual_validation/per_code_reliability_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
QUAL = ROOT / "03_data" / "model_outputs" / "qual_validation"
LABELS = QUAL / "human_coder_labels.csv"

CODES = [f"C{i:02d}" for i in range(1, 13)] + [f"I{i:02d}" for i in range(1, 14)]
CODERS = ["Coder1_Strict", "Coder2_Contextual", "Coder3_Pragmatic"]


def safe_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(np.unique(a)) < 2 and len(np.unique(b)) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def main() -> None:
    labels = pd.read_csv(LABELS)
    mats = {}
    for coder in CODERS:
        part = labels[labels["coder"] == coder].set_index("unit_id")
        mats[coder] = part[CODES].fillna(0).astype(int)
    units = sorted(set.intersection(*(set(m.index) for m in mats.values())))
    for coder in CODERS:
        mats[coder] = mats[coder].loc[units]

    rows = []
    for code in CODES:
        a, b, c = (mats[coder][code].to_numpy() for coder in CODERS)
        majority = ((a + b + c) >= 2).astype(int)
        pairs = {
            "Coder1_vs_Coder2": safe_kappa(a, b),
            "Coder1_vs_Coder3": safe_kappa(a, c),
            "Coder2_vs_Coder3": safe_kappa(b, c),
        }
        vs_majority = {
            coder: safe_kappa(mats[coder][code].to_numpy(), majority)
            for coder in CODERS
        }
        pair_values = [v for v in pairs.values() if np.isfinite(v)]
        maj_values = [v for v in vs_majority.values() if np.isfinite(v)]
        rows.append(
            {
                "code": code,
                "human_positives": int(majority.sum()),
                "kappa_pair_12": pairs["Coder1_vs_Coder2"],
                "kappa_pair_13": pairs["Coder1_vs_Coder3"],
                "kappa_pair_23": pairs["Coder2_vs_Coder3"],
                "kappa_pair_min": min(pair_values) if pair_values else np.nan,
                "kappa_pair_median": float(np.median(pair_values)) if pair_values else np.nan,
                "kappa_pair_max": max(pair_values) if pair_values else np.nan,
                "kappa_majority_min": min(maj_values) if maj_values else np.nan,
                "kappa_majority_median": float(np.median(maj_values)) if maj_values else np.nan,
                "kappa_majority_max": max(maj_values) if maj_values else np.nan,
            }
        )
    per_code = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    per_code.to_csv(QUAL / "per_code_reliability.csv", index=False)

    loo_rows = []
    for held_out in CODERS:
        others = [coder for coder in CODERS if coder != held_out]
        ref_a = mats[others[0]][CODES].to_numpy()
        ref_b = mats[others[1]][CODES].to_numpy()
        agree = ref_a == ref_b
        y_true = ref_a[agree]
        y_pred = mats[held_out][CODES].to_numpy()[agree]
        loo_rows.append(
            {
                "coder": held_out,
                "units": len(units),
                "codes": len(CODES),
                "comparable_cells": int(agree.sum()),
                "human_positives_in_comparable_cells": int(y_true.sum()),
                "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
        )
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(QUAL / "loo_human_reliability.csv", index=False)

    estimable = per_code[np.isfinite(per_code["kappa_pair_median"])]
    below_60 = estimable[estimable["kappa_pair_median"] < 0.60]["code"].tolist()
    below_40 = estimable[estimable["kappa_pair_median"] < 0.40]["code"].tolist()
    summary = {
        "units": len(units),
        "codes": len(CODES),
        "cells": int(len(units) * len(CODES)),
        "zero_cell_share": float(
            1.0 - labels[labels["coder"] == CODERS[0]][CODES].fillna(0).to_numpy().mean()
        ),
        "pairwise_kappa_min": float(estimable["kappa_pair_min"].min()),
        "pairwise_kappa_median_min": float(estimable["kappa_pair_median"].min()),
        "pairwise_kappa_median_max": float(estimable["kappa_pair_median"].max()),
        "codes_below_60_median_pairwise": below_60,
        "codes_below_40_median_pairwise": below_40,
        "leave_one_out_micro_f1": {
            row["coder"]: round(row["micro_f1"], 4) for row in loo.to_dict("records")
        },
        "note": "C08 has no positive labels in the majority gold and is not estimable.",
    }
    (QUAL / "per_code_reliability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(per_code.round(3).to_string(index=False))
    print()
    print(loo.round(4).to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
