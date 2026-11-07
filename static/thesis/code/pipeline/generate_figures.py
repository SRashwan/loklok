"""
Generate the publication figures for the final thesis upgrade.

Reads the deterministic outputs of loklok_pipeline.py and writes 300-DPI PNG
and vector PDF versions into 04_figures/ and the thesis figures directory.

Run:
    python 02_code/pipeline/generate_figures.py --root <package-root>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loklok_pipeline as lp  # noqa: E402

NAVY = "#1f3a5f"
ORANGE = "#e07b39"
TEAL = "#2a9d8f"
GREY = "#8d99ae"
RED = "#c1121f"
GOLD = "#d4a017"
LIGHT = "#e9ecef"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def kde_curve(data: np.ndarray, grid: np.ndarray, bw: float = 0.28) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 3 or np.std(data) < 1e-9:
        return np.zeros_like(grid)
    return gaussian_kde(data, bw_method=bw)(grid)


def save(fig: plt.Figure, name: str, out_png: Path, out_pdf: Path, thesis_fig: Path) -> None:
    out_png.mkdir(parents=True, exist_ok=True)
    out_pdf.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png / f"{name}.png", dpi=300)
    fig.savefig(out_pdf / f"{name}.pdf")
    fig.savefig(thesis_fig / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}")


def load(root: Path) -> Tuple[Dict, Dict[str, np.ndarray], pd.DataFrame]:
    out = root / "03_data" / "model_outputs"
    results = json.loads((out / "pipeline_results.json").read_text(encoding="utf-8"))
    arrays = dict(np.load(out / "figure_data.npz", allow_pickle=True))
    measured = pd.read_csv(root / "03_data" / "measured" / "respondents_52.csv")
    return results, arrays, measured


# ---------------------------------------------------------------------------
# Figure 6.1 -- Student Rating distributions and calibration
# ---------------------------------------------------------------------------


def fig61(results: Dict, arrays: Dict, measured: pd.DataFrame, out_png: Path, out_pdf: Path, thesis: Path):
    sr = arrays["sr_eval"]
    comp = arrays["sr_comp"]
    g0, g1 = sr[:, 0], sr[:, 7]
    c0, c1 = comp[:, 0], comp[:, 7]
    pb_meas = measured["pb_experience_mean"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    ax = axes[0]
    x = np.linspace(2.6, 6.9, 400)
    ax.fill_between(x, kde_curve(g0, x), color=GREY, alpha=0.30, label="G0 (engagement-first)")
    ax.plot(x, kde_curve(g0, x), color=GREY, lw=1.8)
    ax.fill_between(x, kde_curve(g1, x), color=NAVY, alpha=0.25, label="G1 (epistemic-first)")
    ax.plot(x, kde_curve(g1, x), color=NAVY, lw=1.8)
    ax.plot(x, kde_curve(pb_meas, x, 0.35), color=ORANGE, lw=1.6, ls="--", label="Measured wave (N=52)")
    ax.axvline(np.mean(g0), color=GREY, lw=1, ls=":")
    ax.axvline(np.mean(g1), color=NAVY, lw=1, ls=":")
    ax.set_xlabel("Evaluative Student Rating, $\\mathrm{SR}_{\\mathrm{eval}}$ (1-7)")
    ax.set_ylabel("Density")
    ax.set_title("A. Evaluative Student Rating")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.95,
        f"$\\Delta$ = {results['factorial']['delta_sr_eval']['mean']:+.2f}\n"
        f"95% CI [{results['factorial']['delta_sr_eval']['ci_low']:.2f}, "
        f"{results['factorial']['delta_sr_eval']['ci_high']:.2f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    ax = axes[1]
    x = np.linspace(20, 95, 400)
    ax.fill_between(x, kde_curve(c0, x), color=GREY, alpha=0.30, label="G0")
    ax.plot(x, kde_curve(c0, x), color=GREY, lw=1.8)
    ax.fill_between(x, kde_curve(c1, x), color=TEAL, alpha=0.25, label="G1")
    ax.plot(x, kde_curve(c1, x), color=TEAL, lw=1.8)
    ax.axvline(np.mean(c0), color=GREY, lw=1, ls=":")
    ax.axvline(np.mean(c1), color=TEAL, lw=1, ls=":")
    ax.set_xlabel("Competence Student Rating, $\\mathrm{SR}_{\\mathrm{comp}}$ (0-100)")
    ax.set_ylabel("Density")
    ax.set_title("B. Synthetic NMLS Competence Profile")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.95,
        f"$\\Delta$ = {results['factorial']['delta_sr_comp']['mean']:+.2f} pts\n"
        f"95% CI [{results['factorial']['delta_sr_comp']['ci_low']:.2f}, "
        f"{results['factorial']['delta_sr_comp']['ci_high']:.2f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )
    fig.suptitle("Model-Based Distributional Shift in Student Ratings (500 personas $\\times$ $2^3$ cells)", y=1.02, fontsize=11)
    save(fig, "fig6_1_sr_distributions", out_png, out_pdf, thesis)

    pd.DataFrame({"SReval_G0": g0, "SReval_G1": g1, "SRcomp_G0": c0, "SRcomp_G1": c1}).to_csv(
        out_png.parent / "source_data" / "fig6_1_sr_distributions.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.2 -- Full factorial cells and attribution
# ---------------------------------------------------------------------------


def fig62(results: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    cells = results["factorial"]["cell_means"]
    labels = [c["cell"] for c in cells]
    means = np.array([c["mean"] for c in cells])
    lows = np.array([c["ci_low"] for c in cells])
    highs = np.array([c["ci_high"] for c in cells])
    colors = [GREY] + [TEAL] * 6 + [NAVY]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    yerr = np.vstack([means - lows, highs - means])
    ax.bar(range(8), means, yerr=yerr, color=colors, alpha=0.88, capsize=3, edgecolor="white")
    ax.set_xticks(range(8))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("$\\mathrm{SR}_{\\mathrm{eval}}$ (1-7)")
    ax.set_title("A. Within-Persona $2^3$ Factorial Cells")
    for i, m in enumerate(means):
        ax.text(i, m + 0.08, f"{m:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    attr = results["factorial"]["attribution_pct"]
    names = ["A: BRS", "B: MORE", "C: ERIC"]
    vals = [attr["A"], attr["B"], attr["C"]]
    ax.barh(names, vals, color=[NAVY, TEAL, ORANGE], alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel("Share of positive main-effect gain (%)")
    ax.set_title("B. Mechanism Attribution")
    ax.invert_yaxis()
    fig.suptitle("Model-Based Factorial Decomposition of Platform Impact", y=1.02, fontsize=11)
    save(fig, "fig6_2_factorial_cells", out_png, out_pdf, thesis)
    pd.DataFrame({"cell": labels, "mean": means, "ci_low": lows, "ci_high": highs}).to_csv(
        out_png.parent / "source_data" / "fig6_2_factorial_cells.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.3 -- Radar of six student-rating dimensions
# ---------------------------------------------------------------------------


def fig63(results: Dict, arrays: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    domains = {
        "FC": arrays["domain_fc"],
        "CC": arrays["domain_cc"],
        "FP": arrays["domain_fp"],
        "CP": arrays["domain_cp"],
    }
    g0 = [np.mean(domains[d][:, 0]) for d in ["FC", "CC", "FP", "CP"]]
    g1 = [np.mean(domains[d][:, 7]) for d in ["FC", "CC", "FP", "CP"]]
    g0 += [np.mean(arrays["mcs"][:, 0]), np.mean(arrays["pb"][:, 0])]
    g1 += [np.mean(arrays["mcs"][:, 7]), np.mean(arrays["pb"][:, 7])]
    g0 = np.array(g0)
    g1 = np.array(g1)
    # normalize NMLS 1-5 and MCS/PB 1-7 to a common 0-100 display
    n0 = np.concatenate([(g0[:4] - 1) / 4 * 100, (g0[4:] - 1) / 6 * 100])
    n1 = np.concatenate([(g1[:4] - 1) / 4 * 100, (g1[4:] - 1) / 6 * 100])
    labels = ["FC", "CC", "FP", "CP", "MCS", "PB"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    n0p = np.concatenate([n0, n0[:1]])
    n1p = np.concatenate([n1, n1[:1]])

    fig = plt.figure(figsize=(10.2, 4.4))
    ax = fig.add_subplot(1, 2, 1, polar=True)
    ax.plot(angles, n0p, color=GREY, lw=1.8, ls="--", label="G0")
    ax.fill(angles, n0p, color=GREY, alpha=0.12)
    ax.plot(angles, n1p, color=NAVY, lw=2.0, label="G1")
    ax.fill(angles, n1p, color=NAVY, alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7.5)
    ax.set_title("A. Six-Dimension Student-Rating Profile", pad=18)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.18, 1.10))

    ax = fig.add_subplot(1, 2, 2)
    gains = n1 - n0
    ax.bar(labels, gains, color=[NAVY, TEAL, ORANGE, GOLD, GREY, RED], alpha=0.9)
    for i, v in enumerate(gains):
        ax.text(i, v + (1.5 if v >= 0 else -4.5), f"{v:+.1f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("G1 - G0 (0-100 display units)")
    ax.set_title("B. Dimension-Specific Gains")
    fig.suptitle("Synthetic NMLS, MCS, and PB Profiles by Architecture", y=1.02, fontsize=11)
    save(fig, "fig6_3_radar", out_png, out_pdf, thesis)
    pd.DataFrame({"dimension": labels, "G0": n0, "G1": n1, "gain": gains}).to_csv(
        out_png.parent / "source_data" / "fig6_3_radar.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.4 -- Longitudinal trajectories and echo cooling
# ---------------------------------------------------------------------------


def fig64(results: Dict, arrays: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    traj = arrays["sr_eval_t"]  # n x T x 8
    n, T, C = traj.shape
    sessions = np.arange(1, T + 1)
    g0 = traj[:, :, 0]
    g1 = traj[:, :, 7]
    m0, s0 = g0.mean(axis=0), g0.std(axis=0, ddof=1) / math.sqrt(n)
    m1, s1 = g1.mean(axis=0), g1.std(axis=0, ddof=1) / math.sqrt(n)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    ax.plot(sessions, m0, color=GREY, lw=2, marker="o", ms=3.5, label="G0")
    ax.fill_between(sessions, m0 - 1.96 * s0, m0 + 1.96 * s0, color=GREY, alpha=0.18)
    ax.plot(sessions, m1, color=NAVY, lw=2, marker="o", ms=3.5, label="G1")
    ax.fill_between(sessions, m1 - 1.96 * s1, m1 + 1.96 * s1, color=NAVY, alpha=0.18)
    ax.set_xlabel("Listening session (ride)")
    ax.set_ylabel("$\\mathrm{SR}_{\\mathrm{eval}}$ (1-7)")
    ax.set_title("A. Session-by-Session Evaluative Rating")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xticks(sessions[::2])

    echo = arrays["tensor"][:, :, 2, :].mean(axis=0)  # T x 8
    ax = axes[1]
    ax.plot(sessions, echo[:, 0], color=GREY, lw=2, marker="o", ms=3.5, label="G0")
    ax.plot(sessions, echo[:, 7], color=NAVY, lw=2, marker="o", ms=3.5, label="G1")
    ax.axhline(0.60, color=GOLD, lw=1.2, ls="--", label="Amber warning (0.60)")
    ax.axhline(0.75, color=RED, lw=1.2, ls=":", label="Cooling trigger (0.75)")
    for arm, color in [(0, GREY), (7, NAVY)]:
        hits = np.where(echo[:, arm] >= 0.75)[0]
        if hits.size:
            ax.scatter(sessions[hits], echo[hits, arm], s=55, facecolors="none", edgecolors=color, zorder=5)
    ax.set_xlabel("Listening session (ride)")
    ax.set_ylabel("Rolling echo index $R_i(t)$")
    ax.set_title("B. Echo Meter and Algorithmic Cooling")
    ax.legend(frameon=False, loc="upper right", fontsize=7.8)
    ax.set_xticks(sessions[::2])
    fig.suptitle("Longitudinal Model Trajectories (18 sessions)", y=1.02, fontsize=11)
    save(fig, "fig6_4_longitudinal", out_png, out_pdf, thesis)
    pd.DataFrame({"session": sessions, "SR_G0": m0, "SR_G1": m1, "echo_G0": echo[:, 0], "echo_G1": echo[:, 7]}).to_csv(
        out_png.parent / "source_data" / "fig6_4_longitudinal.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.5 -- Epistemic-engagement Pareto frontier and Hamiltonian
# ---------------------------------------------------------------------------


def fig65(results: Dict, arrays: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    sr = arrays["sr_eval"]
    eng = arrays["tensor"][:, :, 5, :].mean(axis=1)  # n x 8
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    colors = [GREY] + [TEAL] * 6 + [NAVY]
    for c in range(8):
        ax.scatter(eng[:, c], sr[:, c], s=8, color=colors[c], alpha=0.12, linewidths=0)
    # cell means
    for c in range(8):
        ax.scatter(eng[:, c].mean(), sr[:, c].mean(), s=80, color=colors[c], edgecolor="black", linewidth=0.5, zorder=5)
    # Pareto frontier over cell means
    pts = sorted([(eng[:, c].mean(), sr[:, c].mean()) for c in range(8)], key=lambda p: p[0])
    frontier = []
    best = -1e9
    for e, s in pts:
        if s > best:
            frontier.append((e, s))
            best = s
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, color=ORANGE, lw=1.6, ls="--", label="Cell-mean frontier")
    ax.set_xlabel("Raw engagement (model units)")
    ax.set_ylabel("$\\mathrm{SR}_{\\mathrm{eval}}$ (1-7)")
    ax.set_title("A. Epistemic-Engagement Trade-Off")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    h0 = results["factorial"]["hamiltonian_g0"]
    h1 = results["factorial"]["hamiltonian_g1"]
    ax.bar(["G0", "G1"], [h0, h1], color=[GREY, NAVY], alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean $\\mathcal{H}$ (model units)")
    ax.set_title("B. Free-Energy Diagnostic")
    ax.text(
        0.5,
        0.04,
        "Direction is reported as estimated; the Hamiltonian is a specification diagnostic,\nnot a confirmatory physical law.",
        transform=ax.transAxes,
        ha="center",
        fontsize=7.3,
        color="0.25",
    )
    fig.suptitle("Model-Based Pareto Trade-Off and Epistemic Energy", y=1.02, fontsize=11)
    save(fig, "fig6_5_pareto_hamiltonian", out_png, out_pdf, thesis)
    pd.DataFrame({"cell": lp.CELL_LABELS, "engagement_mean": [eng[:, c].mean() for c in range(8)], "sr_mean": [sr[:, c].mean() for c in range(8)]}).to_csv(
        out_png.parent / "source_data" / "fig6_5_pareto.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.6 -- GRM-IRT category curves and information
# ---------------------------------------------------------------------------


def fig66(results: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    seed = int(results["metadata"]["seed"])
    psych = lp.ValidatedPsychometricInstrument(seed=seed)
    theta = np.linspace(-3.2, 3.2, 500)
    a = psych.nmls_a[0]
    b = psych.nmls_b[0]
    p_star = 1 / (1 + np.exp(-a * (theta[:, None] - b[None, :])))
    ones = np.ones((len(theta), 1))
    zeros = np.zeros((len(theta), 1))
    p_cat = np.clip(
        np.concatenate([ones, p_star], axis=1) - np.concatenate([p_star, zeros], axis=1), 1e-9, 1.0
    )
    p_cat = p_cat / p_cat.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    for k in range(p_cat.shape[1]):
        ax.plot(theta, p_cat[:, k], lw=1.8, label=f"Category {k+1}")
    ax.set_xlabel("Latent trait $\\theta$")
    ax.set_ylabel("$P(Y_{ij}=k \\mid \\theta)$")
    ax.set_title("A. Samejima GRM Category Curves (NMLS item 1)")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_ylim(-0.02, 1.02)

    ax = axes[1]
    info, sem = psych.compute_fisher_info_and_sem(theta, psych.nmls_a, psych.nmls_b)
    ax.plot(theta, info, color=NAVY, lw=2, label="Test information $\\mathcal{I}(\\theta)$")
    ax.set_xlabel("Latent trait $\\theta$")
    ax.set_ylabel("Fisher information", color=NAVY)
    ax.tick_params(axis="y", labelcolor=NAVY)
    ax2 = ax.twinx()
    ax2.plot(theta, sem, color=ORANGE, lw=1.8, ls="--", label="SEM $=1/\\sqrt{\\mathcal{I}(\\theta)}$")
    ax2.set_ylabel("Standard error of measurement", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("B. NMLS Test Information and SEM")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Model-Based GRM-IRT Measurement Diagnostics (35-Item NMLS)", y=1.02, fontsize=11)
    save(fig, "fig6_6_grm_irt", out_png, out_pdf, thesis)
    pd.DataFrame({"theta": theta, "information": info, "sem": sem}).to_csv(
        out_png.parent / "source_data" / "fig6_6_grm_irt.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.7 -- Forest plot and prior sensitivity
# ---------------------------------------------------------------------------


def fig67(results: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    draws = np.array(results["bayesian"]["draws_tau"])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.hist(draws, bins=40, color=NAVY, alpha=0.85, density=True)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.axvline(float(np.mean(draws)), color=ORANGE, lw=1.4, ls="--", label="posterior mean")
    ax.set_xlabel("Delta SR_eval")
    ax.set_ylabel("Density")
    ax.set_title("A. Model-Internal Posterior")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    prior_draws = results["bayesian_prior_draws"]
    for key, color in zip(
        ["Skeptical_N(0,0.15^2)", "ModerateSkeptical_N(0,0.5^2)", "WeaklyInformative_N(0,1^2)", "Enthusiastic_N(0.5,0.5^2)"],
        [RED, ORANGE, NAVY, TEAL],
    ):
        arr = np.array(prior_draws[key])
        x = np.linspace(np.percentile(arr, 0.2), np.percentile(arr, 99.8), 400)
        ax.plot(x, kde_curve(arr, x, 0.30), lw=1.8, color=color, label=key.replace("_", " "))
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Posterior $\\Delta\\mathrm{SR}_{\\mathrm{eval}}$")
    ax.set_ylabel("Density")
    ax.set_title("B. Prior Sensitivity")
    ax.legend(frameon=False, fontsize=7.0)
    fig.suptitle("Model-Internal Bayesian Check and Prior Sensitivity", y=1.02, fontsize=11)
    save(fig, "fig6_7_forest", out_png, out_pdf, thesis)
    pd.DataFrame({"draw": np.arange(len(draws)), "posterior": draws}).to_csv(
        out_png.parent / "source_data" / "fig6_7_forest.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.8 -- Debiasing diagnostics and latent manifold
# ---------------------------------------------------------------------------


def fig68(results: Dict, arrays: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    balance = pd.DataFrame(results["embedding"]["covariate_balance"])
    balance = balance[balance["covariate"] != "Intercept"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    y = np.arange(len(balance))
    ax.scatter(balance["before"], y, color=GREY, label="Raw", s=30)
    ax.scatter(balance["after_dml"], y, color=NAVY, label="After DML", s=30)
    ax.scatter(balance["after_ipw"], y, color=ORANGE, label="After DML + IPW", s=30)
    for i, (b, d, w) in enumerate(zip(balance["before"], balance["after_dml"], balance["after_ipw"])):
        ax.plot([b, d], [i, i], color="0.8", lw=0.8, zorder=0)
        ax.plot([d, w], [i, i], color="0.8", lw=0.8, zorder=0)
    ax.axvline(0.10, color=RED, lw=1, ls="--", label="0.10 balance threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(balance["covariate"], fontsize=7.5)
    ax.set_xlabel("Absolute correlation with high-interaction stratum")
    ax.set_title("A. Covariate Balance (Love Plot)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    ax = axes[1]
    theta = arrays["theta_real"]
    unis = arrays["university"]
    palette = [NAVY, TEAL, ORANGE, GOLD, GREY, RED]
    for i, u in enumerate(sorted(set(unis.tolist()))):
        mask = unis == u
        ax.scatter(theta[mask, 0], theta[mask, 1], s=22, color=palette[i % len(palette)], label=u, alpha=0.85)
    # Q0 density contour
    try:
        kde = gaussian_kde(theta[:, :2].T, bw_method="scott")
        xg = np.linspace(theta[:, 0].min() - 1, theta[:, 0].max() + 1, 80)
        yg = np.linspace(theta[:, 1].min() - 1, theta[:, 1].max() + 1, 80)
        X, Y = np.meshgrid(xg, yg)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=5, colors="0.6", linewidths=0.6, alpha=0.7)
    except Exception:
        pass
    ax.set_xlabel("Debiased latent dimension 1")
    ax.set_ylabel("Debiased latent dimension 2")
    ax.set_title("B. Spectral Manifold and Q0 Support")
    ax.legend(frameon=False, fontsize=6.8, loc="best")
    fig.suptitle("Stage 1 Debiasing and Manifold Diagnostics", y=1.02, fontsize=11)
    save(fig, "fig6_8_debias_manifold", out_png, out_pdf, thesis)
    balance.to_csv(out_png.parent / "source_data" / "fig6_8_balance.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 6.9 -- Adversarial misinformation stress test
# ---------------------------------------------------------------------------


def fig69(results: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    rows = results["adversarial"]["dose_response"]
    rho = [r["rho_misinfo_pct"] for r in rows]
    f0 = [r["false_reach_G0_pct"] for r in rows]
    f1 = [r["false_reach_G1_pct"] for r in rows]
    rec0 = [r["recovery_sessions_G0"] for r in rows]
    rec1 = [r["recovery_sessions_G1"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    ax.plot(rho, f0, color=RED, marker="o", lw=2, label="G0 false reach")
    ax.plot(rho, f1, color=NAVY, marker="o", lw=2, label="G1 false reach")
    ax.fill_between(rho, f1, f0, color=TEAL, alpha=0.10)
    ax.set_xlabel("Injected misinformation rate $\\rho_{\\mathrm{misinfo}}$ (%)")
    ax.set_ylabel("Falsehood exposure reach (%)")
    ax.set_title("A. Dose-Response Suppression")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        f"Suppression: {min(r['suppression_efficiency_pct'] for r in rows[1:]):.1f}%-{max(r['suppression_efficiency_pct'] for r in rows[1:]):.1f}%",
        xy=(0.98, 0.05),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
    )

    ax = axes[1]
    width = 3.2
    x = np.arange(len(rho))
    ax.bar(x - width / 2, rec0, width, color=RED, alpha=0.85, label="G0")
    ax.bar(x + width / 2, rec1, width, color=NAVY, alpha=0.85, label="G1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}%" for r in rho])
    ax.set_xlabel("Injected misinformation rate")
    ax.set_ylabel("Sessions to recover")
    ax.set_title("B. Post-Shock Recovery")
    ax.legend(frameon=False)
    fig.suptitle("Adversarial Misinformation Stress Test (Model-Based)", y=1.02, fontsize=11)
    save(fig, "fig6_9_misinfo", out_png, out_pdf, thesis)
    pd.DataFrame(rows).to_csv(out_png.parent / "source_data" / "fig6_9_misinfo.csv", index=False)


# ---------------------------------------------------------------------------
# Figure 6.10 -- Sobol sensitivity and LHS survival
# ---------------------------------------------------------------------------


def fig610(results: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    sobol = results["sensitivity"]["sobol"]
    names = [s["parameter"] for s in sobol]
    s1 = [s["S1"] for s in sobol]
    st = [s["ST"] for s in sobol]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    ax.barh(y - 0.18, s1, height=0.34, color=NAVY, alpha=0.9, label="First-order $S_1$")
    ax.barh(y + 0.18, st, height=0.34, color=ORANGE, alpha=0.9, label="Total $S_T$")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index")
    ax.set_title("A. Variance Decomposition")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    f = np.array(results["sensitivity"]["fA"] + results["sensitivity"]["fB"])
    ax.hist(f, bins=18, color=TEAL, alpha=0.80, edgecolor="white")
    ax.axvline(0.25, color=RED, lw=1.4, ls="--", label="Survival threshold (+0.25)")
    ax.set_xlabel("$\\Delta\\mathrm{SR}_{\\mathrm{eval}}$ across LHS runs")
    ax.set_ylabel("Frequency")
    ax.set_title("B. 64-Point LHS Survival")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.95,
        f"Survival = {results['sensitivity']['survival_rate_pct']:.1f}%\n"
        f"Range [{results['sensitivity']['delta_pb_min']:+.3f}, {results['sensitivity']['delta_pb_max']:+.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )
    fig.suptitle("Global Sensitivity and Robustness (Model-Based)", y=1.02, fontsize=11)
    save(fig, "fig6_10_sobol", out_png, out_pdf, thesis)
    pd.DataFrame({"parameter": names, "S1": s1, "ST": st}).to_csv(
        out_png.parent / "source_data" / "fig6_10_sobol.csv", index=False
    )


# ---------------------------------------------------------------------------
# Figure 6.11 -- Zewail pump-probe attentional coherence
# ---------------------------------------------------------------------------


def fig611(out_png: Path, out_pdf: Path, thesis: Path):
    tau = 28.0
    Edist, Tcog = 0.45, 1.0
    delay = np.linspace(0, 120, 500)

    def p_dts(dt, phase=0.0):
        return np.exp(-dt / tau) * np.exp(-Edist / Tcog) * 0.5 * (1 + np.cos(phase))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    ax.plot(delay, p_dts(delay, 0.0), color=NAVY, lw=2, label="Phase-locked ($\\Delta\\phi=0$)")
    ax.plot(delay, p_dts(delay, np.pi / 2), color=TEAL, lw=1.8, ls="--", label="Partial ($\\Delta\\phi=\\pi/2$)")
    ax.plot(delay, p_dts(delay, np.pi), color=GREY, lw=1.8, ls=":", label="Decoupled ($\\Delta\\phi=\\pi$)")
    ax.axvline(12, color=ORANGE, lw=1.2, ls="--")
    ax.axvline(180, color=RED, lw=1.2, ls=":")
    ax.set_xlim(0, 120)
    ax.set_xlabel("Probe delay $\\Delta t$ (s)")
    ax.set_ylabel("$P_{\\mathrm{DTS}}(c,u,\\Delta t)$")
    ax.set_title("A. Coherence Decay and Gating")
    ax.legend(frameon=False, fontsize=8)
    ax.annotate("active\n12 s", xy=(12, p_dts(12, 0)), xytext=(20, 0.68), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax.text(0.98, 0.92, f"$P_{{DTS}}(12\\,s)={p_dts(12,0):.3f}$\n$P_{{DTS}}(180\\,s)={p_dts(180,0):.4f}$", transform=ax.transAxes, ha="right", va="top", fontsize=8)

    ax = axes[1]
    taus = np.linspace(8, 60, 120)
    dts = np.linspace(0, 60, 120)
    D, T = np.meshgrid(dts, taus)
    Z = np.exp(-D / T) * np.exp(-Edist / Tcog)
    im = ax.contourf(D, T, Z, levels=18, cmap="Blues")
    ax.contour(D, T, Z, levels=6, colors="white", linewidths=0.4, alpha=0.6)
    ax.set_xlabel("Probe delay $\\Delta t$ (s)")
    ax.set_ylabel("Dephasing time $\\tau_{\\mathrm{dephase}}$ (s)")
    ax.set_title("B. Coherence Across Dephasing Regimes")
    fig.colorbar(im, ax=ax, label="$P_{\\mathrm{DTS}}$ (phase-locked)")
    fig.suptitle("Zewail-Inspired Pump-Probe Attentional Coherence Gate (Formal Analogy)", y=1.02, fontsize=11)
    save(fig, "fig6_11_zewail", out_png, out_pdf, thesis)


# ---------------------------------------------------------------------------
# Figure 6.12 -- Reliability and measurement invariance
# ---------------------------------------------------------------------------


def fig612(results: Dict, out_png: Path, out_pdf: Path, thesis: Path):
    rel = results["reliability"]
    metrics = ["alpha", "omega", "guttman_l6", "spearman_brown"]
    labels = ["$\\alpha$", "$\\omega_t$", "$\\lambda_6$", "Split-half SB"]
    nmls = [rel["nmls"][m] for m in metrics]
    mcs = [rel["mcs"][m] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.34
    ax.bar(x - w / 2, nmls, w, color=NAVY, alpha=0.9, label="NMLS (35 items)")
    ax.bar(x + w / 2, mcs, w, color=TEAL, alpha=0.9, label="MCS + PB (8 items)")
    ax.axhline(0.80, color=RED, lw=1.1, ls="--", label="0.80 benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reliability coefficient")
    ax.set_title("A. Synthetic GRM Reliability")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1]
    inv = results["invariance"]
    if inv:
        stages = ["Configural", "Metric", "Scalar"]
        vals = [inv["configural"]["CFI"], inv["metric"]["CFI"], inv["scalar"]["CFI"]]
        ax.plot(stages, vals, color=NAVY, marker="o", lw=2)
        ax.axhline(0.95, color=RED, lw=1.1, ls="--", label="CFI = .95")
        ax.set_ylim(0.90, 1.005)
        ax.set_ylabel("CFI")
        ax.set_title("B. Measurement Invariance")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8.5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.55, "Invariance not estimated", ha="center", va="center",
                fontsize=11, transform=ax.transAxes)
        ax.text(0.5, 0.42, "for this synthetic run", ha="center", va="center",
                fontsize=9, transform=ax.transAxes)
    fig.suptitle("Reliability Diagnostics (Model-Based)", y=1.02, fontsize=11)
    save(fig, "fig6_12_reliability", out_png, out_pdf, thesis)
    pd.DataFrame({"metric": metrics, "NMLS": nmls, "MCS_PB": mcs}).to_csv(
        out_png.parent / "source_data" / "fig6_12_reliability.csv", index=False
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = Path(args.root)
    out_png = root / "04_figures" / "png"
    out_pdf = root / "04_figures" / "pdf"
    src = root / "04_figures" / "source_data"
    thesis = root / "01_thesis" / "src" / "figures"
    for p in [out_png, out_pdf, src, thesis]:
        p.mkdir(parents=True, exist_ok=True)

    results, arrays, measured = load(root)
    print("Generating thesis figures ...")
    fig61(results, arrays, measured, out_png, out_pdf, thesis)
    fig62(results, out_png, out_pdf, thesis)
    fig63(results, arrays, out_png, out_pdf, thesis)
    fig64(results, arrays, out_png, out_pdf, thesis)
    fig65(results, arrays, out_png, out_pdf, thesis)
    fig66(results, out_png, out_pdf, thesis)
    fig67(results, out_png, out_pdf, thesis)
    fig68(results, arrays, out_png, out_pdf, thesis)
    fig69(results, out_png, out_pdf, thesis)
    fig610(results, out_png, out_pdf, thesis)
    fig611(out_png, out_pdf, thesis)
    fig612(results, out_png, out_pdf, thesis)
    print("All figures written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
