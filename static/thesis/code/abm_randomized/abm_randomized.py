"""Randomized ABM triangulation for the LOKLOK thesis.

Three pipelines, compared, not averaged:

* Pipeline A: the existing within-persona 2x2x2 full-factorial ABM
  (read from figure_data.npz and pipeline_results.json).
* Pipeline B: agent-randomized ABM. Every simulated agent is randomly
  assigned to one of the eight A/B/C cells; the world dynamics run with
  per-agent switches. Many independent worlds are averaged.
* Pipeline C: cluster-randomized ABM. Treatment is assigned at the
  network stratum level, so everyone inside a cluster shares the same
  cell; interference between agents is modelled by the same dynamics.

Also computed:

* B-design: assignment randomization over the precomputed potential
  outcome matrix (no interference). Comparing B-sim with B-design
  isolates the role of network interference.
* Placebo sham factor D (randomized but absent from the equations).
* Negative-control outcome (pre-treatment opinion_init).
* Qualitative bridge: human gold C01-C05 support vs ABM levers.
* E1-E3 gate: eligibility check plus exploratory E1/E2; E3 stays gated.

Everything is model-conditional. No human causal claim is made.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import loklok_pipeline as v7

FACTORIAL_CELLS = v7.FACTORIAL_CELLS
N_CAND = 16

DEFAULT_OUTCOME_WEIGHTS = dict(
    cred_brs=1.20, cred_dbi=0.55, cred_echo=-0.45, cred_q=0.45, cred_prop=-2.50,
    nmls_brs=0.90, nmls_dbi=0.45, nmls_echo=-0.35, nmls_q=0.40, nmls_prop=-2.00,
)

CELL_OF_A = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=float)
CELL_OF_B = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=float)
CELL_OF_C = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def make_setup(theta, n_personas, n_sessions, seed, graph="ER",
               gate_mode="zewail", gate_tau_scale=1.0, draw_shift=None):
    """Shared pre-treatment world: personas, graph, gate, candidate content."""
    params = v7.ABMParams()
    rng = np.random.default_rng(seed)
    q0 = v7.fit_q0(theta, seed=seed)
    pd_ = v7.draw_personas(q0, n_personas, seed=seed)
    personas = pd_["traits"]
    draws = pd_["draws"].copy()
    if draw_shift:
        axis, delta = draw_shift
        draws[:, axis] = draws[:, axis] + float(delta)
        # rebuild the two shifted traits with the same frozen mapping
        personas = dict(personas)
        personas["critical_scrutiny"] = v7.sigmoid(1.25 * draws[:, 4])
        personas["echo_susceptibility"] = 0.20 + 0.45 * v7.sigmoid(1.15 * draws[:, 2])
    A = v7._make_graph(n_personas, graph, seed=seed)
    A_norm = A / (A.sum(axis=1, keepdims=True) + 1e-8)
    gate = v7._zewail_gate(n_personas, n_sessions, rng, personas["heat_capacity"],
                           mode=gate_mode, tau_scale=gate_tau_scale)
    cand_vecs = rng.normal(0, 1, size=(N_CAND, 4))
    cand_vecs_norm = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-8)
    cand_sai = rng.uniform(-0.35, 0.35, size=N_CAND)
    cand_p_prop = rng.uniform(0.10, 0.85, size=N_CAND)
    cand_quality = 1.0 - 0.75 * cand_p_prop
    return dict(
        params=params, rng=rng, personas=personas, draws=draws, A=A, A_norm=A_norm,
        gate=gate, cand_vecs_norm=cand_vecs_norm, cand_sai=cand_sai,
        cand_p_prop=cand_p_prop, cand_quality=cand_quality,
        n=n_personas, sessions=n_sessions, graph=graph, seed=seed,
        q_bar=float(np.mean(cand_quality)), p_bar=float(np.mean(cand_p_prop)),
    )


def run_assigned_world(setup, cells, world_seed, nmls_noise=0.22, mcs_noise=0.18,
                       outcome_weights=None, calibration_shift=0.0):
    """One world with per-agent treatment cells. Returns agent-level outcomes."""
    ow = dict(DEFAULT_OUTCOME_WEIGHTS)
    if outcome_weights:
        ow.update(outcome_weights)
    P = setup["params"]
    personas = setup["personas"]
    n = setup["n"]
    sessions = setup["sessions"]
    rng = np.random.default_rng(world_seed)

    fA = CELL_OF_A[cells].astype(bool)
    fB = CELL_OF_B[cells].astype(bool)
    fC = CELL_OF_C[cells].astype(bool)

    opinion = personas["opinion_init"].copy()
    brs = personas["brs_prior"].copy()
    echo = personas["echo_susceptibility"].copy()
    rec = rng.normal(0, 1, size=(n, 4))
    chosen_dbi = np.zeros(n)
    q_exp = np.full(n, setup["q_bar"])
    prop_exp = np.full(n, setup["p_bar"])
    engagement = np.zeros(n)

    for t in range(sessions):
        g = setup["gate"][:, t]
        # 1. Laplace diffusion
        opinion = opinion - P.laplace_coef * v7._matvec(np.eye(n) - setup["A_norm"], opinion)
        # 2. DeGroot + Lyapunov with per-agent BRS switch
        lam = np.where(fA, P.lam_g1, P.lam_g0)
        cap = np.where(fA, 1.0 / (1.0 + P.beta_cap * brs), np.ones_like(brs))
        A_eff = setup["A_norm"] * cap[None, :]
        A_eff = A_eff / (A_eff.sum(axis=1, keepdims=True) + 1e-8)
        anchor = np.where(
            fA,
            0.55 * personas["critical_scrutiny"] + 0.25 * g
            + 0.20 * setup["q_bar"] + 0.20,
            0.50,
        )
        brs = np.clip((1 - lam) * v7._matvec(A_eff, brs) + lam * anchor, 0.0, 1.0)
        # 3. Hilbert diversity
        u_norm = np.linalg.norm(rec, axis=1, keepdims=True) + 1e-8
        dbi_mat = 1.0 - P.hilbert_weight * ((rec / u_norm) @ setup["cand_vecs_norm"].T)
        # 4. Percolation moderation (per-agent ERIC switch, candidate-level penalty)
        mod_pen = np.where(fC[:, None],
                           np.maximum(0.0, 1.4 * (setup["cand_p_prop"][None, :] - P.tau_mod)),
                           0.0)
        uvs = np.mean(brs) * (1 - setup["cand_p_prop"][None, :] * np.where(fC[:, None], 0.85, 0.20)) * (1 - mod_pen)
        # 5. Lagrange MORE
        w1 = np.where(fB, P.w1, 1.0)
        w2 = np.where(fB, P.w2, 0.0)
        sai_pen = np.maximum(0.0, np.abs(setup["cand_sai"]) - 0.22)
        lag = w1[:, None] * uvs + w2[:, None] * dbi_mat - P.mu_lag * sai_pen[None, :]
        lag = lag + rng.normal(0, 0.03, (n, N_CAND))
        chosen = np.argmax(lag, axis=1)
        chosen_dbi = dbi_mat[np.arange(n), chosen]
        rec = 0.75 * rec + 0.25 * setup["cand_vecs_norm"][chosen]
        # 6. Entropy / Echo cooling
        echo = 0.75 * echo + 0.25 * (1 - chosen_dbi)
        echo = np.where(fB & (echo >= P.theta_cool), echo * 0.76, echo)
        echo = np.where(fC, echo * P.mod_echo_factor, echo)
        echo = np.clip(echo, 0.0, 1.0)
        # 7-8. State-mediated latent outcomes (mirrors the v8 branch)
        q_exp = setup["cand_quality"][chosen]
        prop_exp = setup["cand_p_prop"][chosen]
        cred_theta = (
            -0.25
            + ow["cred_brs"] * (brs - 0.50)
            + ow["cred_dbi"] * (chosen_dbi - 1.00)
            + ow["cred_echo"] * (echo - 0.30)
            + ow["cred_q"] * (q_exp - setup["q_bar"])
            + ow["cred_prop"] * (prop_exp - setup["p_bar"])
            + 0.80 * (personas["critical_scrutiny"] - 0.5)
            + 0.35 * (g - 0.5)
            + 0.65 * setup["draws"][:, 0]
        )
        nmls_theta = (
            -0.20
            + ow["nmls_brs"] * (brs - 0.50)
            + ow["nmls_dbi"] * (chosen_dbi - 1.00)
            + ow["nmls_echo"] * (echo - 0.30)
            + ow["nmls_q"] * (q_exp - setup["q_bar"])
            + ow["nmls_prop"] * (prop_exp - setup["p_bar"])
            + 1.05 * (personas["critical_scrutiny"] - 0.5)
            + 0.40 * (personas["cohost_preference"] - 0.5)
            + 0.30 * (g - 0.5)
            + 0.75 * setup["draws"][:, 0]
        )
        engagement = np.clip(
            0.74 + 0.35 * (echo - 0.30) - 0.10 * (chosen_dbi - 1.00)
            + 0.08 * (prop_exp - setup["p_bar"]) + 0.08 * (g - 0.5)
            + rng.normal(0, 0.05, n),
            0.0, 1.0,
        )

    psych = v7.ValidatedPsychometricInstrument(seed=world_seed, nmls_noise=nmls_noise,
                                               mcs_noise=mcs_noise)
    theta_cred = np.repeat(cred_theta[:, None], 8, axis=1)
    theta_nmls = np.repeat(nmls_theta[:, None], 35, axis=1)
    _, exp_mcs, _ = psych.simulate_grm_responses(theta_cred, psych.mcs_pb_a,
                                                 psych.mcs_pb_b, noise=mcs_noise)
    _, exp_nmls, _ = psych.simulate_grm_responses(theta_nmls, psych.nmls_a,
                                                  psych.nmls_b, noise=nmls_noise)
    sr_eval = np.mean(exp_mcs, axis=1) + calibration_shift
    sr_comp = 100.0 * (np.sum(exp_nmls, axis=1) - 35.0) / 140.0
    return dict(
        sr_eval=sr_eval, sr_comp=sr_comp, engagement=engagement,
        chosen_dbi=chosen_dbi, echo=echo, brs=brs,
        opinion_init=personas["opinion_init"],
        critical_scrutiny=personas["critical_scrutiny"],
    )


# ---------------------------------------------------------------------------
# World summaries
# ---------------------------------------------------------------------------
def cell_means_from_arrays(sr_eval, sr_comp, engagement, cells):
    out = {"sr_eval": {}, "sr_comp": {}, "engagement": {}}
    for k in range(8):
        m = cells == k
        out["sr_eval"][k] = float(np.mean(sr_eval[m]))
        out["sr_comp"][k] = float(np.mean(sr_comp[m]))
        out["engagement"][k] = float(np.mean(engagement[m]))
    return out


def effect_vector(cell_dict):
    """Main effects and interactions from the eight cell means."""
    m = np.array([cell_dict[k] for k in range(8)])
    def mean_flags(fA, fB, fC):
        return float(np.mean(m[(CELL_OF_A == fA) & (CELL_OF_B == fB) & (CELL_OF_C == fC)]))
    A = (mean_flags(1, 0, 0) + mean_flags(1, 1, 0) + mean_flags(1, 0, 1) + mean_flags(1, 1, 1)
         - mean_flags(0, 0, 0) - mean_flags(0, 1, 0) - mean_flags(0, 0, 1) - mean_flags(0, 1, 1)) / 4.0
    B = (mean_flags(0, 1, 0) + mean_flags(1, 1, 0) + mean_flags(0, 1, 1) + mean_flags(1, 1, 1)
         - mean_flags(0, 0, 0) - mean_flags(1, 0, 0) - mean_flags(0, 0, 1) - mean_flags(1, 0, 1)) / 4.0
    C = (mean_flags(0, 0, 1) + mean_flags(1, 0, 1) + mean_flags(0, 1, 1) + mean_flags(1, 1, 1)
         - mean_flags(0, 0, 0) - mean_flags(1, 0, 0) - mean_flags(0, 1, 0) - mean_flags(1, 1, 0)) / 4.0
    AB = ((mean_flags(1, 1, 0) + mean_flags(1, 1, 1) + mean_flags(0, 0, 0) + mean_flags(0, 0, 1))
          - (mean_flags(1, 0, 0) + mean_flags(1, 0, 1) + mean_flags(0, 1, 0) + mean_flags(0, 1, 1))) / 4.0
    AC = ((mean_flags(1, 0, 1) + mean_flags(1, 1, 1) + mean_flags(0, 0, 0) + mean_flags(0, 1, 0))
          - (mean_flags(1, 0, 0) + mean_flags(1, 1, 0) + mean_flags(0, 0, 1) + mean_flags(0, 1, 1))) / 4.0
    BC = ((mean_flags(0, 1, 1) + mean_flags(1, 1, 1) + mean_flags(0, 0, 0) + mean_flags(1, 0, 0))
          - (mean_flags(0, 1, 0) + mean_flags(1, 1, 0) + mean_flags(0, 0, 1) + mean_flags(1, 0, 1))) / 4.0
    ABC = ((mean_flags(1, 1, 1) + mean_flags(1, 0, 0) + mean_flags(0, 1, 0) + mean_flags(0, 0, 1))
           - (mean_flags(0, 0, 0) + mean_flags(0, 1, 1) + mean_flags(1, 0, 1) + mean_flags(1, 1, 0))) / 4.0
    return dict(A=A, B=B, C=C, AB=AB, AC=AC, BC=BC, ABC=ABC,
                ATE=cell_dict[7] - cell_dict[0])


def world_summary(res, cells, sham, groups):
    cm_eval = cell_means_from_arrays(res["sr_eval"], res["sr_comp"], res["engagement"], cells)
    row = {}
    for name, cd in [("sr_eval", cm_eval["sr_eval"]), ("sr_comp", cm_eval["sr_comp"]),
                     ("engagement", cm_eval["engagement"])]:
        e = effect_vector(cd)
        for k, v in e.items():
            row[f"{name}_{k}"] = v
    # mechanism direction checks
    row["mechanism_dbi_B"] = float(np.mean(res["chosen_dbi"][CELL_OF_B[cells].astype(bool)])
                                   - np.mean(res["chosen_dbi"][~CELL_OF_B[cells].astype(bool)]))
    row["mechanism_brs_A"] = float(np.mean(res["brs"][CELL_OF_A[cells].astype(bool)])
                                   - np.mean(res["brs"][~CELL_OF_A[cells].astype(bool)]))
    bc = (CELL_OF_B[cells].astype(bool) & CELL_OF_C[cells].astype(bool))
    row["mechanism_echo_BC"] = float(np.mean(res["echo"][bc]) - np.mean(res["echo"][~bc])) if (~bc).any() else 0.0
    # placebo: sham factor D is absent from the equations
    row["placebo_D_eval"] = float(np.mean(res["sr_eval"][sham == 1]) - np.mean(res["sr_eval"][sham == 0]))
    row["placebo_D_comp"] = float(np.mean(res["sr_comp"][sham == 1]) - np.mean(res["sr_comp"][sham == 0]))
    # negative control: pre-treatment trait should not respond to assignment
    for eff, flag in [("negA", CELL_OF_A), ("negB", CELL_OF_B), ("negC", CELL_OF_C)]:
        f = flag[cells].astype(bool)
        row[f"negative_{eff}"] = float(np.mean(res["opinion_init"][f]) - np.mean(res["opinion_init"][~f]))
    # heterogeneity: main A/B/C within each stratum
    het = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 10:
            continue
        gA = np.mean(res["sr_eval"][m & CELL_OF_A[cells].astype(bool)]) - np.mean(res["sr_eval"][m & ~CELL_OF_A[cells].astype(bool)])
        gB = np.mean(res["sr_eval"][m & CELL_OF_B[cells].astype(bool)]) - np.mean(res["sr_eval"][m & ~CELL_OF_B[cells].astype(bool)])
        het[f"stratum_{int(g)}"] = dict(A=float(gA), B=float(gB))
    row["heterogeneity"] = het
    return row


def aggregate_worlds(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "heterogeneity"} for r in rows])
    out = {}
    for c in df.columns:
        out[c] = dict(mean=float(df[c].mean()), sd=float(df[c].std(ddof=1)) if len(df) > 1 else 0.0,
                      lo=float(df[c].quantile(0.025)), hi=float(df[c].quantile(0.975)))
    return out


def run_design(theta, n_personas, n_sessions, base_seed, graph="ER", mode="agent",
               n_worlds=20, groups=None, calibration_shift=0.0, draw_shift=None,
               nmls_noise=0.22, mcs_noise=0.18):
    """Run an agent- or cluster-randomized ABM design and return aggregates."""
    setup = make_setup(theta, n_personas, n_sessions, base_seed, graph=graph,
                       draw_shift=draw_shift)
    rows = []
    n_groups = int(np.max(groups) + 1) if groups is not None else 6
    for w in range(n_worlds):
        rng = np.random.default_rng(base_seed + 1000 * (w + 1))
        if mode == "agent":
            cells = rng.integers(0, 8, size=n_personas)
        elif mode == "cluster":
            g_cells = rng.integers(0, 8, size=n_groups)
            cells = g_cells[groups]
        else:
            raise ValueError(mode)
        sham = rng.integers(0, 2, size=n_personas)
        res = run_assigned_world(setup, cells, base_seed + 2000 * (w + 1),
                                 nmls_noise=nmls_noise, mcs_noise=mcs_noise,
                                 calibration_shift=calibration_shift)
        rows.append(world_summary(res, cells, sham, groups))
    agg = aggregate_worlds(rows)
    return dict(agg=agg, worlds=rows, n_worlds=n_worlds, graph=graph, mode=mode)


# ---------------------------------------------------------------------------
# Pipeline A from saved artifacts
# ---------------------------------------------------------------------------
def pipeline_a(npz, engagement_tensor):
    sr_eval = npz["sr_eval"]
    sr_comp = npz["sr_comp"]
    n = sr_eval.shape[0]
    cells = np.zeros(8, dtype=int)
    row = {}
    for name, mat in [("sr_eval", sr_eval), ("sr_comp", sr_comp)]:
        cm = {k: float(np.mean(mat[:, k])) for k in range(8)}
        e = effect_vector(cm)
        for k, v in e.items():
            row[f"{name}_{k}"] = v
    eng = engagement_tensor.mean(axis=0)
    cm_eng = {k: float(eng[k]) for k in range(8)}
    e = effect_vector(cm_eng)
    for k, v in e.items():
        row[f"engagement_{k}"] = v
    return dict(agg={k: dict(mean=v, sd=0.0, lo=v, hi=v) for k, v in row.items()},
                n_worlds=1, graph="ER", mode="within_persona")


def design_based_randomization(npz, engagement_tensor, pre_trait, groups=None,
                               n_draws=400, seed=0):
    """Assignment randomization over the precomputed potential outcomes."""
    rng = np.random.default_rng(seed)
    sr_eval, sr_comp = npz["sr_eval"], npz["sr_comp"]
    n = sr_eval.shape[0]
    rows = []
    for d in range(n_draws):
        cells = rng.integers(0, 8, size=n)
        sham = rng.integers(0, 2, size=n)
        res = dict(sr_eval=sr_eval[np.arange(n), cells],
                   sr_comp=sr_comp[np.arange(n), cells],
                   engagement=engagement_tensor[np.arange(n), cells],
                   chosen_dbi=np.zeros(n), echo=np.zeros(n), brs=np.zeros(n),
                   opinion_init=pre_trait[:n],
                   critical_scrutiny=np.zeros(n))
        row = world_summary(res, cells, sham, groups)
        row.pop("mechanism_dbi_B", None)
        row.pop("mechanism_brs_A", None)
        row.pop("mechanism_echo_BC", None)
        rows.append(row)
    return aggregate_worlds(rows), rows


# ---------------------------------------------------------------------------
# Qualitative bridge and E1-E3
# ---------------------------------------------------------------------------
CODE_LEVER = {"C01": "cred_brs", "C02": "cred_dbi", "C03": "cred_echo",
              "C04": "cred_q", "C05": "cred_prop"}
LEVER_BOUNDS = {"cred_brs": (0, 1.8), "cred_dbi": (0, 0.825), "cred_echo": (-0.675, 0),
                "cred_q": (0, 0.675), "cred_prop": (-3.75, 0)}
LEVER_DEFAULT = {"cred_brs": 1.2, "cred_dbi": 0.55, "cred_echo": -0.45,
                 "cred_q": 0.45, "cred_prop": -2.5}
CODE_ABM_ROLE = {
    "C01": ("BRS / credibility path", "cred_brs, beta_cap"),
    "C02": ("MORE / Lagrange diversity", "w2, theta_cool"),
    "C03": ("DeGroot echo / herd", "cred_echo, lam_g1"),
    "C04": ("ERIC moderation / reflection", "cred_q, tau_mod"),
    "C05": ("Misinformation circulation", "cred_prop, adversarial rho"),
}
CODE_TRAIT = {"C01": ("critical_scrutiny", 4), "C04": ("critical_scrutiny", 4),
              "C03": ("echo_susceptibility", 2)}


def qual_alignment(qual_summary, codebook_names, zero_csv, gold_csv):
    gold = qual_summary["human_gold"]
    q0 = pd.read_csv(zero_csv)
    q1 = pd.read_csv(gold_csv)
    rows = []
    for cid in ["C01", "C02", "C03", "C04", "C05"]:
        n_ref = gold["positive_counts"].get(cid, 0)
        k0 = q0[q0.code_id == cid]["kappa"].iloc[0] if (q0.code_id == cid).any() else np.nan
        k1 = q1[q1.code_id == cid]["kappa"].iloc[0] if (q1.code_id == cid).any() else np.nan
        dec1 = q1[q1.code_id == cid]["decision"].iloc[0] if (q1.code_id == cid).any() else "missing"
        role, levers = CODE_ABM_ROLE[cid]
        rows.append(dict(code_id=cid, name=codebook_names.get(cid, cid), role=role,
                         abm_levers=levers, n_human_positive=n_ref,
                         kappa_qwen_zero=round(float(k0), 3) if np.isfinite(k0) else None,
                         kappa_qwen_gold=round(float(k1), 3) if np.isfinite(k1) else None,
                         decision=dec1,
                         evidence_tier="validated" if dec1 == "usable" else
                         ("descriptive" if dec1 == "descriptive_only" else "insufficient_or_dropped"),
                         interpretation="supports design assumption (descriptive only)" if n_ref > 0 else "no support"))
    return pd.DataFrame(rows)


def e_gate(qual_summary, gold_csv, min_usable=4):
    g = pd.read_csv(gold_csv)
    usable = g[g.decision == "usable"].code_id.tolist()
    mech_usable = [c for c in usable if c in CODE_LEVER]
    eligible = len(mech_usable) >= min_usable
    return dict(rule=f"at least {min_usable} of the mechanism codes usable",
                usable_codes=usable, mechanism_usable=mech_usable,
                eligible=bool(eligible),
                status="eligible" if eligible else "not_eligible")


def e1_exploratory(pipeline_results, qual_summary):
    rows = pipeline_results["sensitivity"]["joint_sweep"]["rows"]
    gold = qual_summary["human_gold"]["positive_counts"]
    total = qual_summary["human_gold"]["units"]
    weights = np.ones(len(rows))
    specs = {}
    for cid, lever in CODE_LEVER.items():
        n = gold.get(cid, 0)
        s = (n + 0.5) / (total + 1.0) if total else 0.5
        lo, hi = LEVER_BOUNDS[lever]
        default = LEVER_DEFAULT[lever]
        target = float(np.clip(default + 0.25 * (s - 0.5), lo, hi))
        sigma = max(1e-6, (hi - lo) / (2.0 + n / 5.0))
        col = np.array([float(r.get(lever, default)) for r in rows])
        weights *= np.exp(-0.5 * ((col - target) / sigma) ** 2)
        specs[cid] = dict(lever=lever, n=n, s=round(float(s), 4), target=round(target, 4),
                          sigma=round(float(sigma), 4), default=default)
    vals = np.array([float(r["delta_sr_eval"]) for r in rows])
    order = np.argsort(vals)
    v, w = vals[order], weights[order]
    c = np.cumsum(w) / np.sum(w)
    weighted = {q: float(np.interp(q, c, v)) for q in (0.025, 0.25, 0.5, 0.75, 0.975)}
    uniform = {q: float(np.quantile(vals, q)) for q in (0.025, 0.25, 0.5, 0.75, 0.975)}
    return dict(status="exploratory_gate_not_met", specs=specs, weighted=weighted,
                uniform=uniform, delta_median=weighted[0.5] - uniform[0.5])


def design_comparison(A_agg, B_agg, B_design_agg, C_agg):
    cols = ["sr_eval_ATE", "sr_comp_ATE", "engagement_ATE",
            "sr_eval_A", "sr_eval_B", "sr_eval_C", "sr_eval_AB", "sr_eval_AC",
            "sr_eval_BC", "sr_eval_ABC"]
    rows = []
    for name, agg, nw in [("A_within_persona", A_agg, 1),
                          ("B_agent_randomized", B_agg, B_agg.get("n_worlds") if isinstance(B_agg, dict) else None),
                          ("B_design_based", B_design_agg, B_design_agg.get("n_worlds") if isinstance(B_design_agg, dict) else None),
                          ("C_cluster_randomized", C_agg, C_agg.get("n_worlds") if isinstance(C_agg, dict) else None)]:
        if not isinstance(agg, dict) or not agg:
            continue
        r = dict(design=name, n_worlds=nw)
        for c in cols:
            if c in agg:
                v = agg[c]
                r[c] = v["mean"]
                r[c + "_lo"] = v["lo"]
                r[c + "_hi"] = v["hi"]
        rows.append(r)
    return pd.DataFrame(rows)


def est_verification(comp_df, qual_align_df, gate, reliability):
    rows = []
    def val(design, col):
        m = comp_df[comp_df.design == design]
        return float(m[col].iloc[0]) if len(m) else np.nan

    designs = list(comp_df.design)
    # EST-P1: BRS architecture raises evaluative rating
    a = [val(d, "sr_eval_A") for d in designs]
    rows.append(dict(proposition="EST-P1 architecture/BRS", check="A main effect on SReval > 0",
                     result=str([round(x, 3) if np.isfinite(x) else None for x in a]),
                     verdict="supported_within_model" if all(x > 0 for x in a) else "design_dependent"))
    b = [val(d, "sr_eval_B") for d in designs]
    rows.append(dict(proposition="EST-P2 diversity/cooling", check="B main effect on SReval > 0",
                     result=str([round(x, 3) if np.isfinite(x) else None for x in b]),
                     verdict="supported_within_model" if all(x > 0 for x in b) else "design_dependent"))
    eng = [val(d, "engagement_ATE") for d in designs]
    rows.append(dict(proposition="EST-P4 engagement trade-off", check="engagement G1-G0 < 0",
                     result=str([round(x, 3) if np.isfinite(x) else None for x in eng]),
                     verdict="supported_within_model" if all(x < 0 for x in eng) else "design_dependent"))
    rows.append(dict(proposition="EST-P3 measurement", check="GRM reliability already reported",
                     result=str(reliability),
                     verdict="inherited_from_v8"))
    rows.append(dict(proposition="E1-E3 qualitative gate",
                     check="at least 4 mechanism codes usable",
                     result=str(gate),
                     verdict="not_eligible" if not gate["eligible"] else "eligible"))
    return pd.DataFrame(rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
