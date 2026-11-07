---
title: "LOKLOK Thesis"
description: "The Social Media Radio Innovation and Its Impact on University Students' Ratings: An Exploratory Study"
tags: ["Thesis", "LOKLOK", "Social Media Radio", "EST", "ABM"]
author: ["Sherif Mohamed Abd Elmonem Rashwan"]
showToc: true
TocOpen: true
hidemeta: false
disableShare: false
cover:
  image: "figures/fig_embedding_space.png"
  alt: "LOKLOK thesis cover figure"
  caption: "BGE-M3 document vector space and stream similarity"
  relative: true
---

**Sherif Mohamed Abd Elmonem Rashwan**  
Media Innovations, Faculty of Mass Communication, Ahram Canadian University (ACU)  
November 2026

**Supervisors:** Dr. Engi Aboul-Ezz and Dr. Mostafa ElBanna

---

## Read the full thesis

- [**Read the full thesis page by page**](read/) ? all 363 pages in the browser.
- [Complete thesis PDF](thesis_final.pdf) - 363 pages, 43/43 package checks passed.
- [Bilingual supervisor summary PDF](Supervisor_Summary_EN_AR.pdf) - 20 pages, English first, Arabic second.
- [English summary](Supervisor_Summary_EN.pdf)
- [Arabic summary](Supervisor_Summary_AR.pdf)
- [Thesis code and notebooks](downloads/loklok_code.zip)
- [Selected thesis data and model outputs](downloads/loklok_data.zip)
- [All thesis figures](downloads/loklok_figures.zip)

The 52-student wave is measured, cross-sectional, and descriptive. The 500 personas are a computational population, not 500 respondents. The G0/G1 contrast is a model-based counterfactual, not a human treatment effect.

---

## Abstract

This thesis asks whether the design of a social media radio platform can improve how university students judge what they hear, without losing the audience. It proposes LOKLOK, a platform built on four ideas: a reputation score for broadcasters, recommendations constrained to keep viewpoints varied, moderation with prompts for reflection, and a built-in measurement surface. Epistemic Systems Theory is offered as the framework that links these design choices to audience judgment.

A live trial was not possible, so the evidence has three layers. The measured layer is a questionnaire completed by 52 students (60 were collected and 8 were removed) who were riding Uber in 6th of October City, Giza. Perceived benefit averaged 5.12 on a seven-point scale (SD = 0.87) and correlated with the overall star rating (rho = .67). The credibility and perceived-benefit scales were both reliable (alpha = .96 and .95), but the factor model fitted poorly, so no claim of structural validity is made. The qualitative layer, a group-session transcript and a researcher's self-report, was used to check assumptions and contains little participant speech about the mechanisms themselves. Three independent coders labelled 193 segments with pairwise Cohen's kappa of .76 to .86, but a local 7B model (micro-F1 .19 zero-shot, .32 with human-gold examples) and a first-pass assistant coding (.45) did not reach the code-level reliability gate for any mechanism code; the qualitative layer therefore remains an assumption audit rather than model evidence. A randomized-assignment triangulation of the same eight cells kept the diversity component (+0.63) and the engagement trade-off in every design but attenuated the BRS component from +0.19 within-persona to +0.04 between agents. The modelling layer is a calibrated in-silico experiment: 500 synthetic personas, drawn from the debiased distribution anchored in the 52 respondents, were run through eight combinations of three design components over 18 listening sessions, with the status-quo arm anchored to the measured mean.

In this model the full design raised the evaluative rating by 0.83 points (95% interval 0.78 to 0.88) and a competence-format rating by 11.3 points on a 100-point scale, while engagement fell by 0.18 on a 0 to 1 scale. Most of the gain came from the diversity component, with moderation contributing mainly through its interaction with it. Across 256 draws that varied the assumed weights and mechanism parameters, the contrast was positive in 96.5% of cases but ranged from -0.24 to +1.81, and it depended chiefly on one declared link between exposure diversity and credibility.

These are model-based implications, not treatment effects. The sample is a convenience sample, and no field causal test is claimed. The thesis contributes an explicit route from theory to a testable platform design, and a plan for a real deployment.

## Why social media radio, not generic social media

The thesis uses social media radio as a bounded, mechanism-oriented case, not as a representative sample of all social media. Radio-like audio is voice-centred and temporally organised; a listener encounters a programmed stream or episode, not an undifferentiated multimodal feed. This makes broadcaster identity, vocal presence, sequence, and repair observable, and gives engagement a behavioural referent (continued listening, replay, interaction) rather than a generic click. It also lets editorial selection, platform selection, and listener selection be separated, so BRS, diversity-constrained recommendation, echo control, and correction can be varied as distinct architectural levers.

A generic "social media" design would confound modality with architecture: text, image, video, short-form, live, and ranked feeds bundle different attention regimes, production costs, and credibility cues; likes, watch time, completed listens, and credibility ratings are not construct-equivalent; and many platform features would change at once. The thesis therefore makes an analytical, not statistical, generalization: LOKLOK is a plausibility probe for EST, and the mechanism, not the medium, is the inferential target.

The pipeline follows the same scope logic. The field wave fixes the population and radio-specific instruments; the qualitative stream supplies candidate design vocabulary, not calibration; the BGE-M3 four-stream representation is a pipeline feature, not evidence of human fidelity; the ABM instantiates the eight cells with 500 personas and 18 sessions inside a radio-like feed; and the randomized-assignment triangulation tests whether the contrast depends on assignment and interference. Extending the pipeline to text, image, or video platforms would require revalidated instruments, modality-specific stimuli, and a new identification argument.

## Qualitative machine-coding audit

Three independent human coders labelled 193 usable FGD/self-report segments on a 25-code codebook. Pairwise Cohen's kappa was .762, .852, and .861, and the majority label formed the human gold. The local Qwen2.5-7B zero-shot pass reached micro-F1 .193, with 32.0% of its evidence quotes not exact substrings of the source. A leakage-safe five-fold pass with one human-gold exemplar per target raised micro-F1 to .320 and lowered fabrication to 16.1%. A first-pass Claude coding reached micro-F1 .452, with I01 and I07 usable and C12, I03, I06, and I13 descriptive only. No mechanism code C01 to C05 passed the usability gate in either Qwen condition: C05 had eight reference positives but was dropped at kappa .259 zero-shot and .323 with gold examples. C04 reached .659 with gold examples but had only five positives and remained insufficient. E1 to E3 were therefore not run. The qualitative layer remains an assumption audit and candidate design vocabulary, not a calibration target or an input to the simulated contrast.

## Randomized ABM triangulation

The eight A/B/C cells were re-analyzed under four designs: the saved within-persona full factorial (A), an agent-randomized ABM with 40 independent worlds (B), a design-based randomization with 400 assignment draws and no interference, and a cluster-randomized ABM with 20 worlds (C).

- The contrast stays positive in every design: +0.827 within-persona, +0.630 between agents, +0.807 design-based, +0.659 cluster-randomized.
- The engagement cost is the most stable quantity: -0.217, -0.232, -0.217, -0.226.
- The diversity main effect is nearly identical everywhere (+0.626, +0.628, +0.627).
- The BRS main effect collapses from +0.190 within-persona to +0.038 between agents, then recovers to +0.182 in the interference-free design. BRS diffusion is the interference-sensitive component.
- Placebo sham factor D: -0.003 on SReval, -0.082 on SRcomp. Pre-treatment negative-control outcome: +0.001, -0.008, 0.000.
- Mechanism directions correct: B raised chosen diversity +1.299, A raised BRS +0.019, B×C lowered echo -0.340.
- Qualitative gate remains unmet (0 of 4 mechanism codes usable). Exploratory E1 moved the sweep median from +0.567 to +0.585; exploratory E2 trait shifts changed the between-agent contrast by -0.043 and -0.042. E3 remains not run.

All of this is model-conditional triangulation, not a human causal claim.

## Key results

| Quantity | Estimate | 95% interval | Test |
|---|---:|---:|---|
| \(\Delta\mathrm{SR}_{\mathrm{eval}}\) | **+0.827** | [0.779, 0.875] | \(t(499)=33.93\), \(p<.001\), \(d_z=1.52\) |
| \(\Delta\mathrm{SR}_{\mathrm{comp}}\) | **+11.30** | [10.64, 11.95] | \(t(499)=33.74\), \(p<.001\), \(d_z=1.51\) |
| \(\Delta\) engagement | **âˆ’0.183** | [âˆ’0.188, âˆ’0.178] | \(t(499)=âˆ’74.41\), \(p<.001\), \(d_z=âˆ’3.33\) |
| Bayesian \(\Delta\mathrm{SR}_{\mathrm{eval}}\) | +0.8265 | HDI [0.7793, 0.8764] | normal summary, not independent evidence |
| 256-point joint sweep | median +0.567 | IQR [0.383, 0.802] | 96.5% positive; default at 78.5th percentile |
| Anchor bootstrap | +0.839 | [0.805, 0.884] | 50 resamples; 100% positive |
| 60-row sensitivity | +0.808 | [0.760, 0.856] | exclusion rule changes magnitude only |
| Gemini persona pilot | +2.650 | [2.342, 2.967] | single-shot appraisal; not 18-session simulation |

Main effects: A **+0.1905**, B **+0.6265**, C **âˆ’0.0121**; BÃ—C **+0.3791**. The diversity-seeking path dominates. The top joint-sweep driver is `cred_dbi` (PRCC +0.77), not BRS or echo. The negative `cred_echo` PRCC is a sign convention because the weight itself is negative.

---

## 1. Problem, questions, and contribution

The thesis starts from a separation in the literature: platform design, audience measurement, and epistemic quality are usually studied separately. Social-media radio combines algorithmic recommendation, live presentation, audience metrics, and a public-service language. Existing work describes filter bubbles, misinformation diffusion, parasocial relations, and media literacy, but rarely links those mechanisms to a measurable audience rating and a testable platform design.

**Research questions**

1. Does platform architecture shape structural credibility and audience evaluation (EST-P1)?
2. Do diversity and cooling mechanisms constrain epistemic concentration (EST-P2)?
3. Can the proposed instruments meet basic psychometric accountability (EST-P3)?
4. Do the EST constructs integrate into a coherent measurement model (EST-P4)?
5. Which mechanisms and weights drive the model-based contrast?
6. What does a Gemini persona-agent pilot add to the comparison?

The contribution is an integrated framework and a transparent testbed: EST and four propositions; a specified LOKLOK architecture (BRS, MORE, ERIC, Echo Meter, Algorithmic Cooling, Zewail gate); a measured wave and qualitative corpus; a reproducible calibrated in-silico experiment; a GRM-IRT measurement layer; and explicit claim tiers. The four-stream BGE-M3 embedding is a pipeline feature, not the headline contribution; the source ablation shows that the semantic block is contextual.

---

## 2. EST propositions and hypotheses

EST proposes four linked propositions. EST-P1 treats platform architecture as a structural condition of credibility. EST-P2 treats diversity and cooling as constraints on epistemic concentration. EST-P3 treats measurement as an accountability requirement. EST-P4 treats participatory performance as the educational test.

Hypotheses: H1 structural credibility, H2 diversity mechanism, H3 psychometric validity, H4 integration, H5 DeGroot conformity (exploratory), H6 scheduling/duration (exploratory), and H7 participatory performance (prospective).

| Proposition | Hypothesis | Evidence in this thesis | Verdict |
|---|---|---|---|
| EST-P1: architecture shapes structural credibility | H1, H4, H5 | Measured PBâ€“star and BRS associations; model BRS path; no field randomisation | Partial; model-conditional, not a field causal claim |
| EST-P2: diversity and cooling | H2 | MORE/Lagrange is the dominant model path; cooling and echo terms implemented; topology check is network-independent | Supported within the model; field mechanism untested |
| EST-P3: measurement accountability | H3 | High internal consistency; poor single-factor CFA; invariance not estimated | Mixed: reliability supported, structural validity not established |
| EST-P4: participatory performance | H7 | Prospective claim-appraisal task; no human task data | Untested; future study required |

In short: H1 is partially supported but not identified as a field effect; H2 is supported within the model; H3 is mixed; H4 is associational; H5 and H6 are exploratory model checks; H7 is prospective. The conclusion keeps the model-based trade-off as the central finding and the human causal claim as future work.

---

## 3. Data and measurement

The measured wave began with 60 raw questionnaires. A 21-field duplicate screen removed eight duplicate vectors, leaving 52 analysis records. The instrument contains an 11-item broadcaster-rating battery (BRS), a five-item perceived-benefit battery (PB), a star rating, ride duration, interaction level, and context items. Internal consistency is high (BRS Î±=.957, PB Î±=.947, Ï‰=.964/.962), but the single-factor CFA fit is poor. Reliability and validity are reported separately.

The qualitative stream contains a 160-segment FGD transcript, a 185-segment self-report transcript, 60 participant-observation accounts, 4,828 corpus documents, and 25 programme briefs. The 52 retained records supply 31 quantitative features; the semantic block supplies 167 features; the total is 198.

### 3.1 Sentence-transformer architecture and process

The semantic encoder is **BAAI/bge-m3**, a multilingual sentence-transformer with an XLM-RoBERTa backbone. It returns a 1,024-dimensional dense vector and supports inputs up to 8,192 tokens. The thesis uses the dense output only; sparse and multi-vector outputs are not used. The encoder is zero-shot: it is not fine-tuned on the thesis data, and no supervised label enters the embedding step.

The process is: assemble 5,258 document texts; clean and truncate each text to the model limit; tokenise and encode mini-batches on a Colab T4 in mixed precision; pool and L2-normalise the dense representation; cache vectors by a hash of model name and document set; project the semantic block to 32 principal components; and combine 167 semantic features with 31 quantitative features. The cache makes the run reproducible without a second API call. This is a measurement pipeline, not a claim that the 1,024 dimensions are psychological constructs.

### 3.2 Embedding and latent-space diagnostics

The 5,258 vectors are unit length. The first two principal components explain 44.9% of the variance (36.9% and 8.0%). Within-stream cosine similarity is highest for participant observation (0.785), followed by stimuli (0.568), corpora (0.559), FGD (0.531), and self-report (0.466). Cross-stream similarity is lower (0.38â€“0.48), and the cosine silhouette is 0.196: moderate, not clean, separation.

The six-axis manifold is a debiasing coordinate system, not a six-factor psychological model. For the 52 anchors, the covariance condition number is 4.815 and the largest absolute off-diagonal correlation is 0.434. The first two principal components explain 49.9% of the latent variance. Mean within-university distance is 3.211 and mean between-university distance is 3.295 (ratio 0.974): university labels do not separate the anchors. The 500 personas are draws from the fitted Qâ‚€; the figure shows the 52 anchors, not 500 respondents.

![BGE-M3 document vector space and similarity](figures/fig_embedding_space.png)

*Figure 1. BGE-M3 document vector space (A) and mean cosine similarity (B). The projection is descriptive, not a validated psychological map.*

![Latent-space diagnostics](figures/fig_latent_diagnostics.png)

*Figure 2. Latent-space diagnostics for the 52 anchors: axis correlations (A), PCA variance (B), within/between university distances (C), and standardised profiles (D).*

---

## 4. Model and inference

The pipeline applies DML and IPW debiasing, then a spectral manifold (PCA + LLE + Laplacian eigenmaps). The maximum nuisance correlation falls from 0.887 to 0.042 after DML; IPW reintroduces some imbalance (0.127), which is reported rather than hidden. The retained six-axis condition number is 4.82. A Gaussian KDE (or a shrunken multivariate normal if ill-conditioned) defines Qâ‚€; 500 personas are drawn.

Each persona is cloned across the \(2^3=8\) A/B/C cells and run for 18 sessions. The state tensor is propagated through eight filters: Hamiltonian, Laplace, Lyapunov, DeGroot, Lagrange, Hilbert, Percolation, and Entropy. The Zewail pumpâ€“probe gate multiplies the evidence anchor and attentive-retention term. The terminal state is scored by a Samejima GRM-IRT model for the 35-item NMLS and the 8-item MCS+PB battery. G0 is anchored to the measured PB mean of 5.119; treatment weights are declared design inputs, not estimates from the 52 respondents.

Inference is reported in two frameworks. The frequentist analysis uses paired t-tests, Cohen's \(d_z\), 95% repeated-sampling intervals, a full-factorial decomposition, and Wasserstein distances. The Bayesian analysis is a one-sample normal summary of the same paired differences. It is a model-internal re-expression, not independent evidence; its interval width depends on the chosen persona count. The joint sweep, not \(P(\Delta>0)\), is the relevant uncertainty statement.

A Gemini persona-agent pilot used 24 measured personas Ã— 8 conditions = 192 calls. It is a single-shot appraisal of the condition description, not an 18-session LLM simulation. It tests whether the direction of the result survives a different agent formalism.

---

## 5. Results

### 5.1 Main contrast and decomposition

The evaluative gain is +0.827 points on the 1â€“7 scale, the competence-format gain is +11.30 points on the 0â€“100 scale, and engagement falls by âˆ’0.183 on the 0â€“1 scale. The engagement cost is substantial in standardized terms (\(d_z=-3.33\)).

Factorial decomposition: A +0.1905, B +0.6265, C âˆ’0.0121; AÃ—B +0.0216, AÃ—C âˆ’0.0084, BÃ—C +0.3791, AÃ—BÃ—C +0.0886. MORE dominates; BRS adds a smaller main effect; ERIC matters conditionally through diversity and cooling. Because BÃ—C is nearly half of the total contrast, a share-of-main-effects attribution is misleading and is not used.

### 5.2 What drives the result

The leave-one-filter-out ablation shows that only DeGroot, Lagrange, and Hilbert change the terminal contrast in this wiring. Disabling DeGroot lowers \(\Delta\mathrm{SR}_{\mathrm{eval}}\) by 0.233 [0.197, 0.269], disabling Lagrange by 0.750 [0.713, 0.787], and disabling Hilbert by 0.453 [0.402, 0.503]. Lagrange is the operationalisation of factor B, so the Lagrange-off run is a second switch-off of the diversity objective, not independent evidence for a separate filter. Laplace, Percolation, and Entropy leave the contrast unchanged, and Lyapunov is negligible. The Hamiltonian is a diagnostic and does not enter the outcome.

The source ablation shows that the quantitative block carries most of the contrast. Dropping all semantic features changes \(\Delta\mathrm{SR}\) by only âˆ’0.009 [âˆ’0.029, 0.010]; shuffling the semantic matrix changes it by âˆ’0.019 [âˆ’0.039, 0.001]. The shuffled-text and quantitative-only runs remain close to the full contrast (+0.808 and +0.818). The 52 respondents therefore set the intercept and the persona spread; the contrast is carried by the mechanism and the state-outcome weights. The semantic block is contextual, not load-bearing.

The 256-point joint sweep varies six mechanism parameters and five state-outcome weights, including zero. The contrast ranges from âˆ’0.244 to +1.810 (mean +0.593, median +0.567, IQR [0.383, 0.802]); 96.5% of draws are positive. The default +0.827 sits at the 78.5th percentile. The top PRCCs are `cred_dbi` +0.77, `cred_brs` +0.31, and `cred_echo` âˆ’0.27. The negative echo value is a sign convention: the weight itself is negative. In one sentence: a diversity-seeking selection rule raises chosen diversity, and chosen diversity is assumed to raise perceived credibility. That is the causal chain the thesis tests, not a claim that the field data identified the weight.

![Frequentist and Bayesian comparison](figures/fig_freq_bayes.png)

*Figure 3. Frequentist and Bayesian summaries of the primary paired contrast. The intervals nearly coincide because they use the same simulated differences; this is not independent evidence.*

### 5.3 Robustness and the semantic question

The contrast is stable to the 60-row file (+0.808 [0.760, 0.856] vs +0.827), to 50 anchor-bootstrap resamples (mean +0.839, 95% [0.805, 0.884], 100% positive), and to ER/WS/BA topology (+0.827/+0.826/+0.832). The topology result follows from structure: the graph enters only through BRS diffusion, and A is the smallest effect. The dominant B path is network-independent; the topology check confirms that structure rather than adding an independent robustness result.

The held-out predictive comparison uses 52 rows and 198 features. Psychometrics alone predict PB best (RÂ²=.915); adding 167 semantic features lowers PB RÂ² to .583, and semantic-only has negative RÂ². In this sample, the semantic block does not improve held-out prediction. With 52 rows and 167 features, that is expected from overfitting alone; it is not evidence that semantics are uninformative in a larger sample. The content-supply stress test sets a heuristic poor-content share of 0.763 from the Uber-review star mean of 1.69; it is an arbitrary stress parameter, not a measured radio-propaganda rate. It lowers \(\Delta\mathrm{SR}\) to +0.636, paired âˆ’0.191 [âˆ’0.261, âˆ’0.122]; the sign remains positive.

### 5.4 Gemini pilot

The Gemini arm reproduced the direction of the evaluative gain but overstated it (+2.650 [2.342, 2.967] vs +0.827), overstated the competence gain (+31.15 vs +11.30), and reversed the engagement sign (+0.163 vs âˆ’0.183). The eight-cell SR_eval comparison had MAE 0.655, RMSE 0.842, Pearson r=0.269, and Spearman Ï=0.071. A single-shot LLM appraisal sees the explicit architecture description but does not run the state updates, echo accumulation, or attention gate; it has no mechanism for the engagement cost. The pilot is a methodological triangulation, not human data and not an 18-session simulation.

![Gemini persona-agent pilot](figures/fig_gemini_pilot.png)

*Figure 4. Gemini persona-agent pilot versus the rule-based ABM. The LLM reproduces the direction of the epistemic gain but not its magnitude or the engagement trade-off.*

---

## 6. Limitations and what remains untested

The thesis does not estimate a human causal effect. The 52-student wave is cross-sectional, convenience-based, and non-randomised. The 500 personas are a computational population, not 500 respondents. The synthetic NMLS profile is a measurement diagnostic, not human media literacy. The Hamiltonian is a formal diagnostic, not a physical law; its sign did not decrease. The BRS is a proposed metric, not a validated consequential ranking instrument. Multigroup invariance is not estimated for the v8 state-mediated matrices. The semantic block is contextual, and the content stress test is arbitrary.

The model's narrow intervals describe simulation variation; they are not human sampling intervals. The Bayesian summary is not independent evidence. The model weights are declared assumptions. The field cannot yet identify the dephasing time, the true poor-content share, or the effect of a real LOKLOK deployment. Those are future empirical questions.

---

## 7. Conclusion and defense readiness

The defensible conclusion is that architecture, audience judgment, and measurement should be studied together. EST supplies a falsifiable language; LOKLOK supplies a specified design; the measured wave supplies anchors and limits; the calibrated in-silico experiment shows what the specified design implies. The central finding is the epistemicâ€“engagement trade-off, not a claim that architecture determines human outcomes. The strongest EST propositions remain conditional, and the human causal claim remains future work.

**Three numbers to remember:** +0.827 evaluative rating points, âˆ’0.183 engagement, and 96.5% positive joint-sweep draws.

**Four expected viva questions**

1. *Lagrange-off removes most of the effect, so is this just MORE?* Yes, and that is the design claim: diversity-seeking selection plus a credibility weight.
2. *Shuffling the text gives the same answer, so what did the data do?* They set the level and persona spread, not the contrast.
3. *Why do the default weights sit at the 78th percentile of the sweep?* They are declared design values, and the sweep brackets them.
4. *Why is the baseline gate open only about 15% of the time?* It is a strong attentional filter; the dephasing time is an assumption, not an empirical estimate.

---

## 8. Reproduction and planning

### 8.1 What is in the package

- `thesis_final.pdf` - complete thesis.
- `Supervisor_Summary_EN_AR.pdf` - bilingual review summary.
- `downloads/loklok_code.zip` - pipeline, R scripts, platform reference, and Colab notebooks.
- `downloads/loklok_data.zip` - measured data, raw survey/FGD files, self-report transcript, and model outputs.
- `downloads/loklok_figures.zip` - all figures and plotting data.
- `figures/` - individually browsable figures on this page.

### 8.2 Exact reproduction steps

1. Install Python 3.10/3.11 with NumPy, SciPy, pandas, matplotlib, scikit-learn, and networkx.
2. Install R 4.3 with `lavaan`, `psych`, and `jsonlite` for the optional cross-checks.
3. From the package root, run:
   `python 02_code/pipeline/loklok_pipeline_v8.py --root . --n-personas 500 --sessions 18 --seed 20261105 --embedding-model BAAI/bge-m3 --device cpu`
4. Regenerate the figures:
   `python 02_code/pipeline/generate_figures.py --root .`
5. Optional R checks: `Rscript 02_code/R/01_measured_wave.R`, `Rscript 02_code/R/02_cfa_invariance.R`, `Rscript 02_code/R/03_bayes_crosscheck.R`.
6. Compile the thesis with pdfLaTeX in three passes.
7. Run `python 07_validation/validate_package.py` and confirm 43/43 checks.

### 8.3 Planning logic

The project was organised in five stages: (1) field instrument and 60-to-52 screen; (2) four-stream embedding and debiasing; (3) persona prior and full-factorial ABM; (4) GRM-IRT measurement and dual inference; (5) robustness, ablation, and claim audit. The planning documents used during the project are retained in the thesis archive, but only the thesis and the reproducible artefacts are included in the public download.

### 8.4 What a reviewer should check first

1. The claim boundary: measured vs model-based vs theoretical vs future.
2. The source ablation: semantics are contextual, not the driver.
3. The joint sweep: median and IQR, not only 96.5% positive.
4. The Bayesian comparison: same estimand, not independent evidence.
5. The content and engagement trade-off: the central substantive finding.

---

## 9. Figure atlas

Every thesis figure is included below at web resolution. The complete high-resolution set is in the figure download.

### Process atlas

![Process 1: embedding and debiasing](figures/fig5_process1_embedding_v8.jpg)

*Process 1. Four-stream ingestion, BGE-M3 embedding, DML/IPW, and manifold fusion.*

![Process 2: factorial design](figures/fig5_process2_factorial.jpg)

*Process 2. Within-persona 2Â³ design and G0/G1 governance regimes.*

![Process 3: tensor ABM](figures/fig5_process3_tensor.jpg)

*Process 3. Parallel tensor ABM and the eight influence filters.*

![Process 5: robustness](figures/fig5_process5_robustness_v8.jpg)

*Process 5. Adversarial, sensitivity, and robustness design.*

![Zewail gate](figures/fig6_zewail_diagram.jpg)

*Zewail pumpâ€“probe attentional coherence gate (formal analogy).*

### Embedding and latent space

![Embedding space](figures/fig_embedding_space.png)

![Latent diagnostics](figures/fig_latent_diagnostics.png)

### Main results

![SR distributions](figures/fig6_1_sr_distributions.png)

![Factorial cells](figures/fig6_2_factorial_cells.png)

![Six-dimension radar](figures/fig6_3_radar.png)

![Longitudinal trajectories](figures/fig6_4_longitudinal.png)

![Pareto and Hamiltonian](figures/fig6_5_pareto_hamiltonian.png)

![GRM-IRT diagnostics](figures/fig6_6_grm_irt.png)

![Bayesian posterior](figures/fig6_7_forest.png)

![Debiasing manifold](figures/fig6_8_debias_manifold.png)

![Misinformation stress test](figures/fig6_9_misinfo.png)

![Sobol sensitivity](figures/fig6_10_sobol.png)

![Zewail diagnostics](figures/fig6_11_zewail.png)

![Reliability](figures/fig6_12_reliability.png)

![Frequentist/Bayesian comparison](figures/fig_freq_bayes.png)

![Gemini pilot](figures/fig_gemini_pilot.png)

### Interface and path diagrams

![Field interface](figures/fig_app_interface.png)

![Survey flow](figures/fig_app_survey.png)

![Feynman persona generation](figures/fig_feynman_personas.png)

![Feynman persona path](figures/fig_feynman_paths.png)

![Gemini endpoints](figures/gemini_abm_endpoints.png)

![Gemini cell scatter](figures/gemini_abm_cells.png)

---

## Citation

Rashwan, S. M. A. (2026). *The Social Media Radio Innovation and Its Impact on University Students' Ratings: An Exploratory Study* [Master's thesis]. Ahram Canadian University.

---

## Integrity note

The measured N=52 matrix is never altered to improve significance. Simulated results are never presented as measured human effects. The synthetic NMLS profile is a measurement diagnostic, not human media literacy. The Hamiltonian is a formal diagnostic, not a physical law. The BRS is a proposed metric, not a validated consequential ranking instrument.







