# LOKLOK

## The Social Media Radio Innovation and Its Impact on University Students' Ratings: An Exploratory Study

A master's thesis and its complete web companion: Epistemic Systems Theory, the LOKLOK platform, a measured student wave, and a calibrated in-silico experiment, published as a classical academic website with an interactive laboratory.

**Author:** Sherif Mohamed Abd Elmonem Rashwan
**Degree:** M.A. Defense, Media Innovations, Faculty of Mass Communication, Ahram Canadian University
**Supervisors:** Dr. Engi Aboul-Ezz (Beni Suef University) and Dr. Mostafa El Banna (Ahram Canadian University)
**Date:** November 2026

| Resource | Link |
| --- | --- |
| Live thesis website | [srashwan.github.io/loklok/thesis](https://srashwan.github.io/loklok/thesis/) |
| Read page by page | [thesis/read](https://srashwan.github.io/loklok/thesis/read/) |
| Interactive laboratory | [thesis/interactive](https://srashwan.github.io/loklok/thesis/interactive/) |
| Full thesis PDF (366 pages) | [thesis_final.pdf](https://srashwan.github.io/loklok/thesis/thesis_final.pdf) |
| Bilingual supervisor summary | [Supervisor_Summary_EN_AR.pdf](https://srashwan.github.io/loklok/thesis/Supervisor_Summary_EN_AR.pdf) |
| Defense presentation | [PDF](https://srashwan.github.io/loklok/thesis/LOKLOK_Defense.pdf) / [PPTX](https://srashwan.github.io/loklok/thesis/LOKLOK_Defense.pptx) / [HTML](https://srashwan.github.io/loklok/thesis/LOKLOK_Defense.html) |
| Feynman-diagram supplement | [Feynman_Supplementary.pdf](https://srashwan.github.io/loklok/thesis/Feynman_Supplementary.pdf) |
| Machine-readable thesis data | [data/thesis_data.json](https://srashwan.github.io/loklok/thesis/data/thesis_data.json) |

---

## What this thesis asks

Platform design, audience measurement, and epistemic quality are usually studied in
separate literatures. This thesis joins them with one question: **can a social media
radio platform be designed so that university students judge what they hear more
carefully, without losing the audience?**

Because no live trial was possible, the evidence is organized in three layers with a
strict claim hierarchy:

1. **The measured layer.** A questionnaire from 52 students (60 collected, 8 duplicate
   vectors removed) riding Uber in 6th of October City, Giza. Perceived benefit 5.12
   (SD 0.87), correlation with the star rating rho = .67, reliability alpha = .96 and
   .95, and a poor single-factor CFA fit, so no structural validity is claimed. This
   layer anchors the model; it is not a causal test.
2. **The qualitative layer.** A focus-group transcript and a researcher self-report,
   193 usable segments coded by three independent human coders on a 25-code codebook
   (pairwise kappa .762, .852, .861). Machine coding (Qwen 7B micro-F1 .193 zero-shot,
   .320 with gold exemplars; Claude first pass .452) met the pre-set usability gate for
   no mechanism code, so this layer is an assumption audit, reported honestly as a null
   result.
3. **The modelling layer.** A calibrated in-silico experiment: 500 synthetic personas
   drawn from the debiased distribution of the 52 respondents (DML + IPW, spectral
   manifold), run through 2^3 = 8 design cells over 18 listening sessions with eight
   mathematical filters and a Zewail-inspired attention gate, scored by a GRM-IRT
   measurement model with the G0 arm anchored to the measured mean.

## Headline results

The epistemic-first configuration G1 against the engagement-first status quo G0,
500 personas, 18 sessions:

| Outcome | Estimate | 95% interval | t(499) | Cohen's d |
| --- | --- | --- | --- | --- |
| Evaluative student rating (1-7) | +0.827 | [0.779, 0.875] | 33.93 | 1.52 |
| Competence-format rating (0-100) | +11.30 | [10.64, 11.95] | 33.74 | 1.51 |
| Engagement (0-1) | -0.183 | [-0.188, -0.178] | -74.41 | -3.33 |

The central finding is a trade-off, not universal superiority: the diversity and
cooling path (B) carries the contrast (main effect +0.6265), the reputation path (A)
adds +0.1905, moderation (C) is near zero alone but contributes through the B x C
interaction (+0.3791). Across 256 draws of the declared weights the contrast is
positive in 96.5% of cases (median +0.567, range -0.244 to +1.810) and depends
chiefly on one declared link: the diversity-to-credibility weight (PRCC +0.77).

A randomized-assignment triangulation re-ran the same eight cells under four designs:

| Design | Worlds | Delta SR eval | Delta engagement |
| --- | --- | --- | --- |
| Within-persona | 1 | +0.827 | -0.217 |
| Agent-randomized | 40 | +0.630 [0.384, 0.851] | -0.232 |
| Design-based (no interference) | 400 | +0.807 [0.470, 1.109] | -0.217 |
| Cluster-randomized | 20 | +0.659 [0.338, 0.948] | -0.226 |

The diversity component and the engagement cost survive every design; the BRS
component does not (A collapses from +0.190 to +0.038 between agents and recovers to
+0.182 in the interference-free design). Internal controls pass: placebo factor D
gives -0.003, mechanism checks move in the declared directions, and topologies
(ER/WS/BA) leave the contrast at +0.827/+0.826/+0.832.

Every model-based number on this site is conditional on the stated design and is
**not a human treatment effect**.

## What the site contains

The thesis section is a hand-built static site in a classical academic design
(Cornell and MIT OpenCourseWare conventions): serif display typography, a crimson
and paper palette, small-caps section kickers, double rules, and a module sidebar.

- **Overview** (`thesis/index.html`): the argument, headline tables, figure gallery,
  documents.
- **Abstract** (`thesis/abstract.html`): the full abstract with section anchors.
- **Reading guide** (`thesis/guide.html`): a one-hour short path, a module map with
  page numbers, claim tiers, and notation.
- **Results** (`thesis/results.html`): every table and all 12 figures with
  captions.
- **Downloads** (`thesis/downloads.html`): the manuscript, summaries, defense deck,
  and supplements.
- **Read online** (`thesis/read/`): all 366 pages rendered 1:1 from the submitted PDF,
  with a searchable table of contents, chapter jumps, keyboard navigation, and a
  progress bar.

### The interactive laboratory

Five dependency-free applets reproduce the thesis arithmetic in the browser:

1. **Trade-off explorer** (`interactive/tradeoff.html`): sliders for the declared
   state-outcome weights, a sampled Pareto frontier, the 256-draw sweep band, and a
   percentile readout.
2. **BRS diffusion simulator** (`interactive/brs.html`): a seven-agent live
   reputation network; toggles the BRS path and the assignment design to show why the
   effect is interference-sensitive.
3. **The eight cells** (`interactive/cells.html`): the full 2^3 factorial with
   click-to-read compositions and a four-design comparison view.
4. **Feynman diagrams** (`interactive/feynman.html`): the ten experimental processes
   in particle-physics notation with every line and vertex explained on click.
5. **The codebook examiner** (`interactive/codebook.html`): all 25 qualitative codes
   with human agreement bars, prevalence dots, and the pre-set usability gate.

Every applet labels its inputs as declared assumptions and its outputs as
illustrative; the exact numbers come from the 18-session simulation in the thesis.

## Repository structure

```text
.
├── .github/workflows/hugo.yml        # Hugo build + GitHub Pages deployment
├── config.yml                        # Hugo configuration and navigation
├── content/                          # Personal pages (profile, books, papers, data)
├── layouts/                          # Hugo templates (PaperMod theme overrides)
├── static/
│   ├── thesis/
│   │   ├── index.html                # Overview
│   │   ├── abstract.html             # Abstract
│   │   ├── guide.html                # Reading guide and module map
│   │   ├── results.html              # Results and figures
│   │   ├── downloads.html            # Materials
│   │   ├── css/thesis.css            # Classical academic design system
│   │   ├── js/site.js                # Reveal animations, counters, data helper
│   │   ├── data/thesis_data.json     # All reported numbers, machine-readable
│   │   ├── interactive/              # The five applets
│   │   ├── read/                     # Page-by-page reader (366 pages)
│   │   ├── pages/                    # 1:1 page renders (WebP) + manifest.json
│   │   ├── figures/                  # Thesis figures and Feynman PNGs
│   │   └── *.pdf, *.pptx             # Manuscript, summaries, deck, supplement
│   └── ...                           # Site-wide assets (images, icons)
└── README.md
```

## Building and deploying

The site is a Hugo 0.138 site with the PaperMod theme; the thesis section is plain
HTML under `static/thesis/` and is copied verbatim at build time.

```bash
# local preview
git clone https://github.com/SRashwan/loklok.git
cd loklok
hugo server

# production build
hugo --minify
```

Pushing to `main` triggers `.github/workflows/hugo.yml`: GitHub Actions installs
Hugo, builds the site into `public/`, uploads it as a Pages artifact, and deploys it
to `https://srashwan.github.io/loklok/`. The `public/` directory is generated and is
not tracked in git.

The page images in `static/thesis/pages/` are rendered from `thesis_final.pdf` with
PyMuPDF at a width of 1000 px; `manifest.json` stores the page count and the table of
contents used by the reader.

## Quality and honesty commitments

- The manuscript is 366 pages with 0 em dashes, no overfull boxes, and no undefined
  references, compiled with pdfLaTeX in three passes.
- The reproducibility package passes 43 of 43 automated checks and ships a SHA-256
  manifest.
- The Arabic front matter and the Arabic supervisor summary are maintained
  alongside the English text; the bilingual summary is 18 pages.
- All figures link back to their source CSVs; the site data JSON is generated from
  the validated outputs, not hand-entered.
- The qualitative machine-coding result is reported as the failure it is; the gate
  was not moved to make it pass.
- Everywhere on the site, measured field evidence, model-based counterfactual
  evidence, theoretical propositions, and prospective plans are kept in separate
  claim tiers.

## Citation

```text
Rashwan, S. M. A. E. (2026). The Social Media Radio Innovation and Its Impact on
University Students' Ratings: An Exploratory Study. Master's thesis, Ahram Canadian
University, Faculty of Mass Communication, Egypt.
```

## Contact

Sherif Mohamed Abd Elmonem Rashwan, Teaching Assistant, Radio and Television
Department, Faculty of Mass Communication, Ahram Canadian University.
Email: sherif.rashwan@san3atech.com

---

Copyright (c) 2026 Sherif Mohamed Abd Elmonem Rashwan. All model-based numbers are
conditional on the stated design and are not human treatment effects.
