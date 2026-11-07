# ---------------------------------------------------------------------------
# 02_cfa_invariance.R
# Lavaan CFA and configural / metric / scalar invariance for the synthetic
# NMLS item batteries exported by the Python pipeline.
#
# Run from the package root:
#   Rscript 02_code/R/02_cfa_invariance.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(lavaan)
  library(jsonlite)
})

root <- getwd()
g0_path <- file.path(root, "03_data", "model_outputs", "nmls_g0_items.csv")
g1_path <- file.path(root, "03_data", "model_outputs", "nmls_g1_items.csv")
if (!file.exists(g0_path) || !file.exists(g1_path)) {
  stop("Run the Python pipeline first so the NMLS item exports exist.")
}

item_names <- sprintf("NMLS%02d", 1:35)
read_items <- function(path, group) {
  d <- read.csv(path, check.names = FALSE)
  names(d) <- item_names
  d$group <- group
  d
}
d <- rbind(read_items(g0_path, 0), read_items(g1_path, 1))

model <- paste(
  "FC =~", paste(item_names[1:7], collapse = " + "),
  "\nCC =~", paste(item_names[8:18], collapse = " + "),
  "\nFP =~", paste(item_names[19:25], collapse = " + "),
  "\nCP =~", paste(item_names[26:35], collapse = " + ")
)

fit_one <- function(equal = NULL) {
  args <- list(
    model = model,
    data = d,
    group = "group",
    estimator = "WLSMV",
    ordered = item_names,
    std.lv = TRUE,
    missing = "pairwise"
  )
  if (!is.null(equal)) args$group.equal <- equal
  do.call(cfa, args)
}

safe_fit <- function(equal = NULL) {
  tryCatch(
    fit_one(equal),
    error = function(e) {
      args <- list(
        model = model,
        data = d,
        group = "group",
        estimator = "MLR",
        std.lv = TRUE,
        missing = "pairwise"
      )
      if (!is.null(equal)) args$group.equal <- equal
      do.call(cfa, args)
    }
  )
}

get_fit <- function(fit) {
  fm <- fitMeasures(fit, c("cfi", "cfi.robust", "tli", "tli.robust", "rmsea", "rmsea.robust", "srmr"))
  # prefer robust values when available
  cfi <- ifelse(!is.na(fm["cfi.robust"]), fm["cfi.robust"], fm["cfi"])
  tli <- ifelse(!is.na(fm["tli.robust"]), fm["tli.robust"], fm["tli"])
  rmsea <- ifelse(!is.na(fm["rmsea.robust"]), fm["rmsea.robust"], fm["rmsea"])
  list(CFI = as.numeric(cfi), TLI = as.numeric(tli), RMSEA = as.numeric(rmsea), SRMR = as.numeric(fm["srmr"]))
}

cat("Fitting configural model ...\n")
fit_config <- safe_fit(NULL)
cat("Fitting metric model ...\n")
fit_metric <- safe_fit("loadings")
cat("Fitting scalar model ...\n")
fit_scalar <- safe_fit(c("loadings", "thresholds"))

fc <- get_fit(fit_config)
fm <- get_fit(fit_metric)
fs <- get_fit(fit_scalar)

out <- list(
  model = model,
  n_g0 = sum(d$group == 0),
  n_g1 = sum(d$group == 1),
  configural = fc,
  metric = c(fm, list(delta_cfi = fm$CFI - fc$CFI, delta_rmsea = fm$RMSEA - fc$RMSEA)),
  scalar = c(fs, list(delta_cfi = fs$CFI - fm$CFI, delta_rmsea = fs$RMSEA - fm$RMSEA)),
  note = "Synthetic GRM item responses; this CFA audits the generating model, not human NMLS validity."
)

out_dir <- file.path(root, "03_data", "model_outputs")
write_json(out, file.path(out_dir, "cfa_r_results.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Wrote", file.path(out_dir, "cfa_r_results.json"), "\n")
