# ---------------------------------------------------------------------------
# 01_measured_wave.R
# Re-derive the measured-wave descriptives, reliability, and associations
# from the de-identified 52-row matrix.
#
# Run from the package root:
#   Rscript 02_code/R/01_measured_wave.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(jsonlite)
})

root <- getwd()
csv_path <- file.path(root, "03_data", "measured", "respondents_52.csv")
if (!file.exists(csv_path)) {
  stop("Cannot find 03_data/measured/respondents_52.csv. Run from package root.")
}
df <- read.csv(csv_path, check.names = FALSE)

br_items <- c("BR1", "BR2", "BR3", "BR4")
sv_items <- c("SV1", "SV2", "SV3", "SV4")
iq_items <- c("IQ1", "IQ2", "IQ3")
pb_items <- c("PB1", "PB2", "PB3", "PB4", "PB5")

df$BR4r <- if ("BR4_reversed" %in% names(df)) df$BR4_reversed else 4 - df$BR4
df$BR_mean <- rowMeans(df[, c("BR1", "BR2", "BR3", "BR4r")], na.rm = TRUE)
df$SV_mean <- rowMeans(df[, sv_items], na.rm = TRUE)
df$IQ_mean <- rowMeans(df[, iq_items], na.rm = TRUE)
df$PB_mean <- rowMeans(df[, pb_items], na.rm = TRUE)
df$BRS_total <- rowSums(df[, c("BR1", "BR2", "BR3", "BR4r", sv_items, iq_items)], na.rm = TRUE)

alpha_for <- function(X) {
  X <- as.matrix(X)
  k <- ncol(X)
  item_vars <- apply(X, 2, var, na.rm = TRUE)
  total_var <- var(rowSums(X, na.rm = TRUE), na.rm = TRUE)
  (k / (k - 1)) * (1 - sum(item_vars) / total_var)
}

out <- list(
  n = nrow(df),
  means = list(
    BR = mean(df$BR_mean), SV = mean(df$SV_mean), IQ = mean(df$IQ_mean),
    PB = mean(df$PB_mean), star = mean(df$star_rating),
    duration = mean(df$ride_duration_minutes)
  ),
  sds = list(
    BR = sd(df$BR_mean), SV = sd(df$SV_mean), IQ = sd(df$IQ_mean),
    PB = sd(df$PB_mean), star = sd(df$star_rating),
    duration = sd(df$ride_duration_minutes)
  ),
  alpha = list(
    BRS = alpha_for(df[, c("BR1", "BR2", "BR3", "BR4r", sv_items, iq_items)]),
    BR = alpha_for(df[, c("BR1", "BR2", "BR3", "BR4r")]),
    SV = alpha_for(df[, sv_items]),
    IQ = alpha_for(df[, iq_items]),
    PB = alpha_for(df[, pb_items])
  ),
  correlations = list(
    PB_star_spearman = cor(df$PB_mean, df$star_rating, method = "spearman"),
    PB_duration_spearman = cor(df$PB_mean, df$ride_duration_minutes, method = "spearman"),
    PB_interaction_spearman = cor(df$PB_mean, df$interaction_ordinal, method = "spearman"),
    BR_PB_spearman = cor(df$BR_mean, df$PB_mean, method = "spearman"),
    SV_PB_spearman = cor(df$SV_mean, df$PB_mean, method = "spearman"),
    IQ_PB_spearman = cor(df$IQ_mean, df$PB_mean, method = "spearman"),
    BR_IQ_pearson = cor(df$BR_mean, df$IQ_mean, method = "pearson"),
    SV_IQ_pearson = cor(df$SV_mean, df$IQ_mean, method = "pearson")
  ),
  interaction_counts = as.list(table(df$interaction_level)),
  university_counts = as.list(table(df$university)),
  provenance = list(
    source = "03_data/measured/respondents_52.csv",
    note = "MEASURED field wave; N=52 after the historical 21-field duplicate screen."
  )
)

out_dir <- file.path(root, "03_data", "measured", "analysis")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write_json(out, file.path(out_dir, "measured_wave_R.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Wrote", file.path(out_dir, "measured_wave_R.json"), "\n")
