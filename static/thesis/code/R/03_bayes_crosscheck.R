# ---------------------------------------------------------------------------
# 03_bayes_crosscheck.R
# Base-R Gibbs cross-check of the hierarchical model contrast.  This is a
# reproduction aid, not the primary analysis (the Python sampler is primary).
#
# Run from the package root:
#   Rscript 02_code/R/03_bayes_crosscheck.R
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(jsonlite)
})

root <- getwd()
path <- file.path(root, "03_data", "model_outputs", "persona_deltas.csv")
if (!file.exists(path)) stop("Run the Python pipeline first so persona_deltas.csv exists.")
d <- read.csv(path, check.names = FALSE)
y <- as.numeric(d$delta_sr_eval)
g <- as.integer(d$group)
G <- max(g) + 1
n <- length(y)

set.seed(20261105)
n_iter <- 8000
burn <- 2000

tau <- mean(y)
alpha <- rep(0, G)
mu_a <- 0
tau_a <- 1
sigma2 <- var(y)
draws_tau <- numeric(n_iter)

for (it in seq_len(n_iter)) {
  for (k in seq_len(G)) {
    idx <- which(g == (k - 1))
    ng <- length(idx)
    if (ng == 0) {
      alpha[k] <- rnorm(1, mu_a, tau_a)
    } else {
      prec <- ng / sigma2 + 1 / tau_a^2
      mn <- (sum(y[idx] - tau) / sigma2 + mu_a / tau_a^2) / prec
      alpha[k] <- rnorm(1, mn, 1 / sqrt(prec))
    }
  }
  resid <- y - alpha[g + 1]
  prec <- n / sigma2 + 1 / 1^2
  mn <- (sum(resid) / sigma2 + 0 / 1^2) / prec
  tau <- rnorm(1, mn, 1 / sqrt(prec))

  prec <- G / tau_a^2 + 1
  mn <- (sum(alpha) / tau_a^2) / prec
  mu_a <- rnorm(1, mn, 1 / sqrt(prec))

  ss <- sum((y - alpha[g + 1] - tau)^2)
  sigma2 <- 1 / rgamma(1, shape = 2 + n / 2, rate = 2 + ss / 2)
  draws_tau[it] <- tau
}

post <- draws_tau[(burn + 1):n_iter]
hdi <- function(x, prob = 0.95) {
  x <- sort(x)
  n <- length(x)
  w <- max(1, floor(prob * n))
  widths <- x[(w + 1):n] - x[1:(n - w)]
  i <- which.min(widths)
  c(x[i], x[i + w])
}
ci <- hdi(post, 0.95)

# Savage-Dickey BF10 with a weakly informative N(0,1) prior
dens <- density(post, n = 4096)
post0 <- approx(dens$x, dens$y, xout = 0)$y
prior0 <- dnorm(0, 0, 1)
bf10 <- min(prior0 / max(post0, 1e-12), 1e4)

out <- list(
  n = n,
  tau_mean = mean(post),
  tau_sd = sd(post),
  tau_hdi_low = ci[1],
  tau_hdi_high = ci[2],
  prob_positive = mean(post > 0),
  bf10 = bf10,
  sigma2_mean = mean(sigma2),
  seed = 20261105,
  note = "Base-R cross-check of the Python hierarchical model; not the primary analysis."
)

out_dir <- file.path(root, "03_data", "model_outputs")
write_json(out, file.path(out_dir, "bayes_r_results.json"), pretty = TRUE, auto_unbox = TRUE)
cat("Wrote", file.path(out_dir, "bayes_r_results.json"), "\n")
