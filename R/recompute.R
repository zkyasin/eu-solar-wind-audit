# Recomputes in R the key statistics of the Python pipeline and compares them with the Python
# results, as the evaluation plan requires (section Product): the monthly modulation volumes, to a
# relative difference of 1e-6, and the coefficients and standard errors of the feed-in and
# consumption signatures, to 1e-3.
#
# Usage:
#   Rscript R/recompute.R --modulation DIR [--modulation DIR ...] [--signature DIR] --output FILE
#
# DIR holds what estimate-modulation, feed-in-signature and consumption-signature write. The
# report lists every comparison; the script exits with status 1 if one fails.
#
# Packages: fixest, jsonlite, nanoparquet.

suppressPackageStartupMessages({
  library(fixest)
  library(jsonlite)
  library(nanoparquet)
})

SUM_TOLERANCE <- 1e-6
REGRESSION_TOLERANCE <- 1e-3
# Relative differences are taken against the Python value, or this one when it is smaller, so that
# values at zero compare in absolute terms.
ZERO <- 1e-9
TIMEZONE <- "Europe/Paris"
MWH_PER_GWH <- 1000

comparison <- function(statistic, python, r, tolerance) {
  difference <- abs(r - python) / pmax(abs(python), ZERO)
  data.frame(statistic, python, r, difference, tolerance, pass = difference <= tolerance)
}

# The months of estimate-modulation (modulation.months): the N hours of each Europe/Paris month,
# the sums of their potential, published output and modulation, and the modulation of the hours
# whose stratum was thin.
modulation_comparisons <- function(directory) {
  paths <- list.files(directory, pattern = "-modulation-hours[.]parquet$", full.names = TRUE)
  do.call(rbind, lapply(paths, function(path) {
    technology <- sub("-modulation-hours[.]parquet$", "", basename(path))
    hours <- read_parquet(path)
    python <- read_parquet(file.path(directory, paste0(technology, "-modulation-months.parquet")))
    month <- format(hours$time, "%Y-%m", tz = TIMEZONE)
    gwh <- function(mwh) tapply(mwh, month, sum, na.rm = TRUE)[python$month] / MWH_PER_GWH
    r <- list(
      negative_hours = as.numeric(table(month)[python$month]),
      potential_gwh = gwh(hours$potential_mwh),
      published_gwh = gwh(hours$published_mwh),
      modulation_gwh = gwh(hours$modulation_mwh),
      thin_stratum_gwh = gwh(ifelse(hours$thin_stratum, hours$modulation_mwh, 0))
    )
    do.call(rbind, lapply(names(r), function(column) {
      statistic <- paste(basename(directory), technology, python$month, column)
      comparison(statistic, python[[column]], as.numeric(r[[column]]), SUM_TOLERANCE)
    }))
  }))
}

# linearmodels scales the clustered sandwich by n / (n - d - k), counting d absorbed effects and
# k regressors; fixest gives the sandwich itself with K.adj and G.adj off. Every hour is kept, as
# in linearmodels: an hour alone in its cell is fitted exactly and adds nothing to the sandwich.
SANDWICH <- ssc(K.adj = FALSE, G.adj = FALSE)

clustered_estimate <- function(fit, weights, absorbed) {
  scale <- nobs(fit) / (nobs(fit) - absorbed - length(coef(fit)))
  c(
    estimate = sum(weights * coef(fit)),
    standard_error = sqrt(scale * drop(weights %*% vcov(fit) %*% weights))
  )
}

# The clear minus overcast contrast of the log ratio, with one effect per cell
# (signature.contrast).
contrast <- function(hours) {
  hours$clear <- as.numeric(hours$sky_class == "clear")
  hours$overcast <- as.numeric(hours$sky_class == "overcast")
  fit <- feols(
    log_ratio ~ clear + overcast | cell, hours,
    cluster = ~day, ssc = SANDWICH, fixef.rm = "none"
  )
  clustered_estimate(fit, c(1, -1), absorbed = length(unique(hours$cell)))
}

# The coefficient of the raw model in the load, with one effect per cell and one per day
# (signature.dip); one effect of the two-way model is redundant.
dip <- function(hours) {
  fit <- feols(
    load_mwh ~ raw_mwh | cell + day, hours,
    cluster = ~day, ssc = SANDWICH, fixef.rm = "none"
  )
  absorbed <- length(unique(hours$cell)) + length(unique(hours$day)) - 1
  clustered_estimate(fit, 1, absorbed)
}

signature_comparisons <- function(directory, name, statistic) {
  path <- file.path(directory, paste0(name, ".parquet"))
  if (!file.exists(path)) {
    return(NULL)
  }
  rows <- read_parquet(path)
  hours <- read_parquet(file.path(directory, paste0(name, "-hours.parquet")))
  do.call(rbind, lapply(seq_len(nrow(rows)), function(i) {
    row <- rows[i, ]
    part <- hours[hours$series == row$series, ]
    label <- paste(name, row$series, row$season)
    if ("min_capacity_factor" %in% names(rows)) {
      part <- part[part$capacity_factor >= row$min_capacity_factor, ]
      label <- paste(label, "from capacity factor", row$min_capacity_factor)
    }
    if (row$season != "all") {
      part <- part[part$season == row$season, ]
    }
    r <- statistic(part)
    rbind(
      comparison(paste(label, "estimate"), row$estimate, r[["estimate"]], REGRESSION_TOLERANCE),
      comparison(
        paste(label, "standard error"), row$standard_error, r[["standard_error"]],
        REGRESSION_TOLERANCE
      )
    )
  }))
}

arguments <- commandArgs(trailingOnly = TRUE)
values <- function(flag) arguments[which(arguments == flag) + 1]
output <- values("--output")
if (length(output) != 1) {
  stop("usage: Rscript R/recompute.R --modulation DIR [--signature DIR] --output FILE")
}
comparisons <- do.call(rbind, c(
  lapply(values("--modulation"), modulation_comparisons),
  lapply(values("--signature"), function(directory) {
    rbind(
      signature_comparisons(directory, "feed-in-signature", contrast),
      signature_comparisons(directory, "consumption-signature", dip)
    )
  })
))
passed <- nrow(comparisons) > 0 && all(comparisons$pass)
write_json(
  list(
    passed = passed,
    largest_difference = list(
      sums = max(c(0, comparisons$difference[comparisons$tolerance == SUM_TOLERANCE])),
      regressions = max(c(0, comparisons$difference[comparisons$tolerance == REGRESSION_TOLERANCE]))
    ),
    comparisons = comparisons
  ),
  output,
  auto_unbox = TRUE, digits = NA, pretty = TRUE
)
quit(status = if (passed) 0 else 1)
