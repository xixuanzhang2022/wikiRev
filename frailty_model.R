library(coxme)

# === CONFIGURATION ===
folder_path <- "/Users/xixuanzhang/Documents/S2/s2_fifi/coxREVnewcUserRevisingStartP/coxraw"
output_folder <- "/Users/xixuanzhang/Documents/S2/s2_fifi/coxREVnewcUserRevisingStartP/cox_surv"
log_file <- file.path(output_folder, "error_log.txt")

extract_random_effects <- FALSE  # Set to TRUE to extract random effect variance (vcoef$sentence)

# Create output directory if it does not exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# === Model extractors ===
extract_fixed_effects <- function(mod) {
  beta <- mod$coefficients
  nvar <- length(beta)
  nfrail <- nrow(mod$var) - nvar
  se <- sqrt(diag(mod$var)[nfrail + 1:nvar])
  z <- round(beta / se, 2)
  p <- signif(1 - pchisq((beta / se)^2, 1), 2)
  data.frame(beta = beta, se = se, z = z, p = p)
}

extract_random_effects <- function(mod) {
  data.frame(var = mod$vcoef$sentence)
}

# === Model runner ===
process_data <- function(f, out_file) {
  tryCatch({
    data <- read.csv(f)
    converged <- TRUE

    fit <- withCallingHandlers(
      coxme(Surv(prev.ivl, event) ~ start.prev + score + sstart + parity +
              v1state + v2state + ustate + intensity + (1 | sentence),
            data = data),
      warning = function(w) {
        if (grepl("did not converge", conditionMessage(w))) {
          cat("Skipping model due to convergence issue:", f, "\n")
          converged <<- FALSE
          invokeRestart("muffleWarning")
        }
      }
    )

    if (converged) {
      results <- if (extract_random_effects) extract_random_effects(fit) else extract_fixed_effects(fit)
      write.csv(results, file = out_file, row.names = TRUE)
      cat("Model fit successfully and saved to:", out_file, "\n")
    }
  }, error = function(e) {
    cat("Error encountered while processing:", f, "\n")
    cat("Error message:", e$message, "\n")
    write(paste("Error in file:", f, "Message:", e$message), file = log_file, append = TRUE)
  })
}

# === Batch run ===
all_files <- list.files(folder_path, pattern = "\\.csv$", full.names = TRUE)
cat("Found", length(all_files), "CSV files.\n")

for (f in all_files) {
  out_file <- file.path(output_folder, paste0("model_", tools::file_path_sans_ext(basename(f)), ".csv"))
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "- Processing:", basename(f), "\n")
  process_data(f, out_file)
}
