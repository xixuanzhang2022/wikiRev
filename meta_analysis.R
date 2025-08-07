library(meta)

# === Extract Meta-Analysis Summary ===
extract_meta_table <- function(mod) {
  data.frame(
    hr = mod$TE.random,
    cilower = mod$lower.random,
    cihigher = mod$upper.random,
    t = mod$statistic.random,
    pval = mod$pval.random,
    i = mod$I2,
    ilower = mod$lower.I2,
    ihigher = mod$upper.I2,
    q = mod$pval.Q
  )
}

# === Run meta-analysis for one file ===
process_meta <- function(input_file, output_file) {
  data <- read.csv(input_file)
  vart <- data$var[1]

  m.gen <- metagen(
    TE = beta,
    seTE = se,
    studlab = id,
    data = data,
    sm = "HR",
    fixed = FALSE,
    random = TRUE,
    method.tau = "DL",
    method.random.ci = "classic",
    title = vart
  )

  result <- extract_meta_table(m.gen)
  write.csv(result, file = output_file, row.names = FALSE)
}


library(dmetar)

# === Run meta-regression with model averaging ===
metareg <- function(data) {
  mod <- multimodel.inference(
    TE = "beta",
    seTE = "se",
    data = data,
    predictors = c("count", "article_score", "num_u", "astart"),
    interaction = FALSE
  )

  coef_df <- as.data.frame(mod$multimodel)
  coef_df$predictor <- rownames(coef_df)

  importance_df <- mod$predictor.importance
  colnames(importance_df) <- c("predictor", "importance")

  merged_df <- merge(coef_df, importance_df, by = "predictor", all.x = TRUE)
  merged_df <- merged_df[, c("predictor", "Estimate", "Std. Error", "z value", "Pr(>|z|)", "importance")]

  # Format numeric columns
  formatted_df <- merged_df
  formatted_df[] <- lapply(formatted_df, function(x) {
    if (is.numeric(x)) format(round(x, 3), nsmall = 3) else x
  })

  # Significance stars
  merged_df$signif <- with(merged_df, ifelse(`Pr(>|z|)` < 0.001, "***",
                                             ifelse(`Pr(>|z|)` < 0.01, "**",
                                                    ifelse(`Pr(>|z|)` < 0.05, "*",
                                                           ifelse(`Pr(>|z|)` < 0.1, ".", "")))))
  formatted_df$signif <- merged_df$signif

  # Order and label predictors
  desired_order <- c("intrcpt", "astart", "num_u", "count", "article_score")
  formatted_df$predictor <- factor(formatted_df$predictor, levels = desired_order)
  formatted_df <- formatted_df[order(formatted_df$predictor), ]

  rownames(formatted_df) <- c(
    "intercept",
    "time of the first revision",
    "number of unique editors",
    "number of revised sentences",
    "article score"
  )

  formatted_df$predictor <- NULL
  return(formatted_df)
}


# === Paths ===
meta_input_dir <- "/Users/xixuanzhang/Documents/S2/s2_fifi/coxREVnewcUserRevisingStartP/meta_sur/"
meta_output_dir <- "/Users/xixuanzhang/Documents/S2/s2_fifi/coxREVnewcUserRevisingStartP/meta_surm/"
reg_output_dir <- "/Users/xixuanzhang/Documents/S2/s2_fifi/coxREVnewcUserRevisingStartP/meta_regm/"

# === List files ===
csv_files <- list.files(meta_input_dir, pattern = "\\.csv$", full.names = FALSE)

cat("Found files:\n")
print(csv_files)

# === Run meta-analysis and regression ===
for (csv_file in csv_files) {
  input_file <- file.path(meta_input_dir, csv_file)

  # --- Meta-analysis ---
  meta_output_file <- file.path(meta_output_dir, paste0("model_", sub("\\.csv$", "", csv_file), ".csv"))
  process_meta(input_file, meta_output_file)

  # --- Meta-regression ---
  data <- read.csv(input_file)
  reg_table <- metareg(data)
  reg_output_file <- file.path(reg_output_dir, paste0("metareg_", sub("\\.csv$", "", csv_file), ".csv"))
  write.csv(reg_table, file = reg_output_file, row.names = TRUE)
}
