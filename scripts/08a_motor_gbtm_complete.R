# ============================================================
# 08a — Phase 2 Step 0: 补齐 motor GBTM(ng=5) 并冻结双模型系数
# 背景: 13b 首跑死于 ng=4 拟合中途 (progress.txt 铁证, 无产物落盘);
#       STORY 不可妥协#3 (motor 主分析) + SAP §3.3 Block C 要求 motor 截断后验。
# 本脚本 = 13b 冻结 spec 的 ng=1 + ng=5 直接拟合 (跳过 ng2-4, 选类规则已冻结 ng=5
#       "与12f可比"), 并把 motor ng=5 与 12f total ng=5 的完整系数导出 JSON
#       供 Python 截断后验引擎 (08b) 使用。
# 验证锚 (08b 侧): Python 引擎全 72h 观测后验 argmax vs 本脚本 classes CSV ≥95% 一致
# ============================================================
suppressWarnings(suppressMessages({ library(arrow); library(lcmm); library(jsonlite) }))
ROOT <- "E:/TBI subtype"
OUTD <- file.path(ROOT, "results/cluster/13_motor")
REP  <- file.path(ROOT, "07_prediction_system/reports")
dir.create(OUTD, showWarnings = FALSE, recursive = TRUE)
lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n")
PROG <- file.path(OUTD, "13b_progress.txt")
pf <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n", file = PROG, append = TRUE)

set.seed(20260610)   # 13b 原脚本顶部种子, 顺序保持一致
b <- as.data.frame(read_parquet(file.path(ROOT, "results/features/13_motor_binned6h.parquet")))
ids <- data.frame(stay_id = unique(b$stay_id)); ids$sid <- seq_len(nrow(ids))
b <- merge(b, ids, by = "stay_id"); b$t10 <- b$t / 10
lg("motor GBTM 输入:", nrow(b), "行 /", nrow(ids), "患者"); pf("08a RESTART motor GBTM n=", nrow(ids))

# --- ng=1 (m1, gridsearch 的 minit) ---
m1 <- hlme(motor ~ t10 + I(t10^2), random = ~ -1, subject = "sid", ng = 1, data = b)
pf("ng=1 BIC=", round(m1$BIC, 1)); lg("ng=1 BIC=", round(m1$BIC, 1))

# --- ng=5 (锁定, 与 12f 同构 gridsearch) ---
pf("fitting ng=5")
t0 <- Sys.time()
m5 <- gridsearch(rep = 20, maxiter = 20, minit = m1,
        hlme(motor ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2),
             random = ~ -1, subject = "sid", ng = 5, data = b))
pp <- postprob(m5)
share <- round(as.numeric(pp[[1]][1, ]), 0)
app   <- round(diag(as.matrix(pp[[2]])), 3)
pf(sprintf("ng=5 BIC=%.1f 占比=%s APP=%s 用时=%.0fmin", m5$BIC,
           paste(share, collapse = "/"), paste(app, collapse = "/"),
           as.numeric(difftime(Sys.time(), t0, units = "mins"))))
lg(sprintf("ng=5 BIC=%.1f | share=%s | APP=%s", m5$BIC,
           paste(share, collapse = "/"), paste(app, collapse = "/")))

# --- 门禁 (SAP 选类规则: BIC<ng1 且 各类>5% 且 APP>0.7) ---
gate <- list(BIC_lt_ng1 = m5$BIC < m1$BIC,
             all_share_gt5pct = all(share / sum(share) > 0.05),
             all_APP_gt0.7 = all(app > 0.7))
lg("gate:", paste(names(gate), gate, sep = "=", collapse = " "))

# --- 表型画像 (13b 原设计: motor72 特征 + d28) ---
cls <- m5$pprob[, c("sid", "class")]; cls <- merge(cls, ids, by = "sid")
coh <- as.data.frame(read_parquet(file.path(ROOT, "results/cohort_audit/02_tbi_cohort.parquet")))[, c("stay_id", "days_to_death")]
g72 <- as.data.frame(read_parquet(file.path(ROOT, "results/features/13_motor_features.parquet")))[, c("stay_id", "motor72_first", "motor72_last", "motor72_delta")]
o <- Reduce(function(a, bb) merge(a, bb, by = "stay_id"), list(cls, coh, g72))
o$d28 <- as.integer(!is.na(o$days_to_death) & pmax(o$days_to_death, 0) <= 28)
prof <- aggregate(o[, c("motor72_first", "motor72_last", "motor72_delta", "d28")],
                  list(表型 = o$class), function(x) round(mean(x, na.rm = TRUE), 3))
prof$n <- as.integer(table(o$class))
print(prof, row.names = FALSE)

# --- 落盘: fits rds + classes csv + metrics json (13b 原定产物名) ---
saveRDS(list("1" = m1, "5" = m5), file.path(OUTD, "13b_motor_gbtm_fits.rds"))
write.csv(o[, c("stay_id", "class")], file.path(OUTD, "13b_motor_classes.csv"), row.names = FALSE)
write_json(list(n = nrow(ids), BIC_ng1 = round(m1$BIC, 1), BIC_ng5 = round(m5$BIC, 1),
  class_share = share, APP = app, gate = gate,
  phenotype_profile = lapply(seq_len(nrow(prof)), function(i) as.list(prof[i, ]))),
  file.path(OUTD, "13b_gbtm_motor_metrics.json"), pretty = TRUE, auto_unbox = TRUE)

# --- 系数冻结: motor ng5 + total ng5 (12f rds) 全系数 → JSON 给 08b Python 引擎 ---
dump_fit <- function(fit, note) {
  best <- fit$best
  lg(note, "coef names:", paste(names(best), collapse = " ; "))
  list(BIC = round(fit$BIC, 1), n_obs_per_class = NULL,
       best = as.list(stats::setNames(as.numeric(best), names(best))))
}
tot_fits <- readRDS(file.path(ROOT, "results/cluster/12_gcs_course/12f_gbtm_fits.rds"))
tot5 <- tot_fits[["5"]]
tot_cls <- read.csv(file.path(ROOT, "results/cluster/12_gcs_course/12f_gbtm_classes.csv"))
coefs <- list(
  generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  spec = list(formula = "y ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2), random = ~ -1 (random intercept), t10 = t/10, binned 6h grid, seed 20260610",
              var_time = "t10", resp_scale = "motor 0-6 / total 3-15"),
  motor = list(n = nrow(ids), ng = 5, class_share = share, APP = app,
               fit = dump_fit(m5, "[motor ng5]")),
  total = list(n = nrow(tot_cls), ng = 5,
               class_share = as.integer(table(tot_cls$class)),
               fit = dump_fit(tot5, "[total ng5]")))
write_json(coefs, file.path(REP, "08a_gbtm_coefs.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = NA)

pf("08A_DONE"); lg("08A_MOTOR_GBTM_DONE")
