# ============================================================
# 08a2 — motor GBTM 完整 BIC 曲线 (ng=1..5 顺序拟合) + 冻结选类规则
# 背景 (2026-09-03 02:13 实录): 08a 直跳 ng=5 收敛到空类解
#   (share=1944/149/130/526/0, class5=0 人, class3=4.73%<5%,
#    门禁 all_share_gt5pct=FALSE)。13b 原设计是 ng=2..5 顺序拟合——
#   顺序拟合消耗 RNG → gridsearch 初始点不同; 直跳路径落到空类局部最优。
# 处置依据 (SAP §8 同族冻结条款): "触发 ng 敏感性 + 稿件正面披露, 不得静默调类"。
#   motor 主分析地位不变 (STORY 不可妥协#3);
#   ng=5 锁定指 discovery total 层 (12f, BIC 140377→101237 单调降, 不动)。
# 选类规则 (本行在看到曲线前冻结, 防 p-hack):
#   ng∈{2,3,4,5} 中通过可行性门禁 (各类占比>5% 且 各类APP>0.7 无NaN) 者,
#   取 BIC 最低者为 motor 主模型; 全不可行 → stop() 报 PI (ng=3 已知可行, 不应发生)。
# 另: ng=2 完成后打印完整 summary (实证 lcmm 参数化/命名, 供 08b 解析校准);
#     选中模型同样打印。
# 产物: canonical 13b_* (选中模型) + 全曲线 metrics + 08a_gbtm_coefs.json (数组式系数)
# 退化直跳产物 → 13_motor/_deprecated_ng5_direct/
# ============================================================
suppressWarnings(suppressMessages({ library(arrow); library(lcmm); library(jsonlite) }))
ROOT <- "E:/TBI subtype"
OUTD <- file.path(ROOT, "results/cluster/13_motor")
REP  <- file.path(ROOT, "07_prediction_system/reports")
DEP  <- file.path(OUTD, "_deprecated_ng5_direct")
dir.create(DEP, showWarnings = FALSE, recursive = TRUE)
lg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n")
PROG <- file.path(OUTD, "13b_progress.txt")
pf <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n", file = PROG, append = TRUE)

# --- 迁移 08a 退化直跳产物 (存在才移, 保留审计痕) ---
for (f in c("13b_motor_gbtm_fits.rds", "13b_motor_classes.csv", "13b_gbtm_motor_metrics.json")) {
  src <- file.path(OUTD, f)
  if (file.exists(src)) { file.rename(src, file.path(DEP, f)); lg("deprecate:", f) }
}
cj <- file.path(REP, "08a_gbtm_coefs.json")
if (file.exists(cj)) file.copy(cj, file.path(DEP, "08a_gbtm_coefs_degenerate_direct.json"),
                               overwrite = TRUE)

set.seed(20260610)   # 13b 同款: seed 后立即 m1, 再顺序 ng=2..5 (复刻原 RNG 路径)
b <- as.data.frame(read_parquet(file.path(ROOT, "results/features/13_motor_binned6h.parquet")))
ids <- data.frame(stay_id = unique(b$stay_id)); ids$sid <- seq_len(nrow(ids))
b <- merge(b, ids, by = "stay_id"); b$t10 <- b$t / 10
lg("motor GBTM 输入:", nrow(b), "行 /", nrow(ids), "患者")
pf("08A2 sequential motor GBTM n=", nrow(ids))

m1 <- hlme(motor ~ t10 + I(t10^2), random = ~ -1, subject = "sid", ng = 1, data = b)
pf("ng=1 BIC=", round(m1$BIC, 1))

fits <- list("1" = m1); curve <- list()
for (k in 2:5) {
  pf("fitting ng=", k); t0 <- Sys.time()
  mk <- gridsearch(rep = 20, maxiter = 20, minit = m1,
          hlme(motor ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2),
               random = ~ -1, subject = "sid", ng = k, data = b))
  fits[[as.character(k)]] <- mk
  pp <- postprob(mk)
  share <- as.numeric(pp[[1]][1, ]); app <- as.numeric(diag(as.matrix(pp[[2]])))
  viable <- all(share / sum(share) > 0.05) && all(!is.na(app) & app > 0.70)
  curve[[as.character(k)]] <- list(BIC = round(mk$BIC, 1), share = share,
                                   APP = round(app, 3), viable = viable)
  pf(sprintf("ng=%d BIC=%.1f share=%s APP=%s viable=%s 用时=%.0fmin", k, mk$BIC,
             paste(round(share), collapse = "/"), paste(round(app, 3), collapse = "/"),
             viable, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  lg(sprintf("ng=%d BIC=%.1f | share=%s | APP=%s | viable=%s", k, mk$BIC,
             paste(round(share), collapse = "/"), paste(round(app, 3), collapse = "/"), viable))
  if (k == 2) { lg("=== ng=2 完整 summary (参数化/命名实证, 供 08b 解析校准) ===")
                print(summary(mk)) }
}

# --- 冻结选类规则执行 ---
vn <- names(curve)[sapply(curve, function(x) isTRUE(x$viable))]
if (length(vn) == 0) { pf("08A2_NO_VIABLE"); stop("08a2: 无可行 ng — 停止报 PI") }
sel <- as.integer(vn[which.min(sapply(curve[vn], function(x) x$BIC))])
bestfit <- fits[[as.character(sel)]]
lg(sprintf("选类: 可行 ng={%s} → 选中 ng=%d (BIC=%.1f)",
           paste(vn, collapse = ","), sel, bestfit$BIC))
pf(sprintf("SELECTED ng=%d BIC=%.1f", sel, bestfit$BIC))
if (sel == 5) lg("注: 顺序路径 ng=5 可行; 与 08a 直跳退化解并存, 稿件披露两者")

# --- 选中模型 summary + 表型画像 ---
lg(sprintf("=== selected ng=%d summary ===", sel)); print(summary(bestfit))
cls <- bestfit$pprob[, c("sid", "class")]; cls <- merge(cls, ids, by = "sid")
coh <- as.data.frame(read_parquet(file.path(ROOT, "results/cohort_audit/02_tbi_cohort.parquet")))[, c("stay_id", "days_to_death")]
g72 <- as.data.frame(read_parquet(file.path(ROOT, "results/features/13_motor_features.parquet")))[, c("stay_id", "motor72_first", "motor72_last", "motor72_delta")]
o <- Reduce(function(a, bb) merge(a, bb, by = "stay_id"), list(cls, coh, g72))
o$d28 <- as.integer(!is.na(o$days_to_death) & pmax(o$days_to_death, 0) <= 28)
prof <- aggregate(o[, c("motor72_first", "motor72_last", "motor72_delta", "d28")],
                  list(表型 = o$class), function(x) round(mean(x, na.rm = TRUE), 3))
prof$n <- as.integer(table(o$class))
print(prof, row.names = FALSE)

# --- canonical 落盘 (全部 ng fits 存档供 §8 Bootstrap 复用) ---
saveRDS(fits, file.path(OUTD, "13b_motor_gbtm_fits.rds"))
write.csv(o[, c("stay_id", "class")], file.path(OUTD, "13b_motor_classes.csv"), row.names = FALSE)
write_json(list(
  n = nrow(ids),
  BIC_by_ng = c("1" = round(m1$BIC, 1), sapply(curve, function(x) x$BIC)),
  selected_ng = sel,
  selection_rule = "viable = 各类占比>5% 且 各类APP>0.7; 可行者取 BIC 最低 (规则冻结于拟合前; SAP §8 同族条款: ng 敏感性 + 正面披露, 不得静默调类)",
  curve = curve,
  degenerate_direct_ng5 = list(
    note = "2026-09-03 08a 直跳 ng=5 得空类解(class5=0), 产物移 _deprecated_ng5_direct/",
    BIC = 53402.8, share = c(1944, 149, 130, 526, 0),
    APP = c(0.727, 0.997, 0.980, 0.957, NA)),
  phenotype_profile = lapply(seq_len(nrow(prof)), function(i) as.list(prof[i, ]))),
  file.path(OUTD, "13b_gbtm_motor_metrics.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")

# --- 系数冻结 (数组式, 保重复名): motor 选中 + total ng5 (12f) ---
dump_fit2 <- function(fit) {
  nm <- names(fit$best); vv <- as.numeric(fit$best)
  lg("coef entries:", length(nm), "个")
  list(BIC = round(fit$BIC, 1),
       best_vec = lapply(seq_along(nm), function(i) list(name = nm[i], value = vv[i])))
}
tot_fits <- readRDS(file.path(ROOT, "results/cluster/12_gcs_course/12f_gbtm_fits.rds"))
tot5 <- tot_fits[["5"]]
tot_cls <- read.csv(file.path(ROOT, "results/cluster/12_gcs_course/12f_gbtm_classes.csv"))
coefs <- list(
  generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  spec = list(formula = "y ~ t10 + I(t10^2), mixture = ~ t10 + I(t10^2), random = ~ -1 (random intercept), t10 = t/10, binned 6h grid, seed 20260610",
              var_time = "t10", resp_scale = "motor 0-6 / total 3-15",
              selection = "motor ng 由 08a2 冻结规则选出 (见 13b_gbtm_motor_metrics.json); total ng=5 锁定 (12f)"),
  motor = list(n = nrow(ids), ng = sel,
               class_share = curve[[as.character(sel)]]$share,
               APP = round(curve[[as.character(sel)]]$APP, 3),
               fit = dump_fit2(bestfit)),
  total = list(n = nrow(tot_cls), ng = 5,
               class_share = as.integer(table(tot_cls$class)),
               fit = dump_fit2(tot5)))
write_json(coefs, file.path(REP, "08a_gbtm_coefs.json"),
           pretty = TRUE, auto_unbox = TRUE, digits = NA, na = "null")

pf("08A2_DONE"); lg("08A2_MOTOR_GBTM_SELECT_DONE")
