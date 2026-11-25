# 12h: motor GBTM (ng=3) 表型稳定性 bootstrap — U3 缺口闭合
# 对象锁定 (2026-09-05 回读源文档核证):
#   - 模型: 08a2_motor_gbtm_select.R — hlme(motor~t10+I(t10^2), mixture=~t10+I(t10^2), random=~-1), ng=3 (冻结规则: 可行 ng 中 BIC 最低)
#   - 数据: results/features/13_motor_binned6h.parquet (motor binned 6h, MIMIC-IV discovery)
#   - 原标签: results/cluster/13_motor/13b_motor_classes.csv
#   - 动机: 主模型 Block C = motor ng=3 截断后验 (prob_m1-3), 而 12g 验的是 total ng=5 (敏感性族)
#          → 本脚本补验主特征族 (DISCLOSURE_LEDGER U3②, SAP §12 修订记录 2026-09-05)
# 范式: 12g_gbtm_bootstrap.R 逐字复刻 (含 2026-09-05 L97 原子化修复), 仅换对象
# 判定标准 (同 SAP §8): 中位 ARI >= 0.75 且 各表型出现率 >= 0.80 (出现 = 该表型成员 >=50% 落入同一 bootstrap 类)
# 规模: ng=3 × B=100; 敏感性 ng=2/4 × B=50 (同一批抽样); gridsearch rep=5 (与 12g 同强度降档, 已披露)
# 断点: 每个 b 后重写 progress JSON, 中断可读回
suppressWarnings(suppressMessages({ library(arrow); library(lcmm); library(jsonlite) }))
ROOT <- "E:/TBI subtype"
OUTD <- file.path(ROOT, "results/cluster/13_motor")
PROG <- file.path(OUTD, "12h_progress.txt")
lg <- function(...) { cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n"); cat(format(Sys.time(), "%H:%M:%S"), "|", ..., "\n", file=PROG, append=TRUE) }

B_MAIN <- 100; B_SENS <- 50
K_MAIN <- 3; K_SENS <- c(2, 4)
ARI_MIN <- 0.75; APP_MIN <- 0.80; PURITY_TH <- 0.5
SEED0 <- 20260905

b <- as.data.frame(read_parquet(file.path(ROOT, "results/features/13_motor_binned6h.parquet")))
orig <- read.csv(file.path(OUTD, "13b_motor_classes.csv"))
stopifnot(all(c("stay_id", "class") %in% names(orig)), max(orig$class) == K_MAIN, nrow(orig) >= 2700)
b <- merge(b, orig, by = "stay_id")
stopifnot(!any(is.na(b$class)))
b$t10 <- b$t / 10
uq <- unique(b$stay_id); n_pat <- length(uq)
stopifnot(n_pat == nrow(orig))   # 数据与标签患者集必须一致 (不写魔法数, 写一致性断言)
lg("bootstrap 对象: n=", n_pat, " 患者 / ", nrow(b), " 行; 原类分布=", paste(table(b$class[!duplicated(b$stay_id)]), collapse="/"))

# ---- 手工 ARI (零依赖, 12g 同款) ----
ari_manual <- function(l1, l2) {
  tab <- table(l1, l2); n <- sum(tab); cs2 <- function(x) choose(x, 2)
  a <- sum(cs2(as.numeric(tab)))
  bb <- sum(cs2(as.numeric(rowSums(tab)))) - a
  cc <- sum(cs2(as.numeric(colSums(tab)))) - a
  M <- cs2(n); if (M == 0) return(NA_real_)
  num <- a - (a + bb) * (a + cc) / M; den <- (a + bb + a + cc) / 2 - (a + bb) * (a + cc) / M
  if (den == 0) return(if (num == 0) 1 else 0); num / den
}
purity_by_class <- function(boot_cls, orig_cls, K) {
  sapply(seq_len(K), function(j) {
    idx <- which(orig_cls == j); if (!length(idx)) return(NA_real_)
    max(tabulate(boot_cls[idx], nbins = K) / length(idx))
  })
}

results <- list(); ckpt <- file.path(OUTD, "12h_boot_progress.json")
if (file.exists(ckpt)) { results <- fromJSON(ckpt, simplifyVector = FALSE); lg("断点读回: 已完成 ", length(results), " 个 b") }

set.seed(SEED0)
samples <- lapply(seq_len(B_MAIN), function(i) sample(uq, replace = TRUE))

t_start <- Sys.time()
for (b_i in seq_len(B_MAIN)) {
  key <- as.character(b_i)
  if (!is.null(results[[key]])) next
  set.seed(SEED0 + b_i)
  samp <- samples[[b_i]]
  rows <- lapply(seq_along(samp), function(d) { x <- b[b$stay_id == samp[d], ]; x$sid <- d; x })
  bd <- do.call(rbind, rows)
  fits <- list()
  m1 <- try(hlme(motor ~ t10 + I(t10^2), random = ~-1, subject = "sid", ng = 1, data = bd), silent = TRUE)
  if (inherits(m1, "try-error")) { results[[key]] <- list(b = b_i, status = "m1_fail"); next }
  for (K in c(K_MAIN, if (b_i <= B_SENS) K_SENS else NULL)) {
    fk <- try(suppressWarnings(gridsearch(rep = 5, maxiter = 20, minit = m1,
      hlme(motor ~ t10 + I(t10^2), mixture = ~t10 + I(t10^2), random = ~-1, subject = "sid", ng = K, data = bd))),
      silent = TRUE)
    if (inherits(fk, "try-error") || !isTRUE(fk$conv %in% c(1, 2))) {
      fits[[as.character(K)]] <- list(conv = if (inherits(fk, "try-error")) NA else fk$conv, status = "fit_fail"); next
    }
    pp <- fk$pprob[, c("sid", "class")]
    m2 <- merge(pp, unique(bd[, c("sid", "stay_id", "class")]), by = "sid")
    u <- m2[!duplicated(m2$stay_id), ]
    ari <- ari_manual(u$class.x, u$class.y)
    pur <- purity_by_class(u$class.x, u$class.y, K)
    sizes <- as.numeric(tabulate(u$class.x, nbins = K))
    fits[[as.character(K)]] <- list(conv = fk$conv, BIC = round(fk$BIC, 1), ari = round(ari, 4),
                                    purity = round(pur, 3), sizes = sizes,
                                    min_size_pct = round(100 * min(sizes) / nrow(u), 2))
  }
  results[[key]] <- list(b = b_i, status = "ok", elapsed_min = round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1), fits = fits)
  write(toJSON(results, auto_unbox = TRUE, digits = 6), ckpt)
  if (b_i %% 5 == 0 || b_i == 1) {
    a3 <- sapply(results, function(r) if (!is.null(r$fits$`3`$ari)) r$fits$`3`$ari else NA)
    lg("b=", b_i, "/", B_MAIN, " ng3 已得 ARI n=", sum(!is.na(a3)), " 中位=", round(median(a3, na.rm = TRUE), 3))
  }
}

# ---- 汇总 + 判定 (12g 同款 + 原子化修复) ----
summ <- list()
for (K in c(K_MAIN, K_SENS)) {
  ks <- as.character(K); Bs <- if (K == K_MAIN) B_MAIN else B_SENS
  aris <- sapply(results, function(r) { v <- r$fits[[ks]]$ari; if (is.null(v)) NA else v })
  purs <- do.call(rbind, lapply(results, function(r) { p <- r$fits[[ks]]$purity; if (is.null(p)) rep(NA, K) else { p <- suppressWarnings(as.numeric(unlist(p))); if (length(p) < K) p <- c(p, rep(NA, K - length(p))); p } }))
  conv_rate <- mean(sapply(results, function(r) { v <- r$fits[[ks]]$conv; !is.null(v) && !is.na(v) && v %in% c(1, 2) }))
  conv_rate_fit <- { nf <- sum(sapply(results, function(r) !is.null(r$fits[[ks]]$ari))); if (nf > 0) sum(sapply(results, function(r) { v <- r$fits[[ks]]$conv; !is.null(v) && !is.na(v) && v %in% c(1, 2) })) / nf else NA }
  app_rate <- apply(purs, 2, function(p) mean(p >= PURITY_TH, na.rm = TRUE))
  summ[[ks]] <- list(B_target = Bs, n_ari = sum(!is.na(aris)), conv_rate = round(conv_rate, 3), conv_rate_among_fits = round(conv_rate_fit, 3),
                     median_ari = round(median(aris, na.rm = TRUE), 4),
                     q25_ari = round(quantile(aris, .25, na.rm = TRUE), 4),
                     q75_ari = round(quantile(aris, .75, na.rm = TRUE), 4),
                     median_purity_by_class = round(apply(purs, 2, median, na.rm = TRUE), 3),
                     appearance_rate_by_class = round(app_rate, 3))
  if (K == K_MAIN) {
    verdict <- list(criterion = sprintf("median ARI>=%.2f & all appearance>=%.2f (purity>=%.1f)", ARI_MIN, APP_MIN, PURITY_TH),
                    median_ari_pass = median(aris, na.rm = TRUE) >= ARI_MIN,
                    appearance_pass = all(app_rate >= APP_MIN),
                    overall = (median(aris, na.rm = TRUE) >= ARI_MIN) && all(app_rate >= APP_MIN))
  }
}
out <- list(object = "motor GBTM ng=3 (08a2 选定, 预测层主特征族 Block C) — U3 缺口闭合补验",
            design = "patient-level bootstrap with replacement; duplicates as new subjects; hlme identical spec; gridsearch rep=5 (与 12g 同强度降档, 已披露); conv_rate_among_fits = 分母校正收敛率 (12g 教训: conv_rate 分母含未跑 b)",
            criteria = verdict, summary = summ,
            elapsed_total_min = round(as.numeric(difftime(Sys.time(), t_start, units = "mins")), 1),
            seed = SEED0, saved = Sys.time())
write(toJSON(out, auto_unbox = TRUE, pretty = TRUE, digits = 6), file.path(OUTD, "12h_motor_gbtm_bootstrap.json"))
lg("12H_BOOTSTRAP_DONE verdict=", verdict$overall, " median_ARI=", summ[["3"]]$median_ari,
   " appearance=", paste(summ[["3"]]$appearance_rate_by_class, collapse="/"))
