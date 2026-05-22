#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motus / RoboTwin TTS 日志与 summary.csv 可视化脚本。

默认用法：
    cd /home/ubuntu/workspace/RoboTwin/policy/Motus
    python tts_log_visualizer_zh_v3.py SUMMARY_CSV OUTER_LOG

也支持旧写法：
    python tts_log_visualizer_zh_v3.py --csv SUMMARY_CSV --log OUTER_LOG

输出目录默认：
    /home/ubuntu/workspace/RoboTwin/policy/Motus/plot/<run_name>/

目录结构：
    <run_name>/
      plots/   图
      csv/     派生表
      report.txt

说明：
    1. 不依赖环境变量。
    2. 所有可配置项集中在顶部 CONFIG 区。
    3. 自动从 outer log 文件名截取 run name。
    4. 自动检测中文字体；没有中文字体时，图内注释自动退化为英文，避免方框。
"""

# =========================
# CONFIG
# =========================
RUNS = []
# 示例：
# RUNS = [
#     {
#         "name": None,
#         "csv": "/path/to/summary.csv",
#         "log": "/path/to/outer.log",
#     },
# ]

DEFAULT_OUTPUT_DIR = "/home/ubuntu/workspace/RoboTwin/policy/Motus/plot"
SUCCESS_BIN_SIZE = 10
HIST_BINS = 30
FIG_DPI = 140
ANNOTATE_PLOTS = True
WRITE_EXTRA_NOTES = False

# 中文字体候选。脚本会自动选择第一个可用字体。
CJK_FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
CJK_FONT_NAME_CANDIDATES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "SimHei",
    "Microsoft YaHei",
]

# =========================
# Imports
# =========================
import argparse
import math
import os
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# Font setup
# =========================
def setup_font():
    """Return (has_chinese_font, font_name_or_None)."""
    for font_path in CJK_FONT_PATH_CANDIDATES:
        p = Path(font_path)
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                name = fm.FontProperties(fname=str(p)).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return True, name
            except Exception:
                pass

    available = {f.name for f in fm.fontManager.ttflist}
    for name in CJK_FONT_NAME_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True, name

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False, None


HAS_CJK_FONT, CJK_FONT_NAME = setup_font()
PLOT_LANG = "zh" if HAS_CJK_FONT else "en"


def zh(cn, en):
    return cn if PLOT_LANG == "zh" else en


# =========================
# Utility functions
# =========================
def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", str(s))


def safe_name(s):
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "run"


def derive_run_name(log_path=None, csv_path=None):
    if log_path:
        stem = Path(log_path).stem
        stem = re.sub(r"_\d{8}_\d{6}$", "", stem)
        if "_tts_" in stem:
            stem = stem.split("_tts_", 1)[1]
        stem = stem.replace("_nodecode", "")
        stem = re.sub(r"_(latent|token|tokens)$", "", stem)
        stem = re.sub(r"_+", "_", stem).strip("_")
        if stem and stem not in {"scan_object", "click_alarmclock"}:
            return safe_name(stem)

    if csv_path:
        p = Path(csv_path)
        # 常见路径：logs_single_xxx/tts/scan_object/summary.csv
        parts = list(p.parts)
        for i, part in enumerate(parts):
            if part.startswith("logs_single_"):
                return safe_name(part)
        return safe_name(p.parent.name or p.stem)

    return "single_run"


def parse_float_list(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    vals = []
    for item in s.split("|"):
        item = item.strip()
        if item == "":
            continue
        try:
            vals.append(float(item))
        except Exception:
            pass
    return vals


def parse_int_list(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    vals = []
    for item in s.split("|"):
        item = item.strip()
        if item == "":
            continue
        try:
            vals.append(int(float(item)))
        except Exception:
            pass
    return vals


def to_num(series, default=np.nan):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def list_margin(vals, lower_is_better=True):
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 2:
        return np.nan
    arr = np.array(vals, dtype=float)
    if lower_is_better:
        order = np.sort(arr)
        return float(order[1] - order[0])
    order = np.sort(arr)[::-1]
    return float(order[0] - order[1])


def tie_count(vals, lower_is_better=True, eps=1e-9):
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return 0
    arr = np.array(vals, dtype=float)
    best = np.min(arr) if lower_is_better else np.max(arr)
    return int(np.sum(np.abs(arr - best) <= eps))


def safe_mean(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan


def safe_median(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100 * float(x):.1f}%"


# =========================
# Log parsing
# =========================
def parse_log_config(log_path):
    cfg = {}
    if not log_path or not Path(log_path).exists():
        return cfg

    patterns = {
        "task_name": r"^Task Name:\s*(.*)$",
        "task_config": r"^Task Config:\s*(.*)$",
        "seed": r"^Seed:\s*(.*)$",
        "tts_enable": r"^TTS Enable:\s*(.*)$",
        "tts_num_samples": r"^TTS Num Samples:\s*(.*)$",
        "tts_method": r"^TTS Method:\s*(.*)$",
        "tts_batch_size": r"^TTS Batch Size:\s*(.*)$",
        "tts_decode_video": r"^TTS Decode Video:\s*(.*)$",
        "tts_video_feature": r"^TTS Video Feature:\s*(.*)$",
        "tts_rank_fusion": r"^TTS Rank Fusion:\s*(.*)$",
        "tts_video_weight": r"^TTS Video Weight:\s*(.*)$",
        "tts_rrf_k": r"^TTS RRF K:\s*(.*)$",
        "tts_video_weight_low": r"^TTS Video W Low:\s*(.*)$",
        "tts_gate_spearman_thresh": r"^TTS Gate Spearman:\s*(.*)$",
        "tts_gate_distance_ratio_thresh": r"^TTS Gate DistRatio:\s*(.*)$",
    }

    for line in Path(log_path).read_text(errors="ignore").splitlines():
        line = strip_ansi(line).strip()
        for k, pat in patterns.items():
            m = re.search(pat, line)
            if m:
                cfg[k] = m.group(1).strip()
    return cfg


def parse_success_log(log_path):
    """Parse cumulative success lines into per-episode success table."""
    cols = ["episode", "cum_success", "cum_total", "cum_rate", "current_seed", "success"]
    if not log_path or not Path(log_path).exists():
        return pd.DataFrame(columns=cols)

    rows = []
    prev_success = 0
    pat = re.compile(
        r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%\s*,\s*current seed:\s*(\d+)"
    )

    for raw in Path(log_path).read_text(errors="ignore").splitlines():
        line = strip_ansi(raw)
        m = pat.search(line)
        if not m:
            continue
        cum_success = int(m.group(1))
        cum_total = int(m.group(2))
        cum_rate = float(m.group(3)) / 100.0
        seed = int(m.group(4))
        success = int(cum_success > prev_success)
        prev_success = cum_success
        rows.append({
            "episode": cum_total,
            "cum_success": cum_success,
            "cum_total": cum_total,
            "cum_rate": cum_rate,
            "current_seed": seed,
            "success": success,
        })

    return pd.DataFrame(rows, columns=cols)


def make_success_bins(success_df, bin_size=10):
    if success_df.empty:
        return pd.DataFrame(columns=["bin_start", "bin_end", "n", "success_sum", "success_rate"])
    df = success_df.copy()
    df["bin_id"] = (df["episode"] - 1) // bin_size
    g = df.groupby("bin_id", as_index=False).agg(
        bin_start=("episode", "min"),
        bin_end=("episode", "max"),
        n=("success", "count"),
        success_sum=("success", "sum"),
    )
    g["success_rate"] = g["success_sum"] / g["n"]
    return g[["bin_start", "bin_end", "n", "success_sum", "success_rate"]]


# =========================
# CSV processing
# =========================
def load_and_derive(csv_path):
    df = pd.read_csv(csv_path)

    # 基础数值列
    for col in [
        "episode", "step", "samples", "selected_idx", "global_medoid_idx",
        "selected_action_rank", "selected_video_rank", "selected_fused_rank",
        "rank_spearman", "spearman_rank_corr", "distance_ratio_video_over_action",
        "video_action_distance_ratio", "selected_avg_l2", "min_avg_l2", "max_avg_l2",
        "mean_avg_l2", "action_avg_l2_min", "action_avg_l2_max", "action_avg_l2_mean",
        "action_avg_l2_std", "video_avg_l2_min", "video_avg_l2_max",
        "video_avg_l2_mean", "video_avg_l2_std", "action_pairwise_median",
        "video_pairwise_median", "selected_fused_score", "video_weight", "rrf_k",
        "action_feature_norm_mean", "video_feature_norm_mean",
        "configured_video_weight", "effective_video_weight", "video_weight_low", "fallback_video_weight",
        "candidate_pool_size", "selected_local_idx", "selected_local_action_rank",
        "selected_local_video_rank", "selected_local_fused_rank", "selected_local_fused_score",
        "gate_rank_spearman", "gate_distance_ratio",
        "gate_rank_spearman_thresh", "gate_spearman_thresh", "gate_distance_ratio_thresh",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "rank_spearman" not in df.columns and "spearman_rank_corr" in df.columns:
        df["rank_spearman"] = pd.to_numeric(df["spearman_rank_corr"], errors="coerce")
    if "distance_ratio_video_over_action" not in df.columns and "video_action_distance_ratio" in df.columns:
        df["distance_ratio_video_over_action"] = pd.to_numeric(df["video_action_distance_ratio"], errors="coerce")

    # 从 rank list 补全 selected_action_rank / selected_video_rank
    if "selected_action_rank" not in df.columns and "action_rank_ids" in df.columns:
        vals = []
        for _, row in df.iterrows():
            ranks = parse_int_list(row.get("action_rank_ids", ""))
            idx = int(row.get("selected_idx", -1)) if pd.notna(row.get("selected_idx", np.nan)) else -1
            vals.append(ranks[idx] if 0 <= idx < len(ranks) else np.nan)
        df["selected_action_rank"] = vals
    if "selected_video_rank" not in df.columns and "video_rank_ids" in df.columns:
        vals = []
        for _, row in df.iterrows():
            ranks = parse_int_list(row.get("video_rank_ids", ""))
            idx = int(row.get("selected_idx", -1)) if pd.notna(row.get("selected_idx", np.nan)) else -1
            vals.append(ranks[idx] if 0 <= idx < len(ranks) else np.nan)
        df["selected_video_rank"] = vals

    # 布尔派生列
    if "selected_action_rank" in df.columns:
        df["selected_action_best"] = (df["selected_action_rank"] == 0).astype(float)
        df["action_override"] = (df["selected_action_rank"] > 0).astype(float)
    else:
        df["selected_action_best"] = np.nan
        df["action_override"] = np.nan

    if "selected_video_rank" in df.columns:
        df["selected_video_best"] = (df["selected_video_rank"] == 0).astype(float)
    else:
        df["selected_video_best"] = np.nan

    if "rank_agree" in df.columns:
        df["rank_agree_bool"] = df["rank_agree"].astype(str).str.lower().isin(["true", "1", "yes"])
        df["rank_agree_num"] = df["rank_agree_bool"].astype(float)
    elif "action_best_idx" in df.columns and "video_best_idx" in df.columns:
        df["rank_agree_num"] = (df["action_best_idx"] == df["video_best_idx"]).astype(float)
    else:
        df["rank_agree_num"] = np.nan

    for bool_col in ["gate_pass", "spearman_gate_pass", "distance_gate_pass"]:
        if bool_col in df.columns:
            df[f"{bool_col}_num"] = df[bool_col].astype(str).str.lower().isin(["true", "1", "yes"]).astype(float)
        else:
            df[f"{bool_col}_num"] = np.nan

    if "effective_video_weight" in df.columns:
        df["video_weight_used"] = pd.to_numeric(df["effective_video_weight"], errors="coerce")
    elif "video_weight" in df.columns:
        df["video_weight_used"] = pd.to_numeric(df["video_weight"], errors="coerce")
    else:
        df["video_weight_used"] = np.nan

    if "candidate_pool_size" in df.columns and "samples" in df.columns:
        pool = pd.to_numeric(df["candidate_pool_size"], errors="coerce")
        samples = pd.to_numeric(df["samples"], errors="coerce")
        df["candidate_pool_ratio"] = pool / samples.replace(0, np.nan)
    else:
        df["candidate_pool_ratio"] = np.nan

    # margin / tie
    action_margins = []
    video_margins = []
    fused_margins = []
    fused_ties = []
    formula_ok = []
    selected_is_best = []

    for _, row in df.iterrows():
        action_vals = parse_float_list(row.get("action_avg_l2", ""))
        video_vals = parse_float_list(row.get("video_avg_l2", ""))
        fused_vals = parse_float_list(row.get("fused_scores", ""))
        direction = str(row.get("fused_score_direction", "")).lower()
        higher = "higher" in direction
        lower = not higher

        action_margins.append(list_margin(action_vals, lower_is_better=True))
        video_margins.append(list_margin(video_vals, lower_is_better=True))
        fused_margins.append(list_margin(fused_vals, lower_is_better=lower) if fused_vals else np.nan)
        fused_ties.append(tie_count(fused_vals, lower_is_better=lower) if fused_vals else 0)

        # 校验 selected 是否是 fused 最优
        idx = row.get("selected_idx", np.nan)
        if fused_vals and pd.notna(idx):
            idx = int(idx)
            arr = np.array(fused_vals, dtype=float)
            best_idx = int(np.argmax(arr) if higher else np.argmin(arr))
            selected_is_best.append(float(idx == best_idx))
        else:
            selected_is_best.append(np.nan)

        # 校验常见公式
        method = str(row.get("rank_fusion_method", "")).lower()
        a_ranks = parse_int_list(row.get("action_rank_ids", ""))
        v_ranks = parse_int_list(row.get("video_rank_ids", ""))
        if fused_vals and a_ranks and v_ranks and len(fused_vals) == len(a_ranks) == len(v_ranks) and np.all(np.isfinite(np.array(fused_vals, dtype=float))):
            w = row.get("effective_video_weight", row.get("video_weight", np.nan))
            if pd.isna(w):
                w = row.get("video_weight", np.nan)
            k = row.get("rrf_k", np.nan)
            try:
                w = float(w)
            except Exception:
                w = np.nan
            try:
                k = float(k)
            except Exception:
                k = 1.0
            if method == "weighted_borda":
                expect = np.array(a_ranks, dtype=float) + w * np.array(v_ranks, dtype=float)
                formula_ok.append(float(np.allclose(expect, np.array(fused_vals), atol=1e-5, equal_nan=True)))
            elif method == "borda":
                expect = np.array(a_ranks, dtype=float) + np.array(v_ranks, dtype=float)
                formula_ok.append(float(np.allclose(expect, np.array(fused_vals), atol=1e-5, equal_nan=True)))
            elif method == "rrf":
                expect = 1.0 / (k + np.array(a_ranks, dtype=float) + 1.0) + w / (k + np.array(v_ranks, dtype=float) + 1.0)
                formula_ok.append(float(np.allclose(expect, np.array(fused_vals), atol=1e-5, equal_nan=True)))
            else:
                formula_ok.append(np.nan)
        else:
            formula_ok.append(np.nan)

    df["action_margin"] = action_margins
    df["video_margin"] = video_margins
    df["fused_margin"] = fused_margins
    df["fused_tie_count"] = fused_ties
    df["selected_is_fused_best"] = selected_is_best
    df["fusion_formula_ok"] = formula_ok

    return df


def aggregate_episode(row_df, success_df=None):
    if row_df.empty or "episode" not in row_df.columns:
        return pd.DataFrame()

    metric_cols = [
        "selected_action_best", "action_override", "selected_video_best",
        "rank_agree_num", "rank_spearman", "distance_ratio_video_over_action",
        "action_margin", "video_margin", "fused_margin", "fused_tie_count",
        "selected_action_rank", "selected_video_rank", "selected_fused_rank",
        "selected_is_fused_best", "fusion_formula_ok",
        "video_weight_used", "gate_pass_num", "spearman_gate_pass_num", "distance_gate_pass_num",
        "candidate_pool_size", "candidate_pool_ratio",
        "selected_local_action_rank", "selected_local_video_rank", "selected_local_fused_rank", "selected_local_fused_score",
    ]
    metric_cols = [c for c in metric_cols if c in row_df.columns]

    agg = row_df.groupby("episode", as_index=False).agg(
        n_tts_rows=("episode", "size"),
        first_step=("step", "min") if "step" in row_df.columns else ("episode", "size"),
        last_step=("step", "max") if "step" in row_df.columns else ("episode", "size"),
    )
    for col in metric_cols:
        tmp = row_df.groupby("episode")[col].mean().reset_index(name=col)
        agg = agg.merge(tmp, on="episode", how="left")

    if success_df is not None and not success_df.empty:
        agg = agg.merge(success_df[["episode", "success", "current_seed", "cum_rate"]], on="episode", how="left")

    return agg


# =========================
# Plot helpers
# =========================
def ensure_dirs(run_dir):
    plots_dir = run_dir / "plots"
    csv_dir = run_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, csv_dir


def add_note(fig, note):
    if not ANNOTATE_PLOTS or not note:
        return
    note = textwrap.fill(note, width=105)
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)


def save_fig(path, note=None):
    fig = plt.gcf()
    if note:
        add_note(fig, note)
    plt.tight_layout(rect=[0, 0.08 if note else 0, 1, 1])
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def line_plot(df, x, y, path, title, xlabel, ylabel, note=None, ylim=None):
    if df.empty or x not in df.columns or y not in df.columns:
        return False
    d = df[[x, y]].dropna()
    if d.empty:
        return False
    plt.figure(figsize=(10, 5))
    plt.plot(d[x], d[y], marker="o", linewidth=1.5, markersize=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.grid(True, alpha=0.3)
    save_fig(path, note)
    return True


def hist_plot(df, col, path, title, xlabel, note=None, bins=HIST_BINS):
    if df.empty or col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return False
    plt.figure(figsize=(9, 5))
    plt.hist(s, bins=bins)
    plt.axvline(s.mean(), linestyle="--", linewidth=1.5, label=f"mean={s.mean():.4g}")
    plt.axvline(s.median(), linestyle=":", linewidth=1.5, label=f"median={s.median():.4g}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(zh("频数", "count"))
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_fig(path, note)
    return True


def bar_plot_counts(df, col, path, title, xlabel, note=None):
    if df.empty or col not in df.columns:
        return False
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return False
    counts = s.astype(int).value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(zh("频数", "count"))
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(path, note)
    return True


def box_success_plot(ep_df, col, path, title, ylabel, note=None):
    if ep_df.empty or col not in ep_df.columns or "success" not in ep_df.columns:
        return False
    d = ep_df[["success", col]].dropna()
    if d.empty or d["success"].nunique() < 2:
        return False
    data = [d[d["success"] == 1][col].values, d[d["success"] == 0][col].values]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=[zh("成功", "success"), zh("失败", "fail")], showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    save_fig(path, note)
    return True


def plot_success(success_df, bins_df, plots_dir):
    if not success_df.empty:
        line_plot(
            success_df, "episode", "cum_rate",
            plots_dir / "success_cumulative.png",
            zh("累计成功率走势", "Cumulative success rate"),
            "episode", zh("累计成功率", "cumulative success rate"),
            note=zh(
                "看成功率是否存在前低后高、前高后低或某一段 seed 明显偏难。曲线只来自 outer log，不依赖 CSV。",
                "Shows whether some seed ranges are much harder or easier. This comes from the outer log only."
            ),
            ylim=(0, 1),
        )
    if not bins_df.empty:
        labels = [f"{int(a)}-{int(b)}" for a, b in zip(bins_df["bin_start"], bins_df["bin_end"])]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, bins_df["success_rate"])
        plt.ylim(0, 1)
        plt.title(zh(f"每 {SUCCESS_BIN_SIZE} 个 episode 的局部成功率", f"Success rate per {SUCCESS_BIN_SIZE} episodes"))
        plt.xlabel("episode bin")
        plt.ylabel(zh("局部成功率", "bin success rate"))
        plt.xticks(rotation=30)
        plt.grid(True, axis="y", alpha=0.25)
        save_fig(
            plots_dir / "success_bins.png",
            zh("看不同 seed 区间的难度是否不均匀。某些柱子明显低，说明那一段 seed 更难或更不稳定。",
               "Shows seed-range difficulty. Low bars indicate harder or less stable seed ranges.")
        )


def plot_metrics(row_df, ep_df, plots_dir):
    # episode-level curves
    curves = [
        ("selected_action_best", zh("每个 episode 选择 action best 的比例", "Per-episode selected action-best ratio"), zh("比例", "ratio"), (0, 1),
         zh("越接近 1，越接近 action-only global medoid；低于 1 表示 video fusion 改变过 action-only 选择。", "Near 1 means action-only behavior; below 1 means video fusion changed some selections.")),
        ("action_override", zh("每个 episode 的 action override 比例", "Per-episode action override ratio"), zh("比例", "ratio"), (0, 1),
         zh("override 表示最终没有选 action rank 0。它是判断视频分支是否真正介入选择的关键指标。", "Override means the final choice is not action rank 0. It directly tests whether video affects selection.")),
        ("selected_video_best", zh("每个 episode 选择 video best 的比例", "Per-episode selected video-best ratio"), zh("比例", "ratio"), (0, 1),
         zh("越高说明最终选择更常与 video rank 第一一致；但过高也可能表示视频项过强。", "Higher means final choices often match video rank 0, but too high may mean video dominates.")),
        ("rank_agree_num", zh("每个 episode 的 action/video best 一致比例", "Per-episode action/video best agreement"), zh("比例", "ratio"), (0, 1),
         zh("action best 和 video best 是否指向同一个候选。高一致性通常说明两个模态判断相互支持。", "Whether action-best and video-best point to the same candidate. High agreement means the modalities support each other.")),
        ("rank_spearman", zh("每个 episode 的 action/video rank Spearman", "Per-episode action/video rank Spearman"), "Spearman", (-1, 1),
         zh("正值说明动作排序和视频排序一致；负值说明两者冲突。它可作为融合置信度信号。", "Positive means action and video rankings agree; negative means conflict. It can be used as a confidence signal.")),
        ("distance_ratio_video_over_action", zh("每个 episode 的 video/action 距离比", "Per-episode video/action distance ratio"), zh("距离比", "distance ratio"), None,
         zh("video pairwise median / action pairwise median。偏高说明视频 latent 相对更分散，可能表示未来预测不稳定或场景更难。", "Video pairwise median over action pairwise median. High values mean video latents are relatively dispersed.")),
        ("action_margin", zh("每个 episode 的 action margin", "Per-episode action margin"), "margin", None,
         zh("action 第一名和第二名距离差。大表示 action 排序更确定；但失败时偏大可能是 confidently wrong。", "Gap between action rank 1 and 2. Large means confident action ranking, but may be confidently wrong in failures.")),
        ("video_margin", zh("每个 episode 的 video margin", "Per-episode video margin"), "margin", None,
         zh("video 第一名和第二名距离差。单独不能代表可靠，最好和 Spearman、distance ratio 一起看。", "Gap between video rank 1 and 2. Use with Spearman and distance ratio, not alone.")),
        ("fused_margin", zh("每个 episode 的 fused margin", "Per-episode fused margin"), "margin", None,
         zh("融合第一名和第二名的分数差。越小表示选择更不稳定。RRF 和 Borda 的绝对尺度不能直接比较。", "Gap between best and second fused scores. Small means unstable choice. RRF and Borda scales are not directly comparable.")),
        ("fused_tie_count", zh("每个 episode 的 fused tie count", "Per-episode fused tie count"), "tie count", None,
         zh("融合分数并列第一的候选数。大于 1 表示平票，可能存在 index bias。", "Number of candidates tied for best fused score. Above 1 indicates ties and possible index bias.")),
        ("video_weight_used", zh("每个 episode 的实际 video weight", "Per-episode effective video weight"), "weight", None,
         zh("gated 方法中这是 gate 后实际使用的视频权重；非 gated 方法通常等于配置权重。", "For gated fusion this is the post-gate video weight; otherwise it usually equals the configured weight.")),
        ("gate_pass_num", zh("每个 episode 的 gate pass 比例", "Per-episode gate-pass ratio"), zh("比例", "ratio"), (0, 1),
         zh("仅 video_gated_fusion 有意义。越高表示越常允许高视频权重。", "Meaningful for video_gated_fusion. Higher means high video weight is allowed more often.")),
        ("candidate_pool_ratio", zh("每个 episode 的 cluster 候选池比例", "Per-episode cluster candidate-pool ratio"), zh("比例", "ratio"), (0, 1),
         zh("仅 video_cluster_fusion 有意义。越低表示 cluster gate 剪掉的候选越多。", "Meaningful for video_cluster_fusion. Lower means the cluster gate prunes more candidates.")),
    ]
    for col, title, ylabel, ylim, note in curves:
        line_plot(ep_df, "episode", col, plots_dir / f"episode_{col}.png", title, "episode", ylabel, note=note, ylim=ylim)

    # histograms/counts
    count_cols = [
        ("selected_action_rank", zh("最终选择的 action rank 分布", "Selected action rank distribution"), "selected_action_rank",
         zh("0 表示选 action medoid。非 0 越多，说明 video fusion 越常改变 action-only 选择。", "0 means action medoid. Non-zero means video fusion changes action-only selection.")),
        ("selected_video_rank", zh("最终选择的 video rank 分布", "Selected video rank distribution"), "selected_video_rank",
         zh("0 表示最终选择也是 video best。若平均 rank 很高，说明视频分支影响较弱。", "0 means final choice is also video-best. High average rank means weak video influence.")),
    ]
    for col, title, xlabel, note in count_cols:
        bar_plot_counts(row_df, col, plots_dir / f"hist_{col}.png", title, xlabel, note=note)

    hist_cols = [
        ("rank_spearman", zh("action/video rank Spearman 分布", "Action/video rank Spearman distribution"), "Spearman",
         zh("整体看两个模态排序是否一致。正值多说明 video latent 不是随机信号。", "Shows whether the two modalities rank candidates consistently. Positive values mean video is not random.")),
        ("distance_ratio_video_over_action", zh("video/action 距离比分布", "Video/action distance ratio distribution"), "distance_ratio_video_over_action",
         zh("诊断视频 latent 相对动作候选是否更分散。它不直接参与 rank fusion，但可作为可靠性信号。", "Diagnoses whether video latents are more dispersed than action candidates. Useful as reliability signal.")),
        ("action_margin", zh("action margin 分布", "Action margin distribution"), "action_margin",
         zh("第一名和第二名差距。接近 0 表示 action 候选难分，高值表示 action 排序明确。", "Gap between top two action candidates. Near 0 means hard to distinguish.")),
        ("video_margin", zh("video margin 分布", "Video margin distribution"), "video_margin",
         zh("第一名和第二名差距。高 video margin 只有在 Spearman 也高时才更可信。", "Gap between top two video candidates. More useful when Spearman is also high.")),
        ("fused_margin", zh("fused margin 分布", "Fused margin distribution"), "fused_margin",
         zh("融合选择稳定性。接近 0 表示第一名和第二名几乎打平。", "Fusion selection stability. Near 0 means almost tied.")),
        ("video_weight_used", zh("实际 video weight 分布", "Effective video weight distribution"), "video_weight_used",
         zh("看 gated 方法中高低权重切换是否频繁。", "Shows how often gated fusion switches between high and low weights.")),
        ("candidate_pool_ratio", zh("cluster 候选池比例分布", "Cluster candidate-pool ratio distribution"), "candidate_pool_ratio",
         zh("看 cluster gate 是否经常剪枝。", "Shows whether the cluster gate often prunes candidates.")),
    ]
    for col, title, xlabel, note in hist_cols:
        hist_plot(row_df, col, plots_dir / f"hist_{col}.png", title, xlabel, note=note)

    # success/failure boxplots
    box_cols = [
        ("rank_spearman", zh("rank Spearman：成功 vs 失败", "Rank Spearman: success vs failure"), "Spearman",
         zh("如果成功组更高，说明 action/video 排序一致性可能是有效置信度信号。", "If success is higher, rank agreement may be a useful confidence signal.")),
        ("distance_ratio_video_over_action", zh("video/action 距离比：成功 vs 失败", "Video/action distance ratio: success vs failure"), "distance ratio",
         zh("如果失败组更高，说明失败时视频 latent 更分散，可考虑降低 video weight。", "If failure is higher, video latents are more dispersed in failures; consider lowering video weight.")),
        ("action_margin", zh("action margin：成功 vs 失败", "Action margin: success vs failure"), "action_margin",
         zh("失败组 margin 高可能表示模型在错误动作上也很自信，即 confidently wrong。", "High failure margin may mean confidently wrong action consensus.")),
        ("video_margin", zh("video margin：成功 vs 失败", "Video margin: success vs failure"), "video_margin",
         zh("video margin 单独不够可靠，应与 Spearman 和 distance ratio 一起解释。", "Video margin alone is not sufficient; interpret with Spearman and distance ratio.")),
        ("action_override", zh("action override：成功 vs 失败", "Action override: success vs failure"), "action_override",
         zh("看视频介入选择是否更常出现在成功或失败 episode。", "Checks whether video intervention is more common in success or failure episodes.")),
        ("gate_pass_num", zh("gate pass：成功 vs 失败", "Gate pass: success vs failure"), "gate_pass",
         zh("如果成功组更高，说明排名一致性 gate 可能是有效信号。", "If success is higher, the rank-consistency gate may be useful.")),
    ]
    for col, title, ylabel, note in box_cols:
        box_success_plot(ep_df, col, plots_dir / f"success_vs_fail_{col}.png", title, ylabel, note=note)


# =========================
# Report
# =========================
def summarize_run(run_name, cfg, success_df, bins_df, row_df, ep_df):
    lines = []
    lines.append(f"# {run_name}")
    lines.append("")

    if cfg:
        lines.append("## Log configuration")
        for k in [
            "task_name", "task_config", "seed", "tts_enable", "tts_num_samples",
            "tts_method", "tts_batch_size", "tts_decode_video", "tts_video_feature",
            "tts_rank_fusion", "tts_video_weight", "tts_rrf_k",
            "tts_video_weight_low", "tts_gate_spearman_thresh", "tts_gate_distance_ratio_thresh",
        ]:
            if k in cfg:
                lines.append(f"- {k}: {cfg[k]}")
        lines.append("")

    lines.append("## Success")
    if success_df.empty:
        lines.append("- No success-rate lines parsed from log.")
    else:
        final = success_df.iloc[-1]
        lines.append(f"- final success: {int(final.cum_success)}/{int(final.cum_total)} = {100 * float(final.cum_rate):.1f}%")
        if not bins_df.empty:
            lines.append("- success by bins:")
            for _, r in bins_df.iterrows():
                lines.append(f"  - {int(r.bin_start)}-{int(r.bin_end)}: {int(r.success_sum)}/{int(r.n)} = {100 * r.success_rate:.1f}%")
    lines.append("")

    lines.append("## CSV / selector diagnostics")
    lines.append(f"- csv rows: {len(row_df)}")
    if "episode" in row_df.columns:
        lines.append(f"- episodes in csv: {row_df['episode'].nunique()}")
    key_metrics = [
        ("selected_action_best", "selected action rank 0 ratio"),
        ("action_override", "action override ratio"),
        ("selected_video_best", "selected video rank 0 ratio"),
        ("rank_agree_num", "action/video best agreement"),
        ("rank_spearman", "rank spearman mean"),
        ("distance_ratio_video_over_action", "distance ratio mean"),
        ("action_margin", "action margin mean"),
        ("video_margin", "video margin mean"),
        ("fused_margin", "fused margin mean"),
        ("fused_tie_count", "fused tie count mean"),
        ("selected_is_fused_best", "selected is fused best"),
        ("fusion_formula_ok", "fusion formula ok"),
        ("video_weight_used", "effective video weight mean"),
        ("gate_pass_num", "gate pass ratio"),
        ("candidate_pool_ratio", "candidate pool ratio mean"),
    ]
    for col, desc in key_metrics:
        if col in row_df.columns:
            val = safe_mean(row_df[col])
            if pd.notna(val):
                if "ratio" in desc or col in {"selected_is_fused_best", "fusion_formula_ok"}:
                    lines.append(f"- {desc}: {pct(val)}")
                else:
                    lines.append(f"- {desc}: {val:.6g}")

    if "selected_action_rank" in row_df.columns:
        counts = row_df["selected_action_rank"].dropna().astype(int).value_counts().sort_index()
        lines.append("- selected_action_rank counts: " + ", ".join(f"{k}:{v}" for k, v in counts.items()))
    if "selected_video_rank" in row_df.columns:
        counts = row_df["selected_video_rank"].dropna().astype(int).value_counts().sort_index()
        lines.append("- selected_video_rank counts: " + ", ".join(f"{k}:{v}" for k, v in counts.items()))

    lines.append("")
    lines.append("## Interpretation hints")
    override = safe_mean(row_df.get("action_override", pd.Series(dtype=float))) if "action_override" in row_df.columns else np.nan
    spearman = safe_mean(row_df.get("rank_spearman", pd.Series(dtype=float))) if "rank_spearman" in row_df.columns else np.nan
    ratio = safe_mean(row_df.get("distance_ratio_video_over_action", pd.Series(dtype=float))) if "distance_ratio_video_over_action" in row_df.columns else np.nan
    if pd.notna(override):
        if override < 0.02:
            lines.append("- action_override is near zero: video fusion is almost degenerated to action-only selection.")
        elif override < 0.25:
            lines.append("- action_override is moderate: video fusion changes some decisions but action rank still dominates.")
        else:
            lines.append("- action_override is high: video may be too influential; check success/failure and margins.")
    if pd.notna(spearman):
        lines.append(f"- mean rank_spearman={spearman:.3f}; higher means action/video rankings are more consistent.")
    if pd.notna(ratio):
        lines.append(f"- mean distance_ratio_video_over_action={ratio:.3f}; high values mean video latents are relatively dispersed.")
    if "gate_pass_num" in row_df.columns:
        gate = safe_mean(row_df["gate_pass_num"])
        if pd.notna(gate):
            lines.append(f"- mean gate_pass={gate:.3f}; only meaningful for video_gated_fusion.")
    if "candidate_pool_ratio" in row_df.columns:
        pool_ratio = safe_mean(row_df["candidate_pool_ratio"])
        if pd.notna(pool_ratio):
            lines.append(f"- mean candidate_pool_ratio={pool_ratio:.3f}; only meaningful for video_cluster_fusion.")

    return "\n".join(lines) + "\n"


# =========================
# Main per-run function
# =========================
def analyze_one_run(run, out_root):
    csv_path = run.get("csv")
    log_path = run.get("log")
    name = run.get("name") or derive_run_name(log_path, csv_path)
    name = safe_name(name)

    run_dir = Path(out_root) / name
    plots_dir, csv_dir = ensure_dirs(run_dir)

    cfg = parse_log_config(log_path)
    success_df = parse_success_log(log_path)
    bins_df = make_success_bins(success_df, SUCCESS_BIN_SIZE)
    row_df = load_and_derive(csv_path)
    ep_df = aggregate_episode(row_df, success_df)

    # save csv outputs
    row_df.to_csv(csv_dir / "row_metrics.csv", index=False)
    ep_df.to_csv(csv_dir / "episode_metrics.csv", index=False)
    success_df.to_csv(csv_dir / "success_by_episode.csv", index=False)
    bins_df.to_csv(csv_dir / "success_bins.csv", index=False)

    # plots
    plot_success(success_df, bins_df, plots_dir)
    plot_metrics(row_df, ep_df, plots_dir)

    # report
    report = summarize_run(name, cfg, success_df, bins_df, row_df, ep_df)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")

    return {
        "name": name,
        "run_dir": run_dir,
        "success_df": success_df,
        "row_df": row_df,
        "ep_df": ep_df,
    }


# =========================
# Optional comparison for multiple RUNS
# =========================
def compare_runs(results, out_root):
    if len(results) < 2:
        return
    compare_dir = Path(out_root) / "compare_plots"
    compare_dir.mkdir(parents=True, exist_ok=True)

    # compare selected episode metrics
    metrics = [
        ("selected_action_best", zh("选择 action best 比例对比", "Selected action-best ratio comparison"), (0, 1)),
        ("action_override", zh("action override 比例对比", "Action override ratio comparison"), (0, 1)),
        ("selected_video_best", zh("选择 video best 比例对比", "Selected video-best ratio comparison"), (0, 1)),
        ("rank_spearman", zh("rank Spearman 对比", "Rank Spearman comparison"), (-1, 1)),
        ("distance_ratio_video_over_action", zh("video/action 距离比对比", "Video/action distance ratio comparison"), None),
        ("video_weight_used", zh("实际 video weight 对比", "Effective video weight comparison"), None),
        ("gate_pass_num", zh("gate pass 比例对比", "Gate-pass ratio comparison"), (0, 1)),
        ("candidate_pool_ratio", zh("cluster 候选池比例对比", "Candidate-pool ratio comparison"), (0, 1)),
    ]
    for col, title, ylim in metrics:
        plt.figure(figsize=(10, 5))
        ok = False
        for res in results:
            ep = res["ep_df"]
            if ep.empty or col not in ep.columns:
                continue
            d = ep[["episode", col]].dropna()
            if d.empty:
                continue
            plt.plot(d["episode"], d[col], marker="o", linewidth=1.2, markersize=3, label=res["name"])
            ok = True
        if not ok:
            plt.close()
            continue
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel(col)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_fig(compare_dir / f"compare_{col}.png", zh("多个 run 的 episode-level 指标对比。注意只有相同 seed 才能做严格 paired 比较。", "Compares episode-level metrics across runs. Strict comparison requires identical seeds."))


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Motus TTS summary.csv and outer log.")
    parser.add_argument("csv_pos", nargs="?", default=None, help="summary.csv path")
    parser.add_argument("log_pos", nargs="?", default=None, help="outer log path")
    parser.add_argument("--csv", dest="csv_opt", default=None, help="summary.csv path")
    parser.add_argument("--log", dest="log_opt", default=None, help="outer log path")
    parser.add_argument("--name", default=None, help="run name; default: inferred from log filename")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_DIR, help="output root directory")
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv_opt or args.csv_pos
    log_path = args.log_opt or args.log_pos
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    runs = []
    if csv_path and log_path:
        runs.append({"name": args.name, "csv": csv_path, "log": log_path})
    else:
        runs.extend(RUNS)

    if not runs:
        raise SystemExit(
            "Need CSV and log. Use either:\n"
            "  python tts_log_visualizer_zh_v3.py SUMMARY.csv OUTER.log\n"
            "or:\n"
            "  python tts_log_visualizer_zh_v3.py --csv SUMMARY.csv --log OUTER.log"
        )

    print(f"Chinese font found: {HAS_CJK_FONT}; font: {CJK_FONT_NAME}; plot language: {PLOT_LANG}")
    print(f"Output root: {out_root}")

    results = []
    for run in runs:
        if not run.get("csv") or not Path(run["csv"]).exists():
            raise FileNotFoundError(f"CSV not found: {run.get('csv')}")
        if not run.get("log") or not Path(run["log"]).exists():
            raise FileNotFoundError(f"Log not found: {run.get('log')}")
        res = analyze_one_run(run, out_root)
        results.append(res)
        print(f"  - {res['name']}: {res['run_dir'] / 'report.txt'}")

    compare_runs(results, out_root)
    print("Done.")


if __name__ == "__main__":
    main()
