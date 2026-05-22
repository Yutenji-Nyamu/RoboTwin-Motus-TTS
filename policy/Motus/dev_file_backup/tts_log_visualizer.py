#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS log and CSV visualizer for Motus / RoboTwin runs.

Usage options:
1) Edit CONFIG at the top and run:
   python tts_log_visualizer.py

2) Analyze one run from command line:
   python tts_log_visualizer.py --csv /path/to/summary.csv --log /path/to/run.log --name my_run --out tts_report

3) Compare multiple runs by editing RUNS in CONFIG.

No environment variables are required.
"""

# =========================
# CONFIG
# =========================
RUNS = [
    {
        "name": "weighted_borda_w05",
        "csv": "/mnt/data/summary_024039.csv",
        "log": "/mnt/data/scan_object_tts_video_rank_weighted_borda_n4_b4_w05_latent_nodecode_20260522_024038.log",
    },
    {
        "name": "rrf_w05_k1",
        "csv": "/mnt/data/summary_0522023744.csv",
        "log": "/mnt/data/scan_object_tts_video_rank_rrf_n4_b4_w05_k1_latent_nodecode_20260522_023744.log",
    },
]
OUTPUT_DIR = "/mnt/data/tts_report"
ROLLING_EPISODE_WINDOW = 10
ROLLING_ROW_WINDOW = 50
SUCCESS_BIN_SIZE = 10

# =========================
# Imports
# =========================
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Parsing helpers
# =========================
def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_log(log_path: Optional[str]) -> pd.DataFrame:
    if not log_path:
        return pd.DataFrame()
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame()

    text = strip_ansi(path.read_text(errors="ignore"))
    rows = []
    last_status = None
    for line in text.splitlines():
        if "Success!" in line:
            last_status = 1
        elif "Fail!" in line:
            last_status = 0

        m = re.search(
            r"Success rate:\s*(\d+)/(\d+)\s*=>\s*([0-9.]+)%.*, current seed:\s*(\d+)",
            line,
        )
        if m:
            rows.append(
                {
                    "episode": int(m.group(2)),
                    "success": int(last_status) if last_status is not None else np.nan,
                    "cum_success": int(m.group(1)),
                    "cum_rate": float(m.group(3)),
                    "seed": int(m.group(4)),
                }
            )
            last_status = None

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["cum_rate_calc"] = out["cum_success"] / out["episode"]
    return out


def parse_pipe_float(x) -> List[float]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    vals = []
    for part in s.split("|"):
        if part == "":
            continue
        try:
            vals.append(float(part))
        except ValueError:
            pass
    return vals


def parse_pipe_int(x) -> List[int]:
    vals = []
    for v in parse_pipe_float(x):
        vals.append(int(round(v)))
    return vals


def compute_row_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    action_margins = []
    video_margins = []
    fused_margins = []
    fused_ties = []
    formula_ok = []
    selection_ok = []

    for _, row in out.iterrows():
        action_vals = parse_pipe_float(row.get("action_avg_l2", np.nan))
        video_vals = parse_pipe_float(row.get("video_avg_l2", np.nan))
        action_ranks = np.array(parse_pipe_int(row.get("action_rank_ids", np.nan)), dtype=float)
        video_ranks = np.array(parse_pipe_int(row.get("video_rank_ids", np.nan)), dtype=float)
        fused_scores = np.array(parse_pipe_float(row.get("fused_scores", np.nan)), dtype=float)

        if len(action_vals) >= 2:
            s = np.sort(np.array(action_vals, dtype=float))
            action_margins.append(float(s[1] - s[0]))
        else:
            action_margins.append(np.nan)

        if len(video_vals) >= 2:
            s = np.sort(np.array(video_vals, dtype=float))
            video_margins.append(float(s[1] - s[0]))
        else:
            video_margins.append(np.nan)

        if len(action_ranks) and len(video_ranks) and len(fused_scores) == len(action_ranks):
            method = str(row.get("rank_fusion_method", "")).strip().lower()
            video_weight = float(row.get("video_weight", 0.0))
            rrf_k = float(row.get("rrf_k", 1.0))
            if method == "rrf":
                calc = 1.0 / (rrf_k + action_ranks + 1.0) + video_weight / (rrf_k + video_ranks + 1.0)
                higher_is_better = True
            else:
                calc = action_ranks + video_weight * video_ranks
                higher_is_better = False

            formula_ok.append(bool(np.allclose(calc, fused_scores, rtol=1e-4, atol=1e-4)))
            selected_idx = int(row.get("selected_idx", -1))
            if higher_is_better:
                best_val = np.max(calc)
                best_idxs = np.flatnonzero(np.isclose(calc, best_val, rtol=1e-6, atol=1e-8))
                sorted_vals = np.sort(calc)[::-1]
                margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else np.nan
            else:
                best_val = np.min(calc)
                best_idxs = np.flatnonzero(np.isclose(calc, best_val, rtol=1e-6, atol=1e-8))
                sorted_vals = np.sort(calc)
                margin = sorted_vals[1] - sorted_vals[0] if len(sorted_vals) > 1 else np.nan

            selection_ok.append(bool(selected_idx in set(best_idxs.tolist())))
            fused_ties.append(int(len(best_idxs)))
            fused_margins.append(float(margin))
        else:
            formula_ok.append(np.nan)
            selection_ok.append(np.nan)
            fused_ties.append(np.nan)
            fused_margins.append(np.nan)

    out["action_margin"] = action_margins
    out["video_margin"] = video_margins
    out["fused_margin"] = fused_margins
    out["fused_tie_count"] = fused_ties
    out["fusion_formula_ok"] = formula_ok
    out["fusion_selection_ok"] = selection_ok
    out["selected_action_best"] = (out.get("selected_action_rank", pd.Series(index=out.index)) == 0).astype(float)
    out["selected_video_best"] = (out.get("selected_video_rank", pd.Series(index=out.index)) == 0).astype(float)
    out["action_override"] = (out.get("selected_action_rank", pd.Series(index=out.index)) > 0).astype(float)
    return out


# =========================
# Plot helpers
# =========================
def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_line(x, y, title, xlabel, ylabel, path: Path):
    plt.figure(figsize=(9, 4.8))
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_plot(path)


def plot_multi_line(series: Dict[str, pd.DataFrame], x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path):
    plt.figure(figsize=(9, 4.8))
    for name, frame in series.items():
        if frame.empty or x_col not in frame or y_col not in frame:
            continue
        plt.plot(frame[x_col], frame[y_col], marker="o", linewidth=1, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(path)


def plot_bar(x, y, title, xlabel, ylabel, path: Path):
    plt.figure(figsize=(9, 4.8))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path)


def plot_hist(values, bins, title, xlabel, ylabel, path: Path):
    plt.figure(figsize=(8, 4.8))
    plt.hist(values.dropna() if hasattr(values, "dropna") else values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path)


def plot_box_by_success(episode_df: pd.DataFrame, metric: str, path: Path):
    if episode_df.empty or "success" not in episode_df or metric not in episode_df:
        return
    data = []
    labels = []
    for val, label in [(0, "fail"), (1, "success")]:
        arr = episode_df.loc[episode_df["success"] == val, metric].dropna().to_numpy()
        if len(arr):
            data.append(arr)
            labels.append(label)
    if not data:
        return
    plt.figure(figsize=(7, 4.8))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.title(f"{metric}: success vs fail")
    plt.xlabel("episode outcome")
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path)


# =========================
# Analysis
# =========================
def make_success_bins(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame()
    rows = []
    max_ep = int(log_df["episode"].max())
    for start in range(1, max_ep + 1, SUCCESS_BIN_SIZE):
        end = min(start + SUCCESS_BIN_SIZE - 1, max_ep)
        sub = log_df[(log_df["episode"] >= start) & (log_df["episode"] <= end)]
        rows.append(
            {
                "episode_start": start,
                "episode_end": end,
                "success_count": int(sub["success"].sum()),
                "episode_count": int(len(sub)),
                "success_rate": float(sub["success"].mean()),
            }
        )
    return pd.DataFrame(rows)


def episode_level_metrics(df: pd.DataFrame, log_df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "step": "count",
        "rank_agree": "mean",
        "rank_spearman": "mean",
        "distance_ratio_video_over_action": "mean",
        "selected_action_best": "mean",
        "selected_video_best": "mean",
        "action_override": "mean",
        "action_avg_l2_mean": "mean",
        "video_avg_l2_mean": "mean",
        "action_avg_l2_std": "mean",
        "video_avg_l2_std": "mean",
        "action_margin": "mean",
        "video_margin": "mean",
        "fused_margin": "mean",
        "fused_tie_count": "mean",
        "fusion_formula_ok": "mean",
        "fusion_selection_ok": "mean",
    }
    existing = {k: v for k, v in agg_spec.items() if k in df.columns}
    ep = df.groupby("episode").agg(existing).reset_index()
    if "step" in ep:
        ep = ep.rename(columns={"step": "tts_calls"})
    if not log_df.empty:
        ep = ep.merge(log_df[["episode", "success", "cum_success", "cum_rate", "seed"]], on="episode", how="left")
    return ep


def write_field_guide(path: Path):
    text = """# TTS CSV 字段可视化建议

| 字段 | 建议可视化 | 目的 |
|---|---|---|
| episode | 作为横轴或分组键 | 看 100 个 episode 的阶段性变化 |
| step | 作为 episode 内横轴 | 看同一 episode 内 TTS 调用是否漂移 |
| samples | 常量检查 | 确认 K 是否等于命令行设置 |
| tts_method, selection_stage | value_counts | 确认实际跑的是哪个 selector |
| selected_idx | 直方图 | 看候选 index 是否存在偏置 |
| selected_rank, selected_fused_rank | 直方图 | 检查是否总是选 fused rank 0，是否有异常 |
| selected_prob, selection_probs | 直方图或跳过 | 确定性方法通常全为 1，rank_softmax 时才重要 |
| global_medoid_idx, action_best_idx | 一致性检查和直方图 | 确认 global_medoid 是否等于 action best |
| global_medoid_avg_l2 | episode 曲线 | 看动作候选共识强弱 |
| unimodal, s_score | KeyStone/cluster 方法画曲线 | video_rank_fusion 中一般为空 |
| num_clusters, selected_cluster, selected_cluster_size, cluster_counts, cluster_ids | cluster 方法画分布 | 判断是否真的发生多峰聚类 |
| tau, kmeans_iters, rank_temperature | 常量检查 | 防止日志和命令行不一致 |
| rank_ids | 直方图或跳过 | 旧字段，video_rank_fusion 中看 fused rank 更清楚 |
| selected_avg_l2, min_avg_l2, max_avg_l2, mean_avg_l2, avg_l2 | 曲线和 margin | 判断动作候选是否足够分散，selector 是否只选 min avg L2 |
| rank_fusion_method, video_feature_type, video_weight, rrf_k, fused_score_direction | 常量检查 | 确认 fusion 配置 |
| action_feature_dim, video_feature_dim | 常量检查 | 确认动作和视频特征维度没有变化 |
| action_best_idx, video_best_idx, rank_agree | agreement 曲线 | 判断 action 和 video 是否在支持同一个候选 |
| rank_spearman | episode 曲线和直方图 | 判断 action rank 与 video rank 的整体相关性 |
| selected_action_rank, selected_video_rank | 直方图和 episode 曲线 | 判断融合是否真的覆盖 action best 或 video best |
| selected_fused_score, fused_scores | margin 曲线 | 判断融合选择是否稳定，是否大量 tie |
| action_rank_ids, video_rank_ids, fused_rank_ids | 一致性检查 | 判断 rank 计算是否符合 avg L2 排序 |
| action_avg_l2_min/max/mean/std, action_avg_l2 | 曲线和 margin | 判断动作候选分布形态和分离度 |
| video_avg_l2_min/max/mean/std, video_avg_l2 | 曲线和 margin | 判断视频 latent 候选分布形态和分离度 |
| action_pairwise_median, video_pairwise_median, distance_ratio_video_over_action | 曲线 | 判断两种模态距离尺度是否漂移 |
| action_feature_norm_mean, video_feature_norm_mean | 曲线 | 检查特征范数是否异常漂移 |
"""
    path.write_text(text, encoding="utf-8")


def analyze_run(run: Dict[str, str], out_root: Path) -> Dict[str, pd.DataFrame]:
    name = run["name"]
    run_dir = out_root / name
    plot_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run["csv"])
    df = compute_row_diagnostics(df)
    log_df = parse_log(run.get("log"))
    bins_df = make_success_bins(log_df)
    ep_df = episode_level_metrics(df, log_df)

    df.to_csv(run_dir / "row_metrics.csv", index=False)
    log_df.to_csv(run_dir / "success_by_episode.csv", index=False)
    bins_df.to_csv(run_dir / "success_bins.csv", index=False)
    ep_df.to_csv(run_dir / "episode_metrics.csv", index=False)

    # Success plots.
    if not log_df.empty:
        plot_line(
            log_df["episode"], log_df["cum_rate"],
            f"{name}: cumulative success rate", "episode", "cumulative success rate (%)",
            plot_dir / "success_cumulative_rate.png",
        )
        if not bins_df.empty:
            xlabels = [f"{int(a)}-{int(b)}" for a, b in zip(bins_df["episode_start"], bins_df["episode_end"])]
            plot_bar(
                xlabels, bins_df["success_rate"] * 100,
                f"{name}: {SUCCESS_BIN_SIZE}-episode success rate", "episode bin", "success rate (%)",
                plot_dir / "success_bins.png",
            )
        plot_bar(
            log_df["episode"], log_df["success"],
            f"{name}: binary success sequence", "episode", "success",
            plot_dir / "success_sequence.png",
        )

    # Core CSV plots.
    if not ep_df.empty:
        plot_bar(ep_df["episode"], ep_df["tts_calls"], f"{name}: TTS calls per episode", "episode", "TTS calls", plot_dir / "tts_calls_per_episode.png")
        for metric in [
            "rank_agree", "rank_spearman", "distance_ratio_video_over_action",
            "selected_action_best", "selected_video_best", "action_override",
            "action_avg_l2_mean", "video_avg_l2_mean", "action_avg_l2_std", "video_avg_l2_std",
            "action_margin", "video_margin", "fused_margin", "fused_tie_count",
        ]:
            if metric in ep_df.columns:
                plot_line(ep_df["episode"], ep_df[metric], f"{name}: {metric} per episode", "episode", metric, plot_dir / f"{metric}_per_episode.png")

        for metric in ["tts_calls", "rank_agree", "rank_spearman", "distance_ratio_video_over_action", "selected_action_best", "selected_video_best", "action_margin", "video_margin", "fused_margin"]:
            plot_box_by_success(ep_df, metric, plot_dir / f"{metric}_success_vs_fail.png")

    # Row-level histograms.
    for col in ["selected_idx", "selected_action_rank", "selected_video_rank", "selected_fused_rank", "rank_spearman", "distance_ratio_video_over_action", "action_margin", "video_margin", "fused_margin"]:
        if col in df.columns:
            values = df[col]
            if pd.api.types.is_numeric_dtype(values):
                plot_hist(values, bins=30, title=f"{name}: {col} histogram", xlabel=col, ylabel="count", path=plot_dir / f"{col}_hist.png")

    # Text report.
    with open(run_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"Run: {name}\n")
        f.write(f"CSV: {run['csv']}\n")
        f.write(f"Log: {run.get('log', '')}\n\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Episodes in CSV: {df['episode'].nunique()}\n")
        if not log_df.empty:
            final = log_df.iloc[-1]
            f.write(f"Final success: {int(final['cum_success'])}/{int(final['episode'])} = {float(final['cum_rate']):.1f}%\n")
        f.write("\nConstants / value counts:\n")
        for col in ["samples", "tts_method", "selection_stage", "rank_fusion_method", "video_feature_type", "video_weight", "rrf_k", "fused_score_direction", "action_feature_dim", "video_feature_dim"]:
            if col in df.columns:
                f.write(f"{col}: {df[col].value_counts(dropna=False).head(20).to_dict()}\n")
        f.write("\nSanity checks:\n")
        for col in ["fusion_formula_ok", "fusion_selection_ok"]:
            if col in df.columns:
                f.write(f"{col}: mean={df[col].mean():.6f}, bad={(df[col] == False).sum()}\n")
        f.write(f"fused tie rows: {(df['fused_tie_count'] > 1).sum()} / {len(df)}\n")
        f.write("\nKey row-level means:\n")
        for col in ["rank_agree", "rank_spearman", "selected_action_best", "selected_video_best", "action_override", "distance_ratio_video_over_action", "action_margin", "video_margin", "fused_margin"]:
            if col in df.columns:
                f.write(f"{col}: mean={df[col].mean():.6f}, median={df[col].median():.6f}\n")
        if not bins_df.empty:
            f.write("\nSuccess bins:\n")
            f.write(bins_df.to_string(index=False))
            f.write("\n")

    return {"rows": df, "log": log_df, "bins": bins_df, "episode": ep_df}


def make_compare_outputs(results: Dict[str, Dict[str, pd.DataFrame]], out_root: Path):
    compare_dir = out_root / "compare_plots"
    compare_dir.mkdir(parents=True, exist_ok=True)

    log_frames = {name: d["log"] for name, d in results.items() if not d["log"].empty}
    if log_frames:
        plot_multi_line(log_frames, "episode", "cum_rate", "Comparison: cumulative success rate", "episode", "cumulative success rate (%)", compare_dir / "compare_cumulative_success.png")

        bin_frames = {}
        for name, d in results.items():
            b = d["bins"]
            if not b.empty:
                b = b.copy()
                b["bin_mid"] = (b["episode_start"] + b["episode_end"]) / 2
                b["success_rate_pct"] = b["success_rate"] * 100
                bin_frames[name] = b
        if bin_frames:
            plot_multi_line(bin_frames, "bin_mid", "success_rate_pct", f"Comparison: {SUCCESS_BIN_SIZE}-episode success rate", "episode bin midpoint", "success rate (%)", compare_dir / "compare_success_bins.png")

    ep_frames = {name: d["episode"] for name, d in results.items() if not d["episode"].empty}
    for metric in ["rank_agree", "rank_spearman", "selected_action_best", "selected_video_best", "distance_ratio_video_over_action", "action_margin", "video_margin", "fused_margin", "tts_calls"]:
        frames = {name: frame for name, frame in ep_frames.items() if metric in frame.columns}
        if frames:
            plot_multi_line(frames, "episode", metric, f"Comparison: {metric}", "episode", metric, compare_dir / f"compare_{metric}.png")

    # Common seed comparison if at least two logs exist.
    names = list(log_frames.keys())
    if len(names) >= 2:
        base_name = names[0]
        base = log_frames[base_name][["seed", "episode", "success"]].rename(columns={"episode": f"episode_{base_name}", "success": f"success_{base_name}"})
        merged = base.copy()
        for name in names[1:]:
            cur = log_frames[name][["seed", "episode", "success"]].rename(columns={"episode": f"episode_{name}", "success": f"success_{name}"})
            merged = merged.merge(cur, on="seed", how="inner")
        merged.to_csv(out_root / "common_seed_comparison.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="", help="Path to one summary.csv file")
    parser.add_argument("--log", type=str, default="", help="Path to the matching outer .log file")
    parser.add_argument("--name", type=str, default="single_run", help="Name for one-run analysis")
    parser.add_argument("--out", type=str, default=OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    write_field_guide(out_root / "FIELD_VISUALIZATION_GUIDE.md")

    if args.csv:
        runs = [{"name": args.name, "csv": args.csv, "log": args.log}]
    else:
        runs = RUNS

    results = {}
    for run in runs:
        results[run["name"]] = analyze_run(run, out_root)

    if len(results) > 1:
        make_compare_outputs(results, out_root)

    print(f"Wrote report to: {out_root}")
    for run in runs:
        print(f"  - {run['name']}: {out_root / run['name'] / 'report.txt'}")


if __name__ == "__main__":
    main()
