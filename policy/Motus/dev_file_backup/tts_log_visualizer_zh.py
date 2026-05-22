#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motus / RoboTwin TTS 日志与 CSV 可视化脚本，带中文图内注释。

用法 1，直接改顶部 CONFIG 后运行：
    python tts_log_visualizer_zh.py

用法 2，命令行分析单个 run：
    python tts_log_visualizer_zh.py \
      --csv /path/to/summary.csv \
      --log /path/to/outer.log \
      --name my_run \
      --out /path/to/tts_report_zh

该脚本不依赖环境变量。所有可配置项都在顶部 CONFIG 区。
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
OUTPUT_DIR = "/mnt/data/tts_report_zh"
SUCCESS_BIN_SIZE = 10
HIST_BINS = 30
FIG_DPI = 130
ANNOTATE_PLOTS = True
SAVE_PLOT_NOTES_TXT = False

# 画图尺寸。底部留白用于中文注释。
FIGSIZE_LINE = (10.5, 6.0)
FIGSIZE_HIST = (9.5, 5.8)
FIGSIZE_BOX = (8.8, 5.8)
FIGSIZE_BAR = (10.5, 6.0)

# =========================
# Imports
# =========================
import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# =========================
# Matplotlib Chinese font
# =========================
def setup_chinese_font():
    """尽量自动选择系统中已有中文字体。不打包也不导出任何字体文件。"""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "AR PL UMing CN",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


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
    return [int(round(v)) for v in parse_pipe_float(x)]


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


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
            video_weight = safe_float(row.get("video_weight", 0.0), 0.0)
            rrf_k = safe_float(row.get("rrf_k", 1.0), 1.0)
            if method == "rrf":
                calc = 1.0 / (rrf_k + action_ranks + 1.0) + video_weight / (rrf_k + video_ranks + 1.0)
                higher_is_better = True
            else:
                calc = action_ranks + video_weight * video_ranks
                higher_is_better = False

            formula_ok.append(bool(np.allclose(calc, fused_scores, rtol=1e-4, atol=1e-4)))
            selected_idx = int(safe_float(row.get("selected_idx", -1), -1))
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

    selected_action_rank = pd.to_numeric(out.get("selected_action_rank", pd.Series(index=out.index)), errors="coerce")
    selected_video_rank = pd.to_numeric(out.get("selected_video_rank", pd.Series(index=out.index)), errors="coerce")
    out["selected_action_best"] = (selected_action_rank == 0).astype(float)
    out["selected_video_best"] = (selected_video_rank == 0).astype(float)
    out["action_override"] = (selected_action_rank > 0).astype(float)
    return out


# =========================
# Plot annotation helpers
# =========================
def wrap_note(note: str, width: int = 70) -> str:
    if not note:
        return ""
    lines = []
    for paragraph in str(note).split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        lines.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False))
    return "\n".join(lines)


def add_note(fig, note: str):
    if not ANNOTATE_PLOTS or not note:
        return
    fig.subplots_adjust(bottom=0.30)
    fig.text(
        0.02,
        0.02,
        wrap_note(note),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f6f6f6", edgecolor="#bbbbbb", alpha=0.95),
    )


def save_plot(path: Path, note: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    add_note(fig, note)
    plt.savefig(path, dpi=FIG_DPI)
    if SAVE_PLOT_NOTES_TXT and note:
        path.with_suffix(".txt").write_text(note, encoding="utf-8")
    plt.close()


def finite_series(values) -> pd.Series:
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def describe_numeric(values) -> str:
    s = finite_series(values)
    if len(s) == 0:
        return "无有效数值。"
    return f"均值 {s.mean():.4g}，中位数 {s.median():.4g}，范围 [{s.min():.4g}, {s.max():.4g}]。"


def metric_mean(frame: pd.DataFrame, metric: str) -> float:
    if frame.empty or metric not in frame:
        return np.nan
    return float(pd.to_numeric(frame[metric], errors="coerce").mean())


def note_for_metric(metric: str, values=None, run_name: str = "") -> str:
    prefix = f"{run_name}：" if run_name else ""
    stats = describe_numeric(values) if values is not None else ""
    if metric == "selected_action_best":
        return prefix + stats + " 该值表示最终选择是否仍是 action-space 全局 medoid。接近 1 说明方法几乎退化为 action-only；低于 1 说明 video rank 有机会改变动作选择。"
    if metric == "action_override":
        return prefix + stats + " action override = selected_action_rank > 0，即最终融合没有选 action best，而是被视频排序或融合项推到其他候选。0 说明视频没有实际介入选择。"
    if metric == "selected_video_best":
        return prefix + stats + " 该值表示最终选择是否也是 video-space medoid。越高说明最终候选更符合视频 latent 共识，但不等价于任务一定成功。"
    if metric == "rank_agree":
        return prefix + stats + " rank_agree 表示 action best 与 video best 是否为同一个候选。低说明两种模态常有分歧。"
    if metric in ["rank_spearman", "spearman_rank_corr"]:
        return prefix + stats + " Spearman 衡量 action rank 与 video rank 的整体一致性。正值越高，说明动作共识和视频共识越一致；负值表示两者排序冲突。K=4 时取值会比较离散。"
    if metric == "distance_ratio_video_over_action":
        return prefix + stats + " 这是 video pairwise median / action pairwise median。因为当前融合用 rank，所以该尺度不会直接支配选择；它主要诊断视频 latent 相对动作候选是否更分散。失败 episode 偏高时，可能表示未来视频预测更不稳定或场景更难。"
    if metric == "action_margin":
        return prefix + stats + " action_margin 是 action avg-L2 第一名和第二名的差。越大说明 action medoid 更明确；越接近 0 说明动作候选之间难分高下，视频项更容易改变最终选择。"
    if metric == "video_margin":
        return prefix + stats + " video_margin 是 video avg-L2 第一名和第二名的差。越大说明视频 latent 共识更明确；但如果失败中更大，可能表示视频共识明确但不一定与成功动作一致。"
    if metric == "fused_margin":
        return prefix + stats + " fused_margin 是融合分数第一名和第二名的差。越小越不稳定，接近 0 代表 tie 或近似 tie。不同融合公式的分数尺度不同，Borda 与 RRF 的绝对数值不能直接比较。"
    if metric == "fused_tie_count":
        return prefix + stats + " fused_tie_count > 1 表示多个候选融合分数并列最优，代码通常按索引顺序取第一个，可能引入 index bias。"
    if metric == "tts_calls":
        return prefix + stats + " TTS calls 多通常表示 episode 跑得更久。失败 episode 往往跑满步数，因此按 row 统计会被失败 episode 过度加权。"
    if metric == "cum_rate":
        return prefix + "累计成功率曲线会受 seed 顺序影响。若前段 seed 特别难或特别简单，曲线会先低后高或先高后低；严肃比较要用相同 seed list。"
    if metric == "success_rate_pct":
        return prefix + f"每 {SUCCESS_BIN_SIZE} 个 episode 的局部成功率，用来检查 seed 难度是否分布不均。"
    if metric == "selected_idx":
        return prefix + stats + " 候选 index 的直方图用于检查采样或 tie-breaking 是否偏向某个 index。"
    if metric in ["selected_action_rank", "selected_video_rank", "selected_fused_rank"]:
        return prefix + stats + " rank=0 表示该模态或融合分数下的第一名。分布越集中在 0，选择越保守；出现高 rank 说明融合或采样改变了原始排序。"
    return prefix + stats


def note_for_success_box(episode_df: pd.DataFrame, metric: str) -> str:
    if episode_df.empty or "success" not in episode_df or metric not in episode_df:
        return ""
    fail = finite_series(episode_df.loc[episode_df["success"] == 0, metric])
    succ = finite_series(episode_df.loc[episode_df["success"] == 1, metric])
    if len(fail) == 0 or len(succ) == 0:
        return note_for_metric(metric, episode_df[metric])
    base = f"失败均值 {fail.mean():.4g}，成功均值 {succ.mean():.4g}。"
    warn = " 注意：成功 episode 通常更早结束，失败 episode 往往有更多 TTS 调用，所以这是相关性诊断，不是因果结论。"
    return base + " " + note_for_metric(metric, episode_df[metric]) + warn


def note_for_compare(metric: str, frames: Dict[str, pd.DataFrame]) -> str:
    means = []
    for name, frame in frames.items():
        if metric in frame:
            means.append(f"{name} 均值 {metric_mean(frame, metric):.4g}")
    head = "；".join(means) + "。" if means else ""
    if metric == "selected_action_best":
        return head + "若某条曲线长期等于 1，说明该方法总是选择 action best，视频分支没有真正覆盖 action-only 决策。"
    if metric == "action_override":
        return head + "这是视频融合改变 action best 的比例。0 表示退化为 action-only；过高则可能说明视频项过强。"
    if metric == "fused_margin":
        return head + "Borda 和 RRF 分数尺度不同，不能直接比较绝对大小。重点看同一方法内是否接近 0，以及是否出现 tie。"
    if metric == "rank_spearman":
        return head + "越高表示 action/video 排序越一致。若成功 episode 更高，说明模态一致性可能是有用的置信信号。"
    if metric == "distance_ratio_video_over_action":
        return head + "该比值主要用于诊断视频 latent 相对动作候选的离散程度。rank 融合不会被原始尺度直接支配。"
    if metric == "success_rate_pct":
        return head + "局部成功率用于检查 seed 分布是否阶段性偏难或偏易。"
    if metric == "cum_rate":
        return head + "累计成功率不能替代 paired seed 对照。若 seed 序列不同，曲线差异包含 seed 难度差异。"
    return head + note_for_metric(metric)


# =========================
# Plot helpers
# =========================
def plot_line(x, y, title, xlabel, ylabel, path: Path, note: str = ""):
    plt.figure(figsize=FIGSIZE_LINE)
    plt.plot(x, y, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    save_plot(path, note)


def plot_multi_line(series: Dict[str, pd.DataFrame], x_col: str, y_col: str, title: str, xlabel: str, ylabel: str, path: Path, note: str = ""):
    plt.figure(figsize=FIGSIZE_LINE)
    for name, frame in series.items():
        if frame.empty or x_col not in frame or y_col not in frame:
            continue
        plt.plot(frame[x_col], frame[y_col], marker="o", linewidth=1, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(path, note)


def plot_bar(x, y, title, xlabel, ylabel, path: Path, note: str = ""):
    plt.figure(figsize=FIGSIZE_BAR)
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    if len(x) > 12:
        plt.xticks(rotation=45, ha="right")
    save_plot(path, note)


def plot_hist(values, bins, title, xlabel, ylabel, path: Path, note: str = ""):
    vals = finite_series(values)
    if len(vals) == 0:
        return
    plt.figure(figsize=FIGSIZE_HIST)
    plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path, note)


def plot_box_by_success(episode_df: pd.DataFrame, metric: str, path: Path):
    if episode_df.empty or "success" not in episode_df or metric not in episode_df:
        return
    data = []
    labels = []
    for val, label in [(0, "fail 失败"), (1, "success 成功")]:
        arr = finite_series(episode_df.loc[episode_df["success"] == val, metric]).to_numpy()
        if len(arr):
            data.append(arr)
            labels.append(label)
    if not data:
        return
    plt.figure(figsize=FIGSIZE_BOX)
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.title(f"{metric}: 成功 vs 失败")
    plt.xlabel("episode outcome")
    plt.ylabel(metric)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(path, note_for_success_box(episode_df, metric))


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
    # 将 row-level TTS 记录聚合到 episode 级，避免失败 episode 因为步数多而支配总体判断。
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
    text = """# TTS CSV 字段可视化建议，中文注释版

## 最优先看的图

1. `compare_cumulative_success.png` 和 `compare_success_bins.png`：看成功率随 episode 的变化，以及 seed 阶段性难度是否不均匀。
2. `compare_selected_action_best.png` 和单 run 的 `action_override_per_episode.png`：看 video fusion 是否真的改变了 action-only 选择。
3. `compare_rank_spearman.png`：看 action rank 与 video rank 是否一致。
4. `compare_fused_margin.png` 与 `fused_tie_count_per_episode.png`：看融合决策是否稳定，以及是否存在大量并列最优。
5. `distance_ratio_video_over_action_*`：看视频 latent 相对动作候选的离散度，辅助判断视频分支是否稳定。

## 字段解释

| 字段 | 建议可视化 | 目的 |
|---|---|---|
| episode | 横轴或分组键 | 看 100 个 episode 的阶段性变化 |
| step | episode 内横轴 | 看同一 episode 内 TTS 调用是否漂移 |
| samples | 常量检查 | 确认 K 是否等于命令行设置 |
| tts_method, selection_stage | value_counts | 确认实际跑的是哪个 selector |
| selected_idx | 直方图 | 看候选 index 是否存在偏置 |
| selected_action_rank | 直方图和 episode 曲线 | rank=0 表示选 action medoid；大于 0 表示 video/fusion 覆盖了 action-only |
| selected_video_rank | 直方图和 episode 曲线 | rank=0 表示选 video medoid；越高说明越偏离视频共识 |
| selected_fused_rank | 直方图 | 正常应为 0；不是 0 则要检查 selector 或日志 |
| action_override | episode 曲线 | selected_action_rank > 0 的比例；衡量 video 是否实际介入 |
| rank_agree | episode 曲线 | action best 与 video best 是否相同 |
| rank_spearman | 曲线和直方图 | action rank 与 video rank 的整体相关性 |
| action_margin | 曲线、直方图、成功失败箱线图 | action 第一名与第二名 avg-L2 差距；越小越容易被视频项覆盖 |
| video_margin | 曲线、直方图、成功失败箱线图 | video 第一名与第二名 avg-L2 差距；越大说明视频共识更明确 |
| fused_margin | 曲线和直方图 | 融合第一名与第二名差距；接近 0 表示不稳定或 tie |
| fused_tie_count | 曲线 | 大于 1 表示并列最优，可能出现 index bias |
| distance_ratio_video_over_action | 曲线、直方图、成功失败箱线图 | video pairwise median / action pairwise median；诊断模态尺度和视频 latent 离散度 |
| action_feature_norm_mean, video_feature_norm_mean | 曲线 | 检查特征范数是否异常漂移 |
| s_score, cluster_counts, selected_cluster | 仅 cluster/KeyStone 方法画 | 当前 video_rank_fusion 通常为空 |

## 读图注意

- 不要用 row-level 统计直接比较成功/失败，因为失败 episode 往往跑满步数，会产生更多 TTS 行。
- 优先看 episode-level 聚合图。
- Borda 与 RRF 的 fused score 尺度不同，`fused_margin` 的绝对值不能跨方法直接比较。
- 如果 `selected_action_best` 长期等于 1 且 `action_override` 等于 0，说明 video fusion 在实际选择上没有生效。
"""
    path.write_text(text, encoding="utf-8")


def write_rrf_tuning_note(path: Path):
    text = """# RRF 调参说明

当前 RRF 公式为：

score_i = 1 / (rrf_k + action_rank_i + 1) + video_weight / (rrf_k + video_rank_i + 1)

分数越大越好。K=4 时 rank 只有 0,1,2,3。

在当前配置 rrf_k=1, video_weight=0.5 下，最极端情况也很难让视频覆盖 action best：

- action best 但 video 最差：1/(1+0+1) + 0.5/(1+3+1) = 0.5 + 0.1 = 0.6
- action 第二但 video 最好：1/(1+1+1) + 0.5/(1+0+1) = 0.333 + 0.25 = 0.583

所以 action 第二名即使 video rank=0，也打不过 action 第一名。这解释了为什么当前 RRF 的 action_override=0。

结论：如果目标是让 video rank 真正介入，不能继续用 rrf_k=1, video_weight=0.5。

建议 sweep：

1. 固定 rrf_k=1，扫 video_weight = 0.5, 0.75, 1.0, 1.5。
2. 固定 video_weight=0.5，扫 rrf_k = 1, 2, 4, 8, 16。注意，不是降低 rrf_k，而是增大 rrf_k 才会让 rank 差异更平滑，使 RRF 更接近 weighted Borda。
3. 实用起点：rrf_k=4, video_weight=0.75 或 rrf_k=4, video_weight=1.0。
4. 目标不是 override 越高越好。建议先把 action_override 控制在 5% 到 25% 区间。如果仍为 0，说明视频没有介入；如果过高，说明视频项可能过强。
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
            f"{name}: 累计成功率", "episode", "cumulative success rate (%)",
            plot_dir / "success_cumulative_rate.png",
            note_for_metric("cum_rate", log_df["cum_rate"], name),
        )
        if not bins_df.empty:
            xlabels = [f"{int(a)}-{int(b)}" for a, b in zip(bins_df["episode_start"], bins_df["episode_end"])]
            plot_bar(
                xlabels, bins_df["success_rate"] * 100,
                f"{name}: 每 {SUCCESS_BIN_SIZE} 个 episode 的局部成功率", "episode bin", "success rate (%)",
                plot_dir / "success_bins.png",
                note_for_metric("success_rate_pct", bins_df["success_rate"] * 100, name),
            )
        plot_bar(
            log_df["episode"], log_df["success"],
            f"{name}: 成功失败序列", "episode", "success",
            plot_dir / "success_sequence.png",
            "每个柱子代表一个 episode 是否成功。它可用于观察是否存在连续难 seed 或连续易 seed。",
        )

    # Core CSV plots.
    if not ep_df.empty:
        plot_bar(
            ep_df["episode"], ep_df["tts_calls"],
            f"{name}: 每个 episode 的 TTS 调用数", "episode", "TTS calls",
            plot_dir / "tts_calls_per_episode.png",
            note_for_metric("tts_calls", ep_df["tts_calls"], name),
        )
        for metric in [
            "selected_action_best", "action_override", "selected_video_best",
            "rank_agree", "rank_spearman", "distance_ratio_video_over_action",
            "action_margin", "video_margin", "fused_margin", "fused_tie_count",
        ]:
            if metric in ep_df.columns:
                plot_line(
                    ep_df["episode"], ep_df[metric],
                    f"{name}: {metric} 每 episode 均值", "episode", metric,
                    plot_dir / f"{metric}_per_episode.png",
                    note_for_metric(metric, ep_df[metric], name),
                )

        for metric in [
            "rank_spearman", "distance_ratio_video_over_action",
            "action_margin", "video_margin", "fused_margin",
        ]:
            plot_box_by_success(ep_df, metric, plot_dir / f"{metric}_success_vs_fail.png")

    # Row-level histograms.
    for col in [
        "selected_action_rank", "selected_video_rank",
        "rank_spearman", "distance_ratio_video_over_action",
        "action_margin", "video_margin", "fused_margin",
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if pd.api.types.is_numeric_dtype(values):
                plot_hist(
                    values, bins=HIST_BINS,
                    title=f"{name}: {col} 分布", xlabel=col, ylabel="count",
                    path=plot_dir / f"{col}_hist.png",
                    note=note_for_metric(col, values, name),
                )

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
        if "fused_tie_count" in df.columns:
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
        plot_multi_line(
            log_frames, "episode", "cum_rate",
            "Comparison: 累计成功率", "episode", "cumulative success rate (%)",
            compare_dir / "compare_cumulative_success.png",
            note_for_compare("cum_rate", log_frames),
        )

        bin_frames = {}
        for name, d in results.items():
            b = d["bins"]
            if not b.empty:
                b = b.copy()
                b["bin_mid"] = (b["episode_start"] + b["episode_end"]) / 2
                b["success_rate_pct"] = b["success_rate"] * 100
                bin_frames[name] = b
        if bin_frames:
            plot_multi_line(
                bin_frames, "bin_mid", "success_rate_pct",
                f"Comparison: 每 {SUCCESS_BIN_SIZE} 个 episode 的局部成功率", "episode bin midpoint", "success rate (%)",
                compare_dir / "compare_success_bins.png",
                note_for_compare("success_rate_pct", bin_frames),
            )

    ep_frames = {name: d["episode"] for name, d in results.items() if not d["episode"].empty}
    compare_metrics = [
        "selected_action_best", "action_override", "selected_video_best",
        "rank_agree", "rank_spearman", "distance_ratio_video_over_action",
        "action_margin", "video_margin", "fused_margin", "tts_calls",
    ]
    for metric in compare_metrics:
        frames = {name: frame for name, frame in ep_frames.items() if metric in frame.columns}
        if frames:
            plot_multi_line(
                frames, "episode", metric,
                f"Comparison: {metric}", "episode", metric,
                compare_dir / f"compare_{metric}.png",
                note_for_compare(metric, frames),
            )

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
    write_field_guide(out_root / "FIELD_VISUALIZATION_GUIDE_ZH.md")
    write_rrf_tuning_note(out_root / "RRF_TUNING_NOTE_ZH.md")

    if args.csv:
        runs = [{"name": args.name, "csv": args.csv, "log": args.log}]
    else:
        runs = RUNS

    results = {}
    for run in runs:
        results[run["name"]] = analyze_run(run, out_root)

    if len(results) > 1:
        make_compare_outputs(results, out_root)

    print(f"Wrote annotated report to: {out_root}")
    for run in runs:
        print(f"  - {run['name']}: {out_root / run['name'] / 'report.txt'}")


if __name__ == "__main__":
    main()
