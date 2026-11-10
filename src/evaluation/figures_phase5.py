"""Phase 5 figures for the paper and the slides: results/phase5/fig_*.png (+ .pdf).

    python -m src.evaluation.figures_phase5
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import metrics as M

R = Path("results/phase5")
SHORT = {"gpt-4-1106-preview": "GPT-4", "gpt-3.5-turbo-1106": "GPT-3.5", "claude-v1": "Claude 1",
         "claude-v2": "Claude 2", "claude-instant-v1": "Claude Instant", "meta/llama-2-70b-chat": "Llama-2 70B",
         "mistralai/mixtral-8x7b-chat": "Mixtral 8x7B", "zero-one-ai/Yi-34B-Chat": "Yi 34B",
         "WizardLM/WizardLM-13B-V1.2": "WizardLM 13B", "meta/code-llama-instruct-34b-chat": "CodeLlama 34B",
         "mistralai/mistral-7b-chat": "Mistral 7B"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(R / f"fig_{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", R / f"fig_{name}.png")


def plane(e1, pool, ax, title):
    s = e1[pool][0]  # seed 11 for the picture; tables use 5-seed means
    singles = np.array([[x["cost"], x["quality"]] for x in s["singles"]]) * [1000, 1]
    hull = M.upper_hull(singles / [1000, 1]) * [1000, 1]
    ax.plot(hull[:, 0], hull[:, 1], "--", color="0.55", lw=1, label="Zero router (mix of fixed models)")
    ax.scatter(singles[:, 0], singles[:, 1], s=14, color="0.35", zorder=3, label="Single models")
    for (c, q), x in zip(singles, s["singles"]):
        if SHORT[x["model"]] in ("GPT-4", "GPT-3.5", "Claude Instant", "Claude 2", "Mixtral 8x7B", "Llama-2 70B"):
            ax.annotate(SHORT[x["model"]], (c, q), fontsize=6, xytext=(3, -8), textcoords="offset points", color="0.3")
    r = np.array([[x["cost"] * 1000, x["quality"]] for x in s["router"]])
    rh = M.upper_hull(r / [1000, 1]) * [1000, 1]
    ax.plot(rh[:, 0], rh[:, 1], "-", color="#5b2a86", lw=1.6, label="Router (BERT + RF), hull")
    ax.scatter(r[:, 0], r[:, 1], s=10, color="#5b2a86", zorder=4)
    ax.scatter([s["oracle"]["cost"] * 1000], [s["oracle"]["quality"]], marker="*", s=60, color="#e07b00",
               zorder=5, label="Oracle")
    ax.scatter([s["random"]["cost"] * 1000], [s["random"]["quality"]], marker="x", s=24, color="#b00020",
               zorder=5, label="Random")
    ax.set_xlabel("Mean cost per query (×10⁻³ USD)")
    ax.set_ylabel("Mean quality")
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, singles[:, 0].max() * 1.08)
    ax.set_ylim(0.55 if pool == "p5" else 0.15, 0.95)


def fig_plane():
    e1 = json.loads((R / "e1_offline.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    plane(e1, "p5", axes[0], "(a) 5 proprietary models, 30/60/10")
    plane(e1, "p11", axes[1], "(b) 11 models, 80/20 (MixLLM setting)")
    axes[0].legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "plane")
    # one-column version for the two-column paper
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6))
    plane(e1, "p5", axes[0], "(a) 5 proprietary models, 30/60/10")
    plane(e1, "p11", axes[1], "(b) 11 models, 80/20 (MixLLM setting)")
    axes[0].legend(fontsize=5.5, loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "plane_col")


def fig_substitution():
    a = json.loads((R / "e345_attacks.json").read_text())["substitution"]
    rf = json.loads((R / "e2_rf_online.json").read_text()) if (R / "e2_rf_online.json").exists() else []
    w = 0.1
    ecr = np.array([r["ecr_sub_gpt4_share_by_1000"] for r in a if r["w_C"] == w])
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(ecr.shape[1]) + 0.5
    ax.axvspan(0, 0.25 * ecr.shape[1], color="0.93")
    ax.axvline(0.25 * ecr.shape[1], color="0.5", lw=0.8, ls=":")
    ax.text(0.25 * ecr.shape[1] + 0.3, 0.95, "GPT-4 silently\nreplaced by GPT-3.5", fontsize=6, va="top")
    off = np.mean([r["offline_sub"]["gpt4_share"] for r in a if r["w_C"] == w])
    ax.plot([0, ecr.shape[1]], [off, off], color="0.4", lw=1, ls="--", label="Offline router (frozen)")
    rs = np.array([r["gpt4_share_by_1000"] for r in rf if r["w_C"] == w and r["scenario"] == "substitution"])
    if len(rs):
        ax.plot(x[: rs.shape[1]], rs.mean(0), color="#2a7ab0", lw=1.3, label="Phase 4 online (forest retraining)")
    ax.plot(x, ecr.mean(0), color="#5b2a86", lw=1.6, label="Phase 5 evidence-corrected router")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Online stream (thousand queries)")
    ax.set_ylabel("Share routed to GPT-4")
    ax.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "substitution")


def fig_poison():
    p = json.loads((R / "e4_poison.json").read_text())
    w = 0.1
    rhos = sorted({r["rho"] for r in p})
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    clean = np.mean([r["clean"]["quality"] for r in p if r["w_C"] == w])
    for name, lab, col in (("none", "No defence", "#b00020"), ("audit_local", "5% audits, local store", "#e07b00"),
                           ("ledger_agreement", "Ledger, agreement rule", "#2a7ab0"),
                           ("ledger_failures", "Ledger, failure-count rule", "#5b2a86")):
        ax.plot(rhos, [np.mean([r[name]["quality"] for r in p if r["w_C"] == w and r["rho"] == rho]) for rho in rhos],
                "o-", ms=3, color=col, label=lab)
    ax.axhline(clean, color="0.4", ls="--", lw=0.8, label="No attack")
    ax.set_xlabel("Share of malicious reporters")
    ax.set_ylabel("Test quality")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    save(fig, "poison")


if __name__ == "__main__":
    for f in (fig_plane, fig_substitution, fig_poison):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
