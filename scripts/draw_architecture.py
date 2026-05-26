#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render the 9-step pipeline architecture diagram to assets/architecture.png.

Run:
    python scripts/draw_architecture.py
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "assets" / "architecture.png"

# Color palette (Material-design-ish; clean and projector-readable)
NORMAL_FILL = "#E8F0FE"
NORMAL_EDGE = "#1A73E8"
CORE_FILL   = "#FCE8E6"   # Nemotron core (red highlight)
CORE_EDGE   = "#D33B27"
DUAL_FILL   = "#FEF7E0"   # imaging dual path (amber)
DUAL_EDGE   = "#F9AB00"
LOOP_FILL   = "#E6F4EA"   # autonomous loop (green)
LOOP_EDGE   = "#1E8E3E"
GUARD_FILL  = "#F3E8FD"   # guardrails (purple)
GUARD_EDGE  = "#7E57C2"


def box(ax, x, y, w, h, text, fill, edge, fontsize=10, fontweight="normal"):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=fill, edgecolor=edge, linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


def arrow(ax, x1, y1, x2, y2, color="#222", lw=1.4, ls="-"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls),
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 12), dpi=160)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title ----
    ax.text(8, 15.55, "Pediatric Neuro-Oncology Surgical Planning Agent",
            ha="center", va="center", fontsize=17, fontweight="bold")
    ax.text(8, 15.05,
            "Autonomous long-agent pipeline   ·   core reasoning: NVIDIA Nemotron 3 Super",
            ha="center", va="center", fontsize=11, style="italic", color="#555")

    # ---- 9-step central pipeline ----
    center_x = 5.0
    step_w = 8.6

    steps = [
        # (y, h, fill, edge, text, fontsize, weight)
        (14.30, 0.78, NORMAL_FILL, NORMAL_EDGE,
         "1. De-identification   (PHI regex scrub)", 11, "bold"),
        (13.00, 1.40, DUAL_FILL, DUAL_EDGE,
         "2. Imaging   (optional, dual path)\n"
         "JPG / PNG   →   Nemotron Nano Omni VLM\n"
         "DICOM / NIfTI   →   SAFE_MODE  (preprocessing only, default ON)",
         10, "bold"),
        (11.55, 0.70, NORMAL_FILL, NORMAL_EDGE,
         "3. Completeness Check", 11, "bold"),
        (10.50, 0.90, NORMAL_FILL, NORMAL_EDGE,
         "4. RAG Retrieval   (TF-IDF + auto-picks refreshed PubMed / CT.gov evidence)",
         10.5, "bold"),
        (9.00, 1.30, CORE_FILL, CORE_EDGE,
         "5. NEMOTRON 3 SUPER    ←    CORE REASONING ENGINE\n"
         "(NVIDIA hosted;  deterministic MOCK fallback when no API key)",
         12, "bold"),
        (7.50, 0.75, NORMAL_FILL, NORMAL_EDGE,
         "6. Clinical Trial Matching   (age + tumor type + molecular markers)",
         10.5, "bold"),
        (6.10, 1.30, NORMAL_FILL, NORMAL_EDGE,
         "7. Drug Sensitivity Ranking   (ML model, GDSC2 cell lines)\n"
         "KNS-42 OFF-DIST surrogate for DMG\n"
         "D-263MG excluded (Cellosaurus possibly-misidentified flag)",
         10, "bold"),
        (4.55, 0.90, GUARD_FILL, GUARD_EDGE,
         "8. Guardrails   (7 policies, incl. Policy 7: drug-section preclinical disclaimers)",
         10.5, "bold"),
        (3.35, 0.75, NORMAL_FILL, NORMAL_EDGE,
         "9. MDT Report Generation   (Markdown)", 11, "bold"),
    ]

    for y, h, fill, edge, text, fs, fw in steps:
        box(ax, center_x, y, step_w, h, text, fill, edge, fontsize=fs, fontweight=fw)

    # Downward arrows between boxes
    for i in range(len(steps) - 1):
        y_from = steps[i][0] - steps[i][1] / 2
        y_to   = steps[i + 1][0] + steps[i + 1][1] / 2
        arrow(ax, center_x, y_from - 0.02, center_x, y_to + 0.02)

    # ---- Right-side: Autonomous Refresh Loop ----
    loop_x, loop_y, loop_w, loop_h = 13.1, 11.5, 5.4, 5.0
    box(ax, loop_x, loop_y, loop_w, loop_h,
        "Autonomous Refresh Loop\n"
        "tools/autonomous_refresh_loop.py\n"
        "\n"
        "Every N hours  (default 6h):\n"
        "  •  PubMed E-utilities\n"
        "  •  ClinicalTrials.gov v2 API\n"
        "      →  writes  rag_sources/*.jsonl\n"
        "      →  triggers  watcher.py\n"
        "      →  triggers  run_demo.py\n"
        "\n"
        "Timeouts: 20s  ·  --offline mode\n"
        "for demo resilience",
        LOOP_FILL, LOOP_EDGE, fontsize=9)

    # Dashed arrow from loop sidebar into step 4 (RAG)
    arrow(ax,
          loop_x - loop_w / 2,         loop_y - loop_h / 2 + 0.30,
          center_x + step_w / 2 + 0.10, 10.50,
          color=LOOP_EDGE, lw=1.6, ls="--")
    # Label placed in the gap between step-4 right edge (x=9.3) and sidebar
    # left edge (x=10.4), so it doesn't overlap either.
    ax.text(9.85, 9.55, "feeds new\nevidence",
            ha="center", va="top",
            fontsize=9, style="italic", color=LOOP_EDGE)

    # ---- Bottom safety strip ----
    safety_text = (
        "DECISION SUPPORT, NOT A DECISION   ·   All outputs require radiology / "
        "neurosurgery / MDT verification   ·   Research prototype only"
    )
    ax.text(8, 1.05, safety_text, ha="center", va="center",
            fontsize=10, style="italic", color="#B00020",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#FFF3F3", edgecolor="#B00020", linewidth=1.2))

    # ---- Legend ----
    legend_y = 2.20
    legend_items = [
        (CORE_FILL, CORE_EDGE, "core LLM"),
        (DUAL_FILL, DUAL_EDGE, "dual-path input"),
        (GUARD_FILL, GUARD_EDGE, "guardrails"),
        (LOOP_FILL, LOOP_EDGE, "autonomous loop"),
    ]
    lx = 1.0
    for fill, edge, label in legend_items:
        box(ax, lx + 0.35, legend_y, 0.55, 0.32, "", fill, edge)
        ax.text(lx + 0.78, legend_y, label, ha="left", va="center", fontsize=9.5)
        lx += 3.2

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(OUTPUT_PATH), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Saved {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
