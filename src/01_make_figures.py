"""
01 号 · 报告用图（2.1.2 格点图、2.1.5 接缝剖面）

两图都从仓库内既有产物重算，不手工描点：
  figures/fig_domain_map.png  ← src/config.py 的 26 节点注册表 + data/geodata 陆地边界
  figures/fig_run_seam.png    ← data/haze_panel_clean.parquet + models/seam_report.json

用法：
    python src/01_make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

LAND = os.path.join(C.DATA_DIR, "geodata", "land_50m_kl.geojson")
PANEL = os.path.join(C.DATA_DIR, "haze_panel_clean.parquet")
SEAM = os.path.join(C.MODEL_DIR, "seam_report.json")

STYLE = {
    "receptor": dict(marker="s", s=95, c="#c0392b", edgecolor="white", linewidth=1.1, zorder=6),
    "corridor": dict(marker="o", s=64, c="#e08a1e", edgecolor="white", linewidth=0.9, zorder=5),
    "source": dict(marker="^", s=62, c="#2f6f8f", edgecolor="white", linewidth=0.9, zorder=4),
}
LABEL = {
    "KL_City": (6, 6), "Port_Klang": (-46, -14), "Sepang": (8, -12),
    "Penang": (-40, 6), "Johor_Bahru": (7, -10), "Kuantan": (7, 2),
    "Ipoh": (-30, 5), "Kota_Bharu": (7, 2),
    "Medan": (-42, 3), "Palembang": (7, -10), "Pontianak": (7, 0),
    "Palangkaraya": (7, 0), "Banjarmasin": (7, -8), "Jambi": (7, -2),
}


def _land_polys() -> list[np.ndarray]:
    with open(LAND, encoding="utf-8") as f:
        gj = json.load(f)
    out = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0], dtype=float)
            if ring[:, 0].max() < 94 or ring[:, 0].min() > 116:
                continue
            if ring[:, 1].max() < -5 or ring[:, 1].min() > 8:
                continue
            out.append(ring)
    return out


def fig_domain() -> None:
    polys = _land_polys()
    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=300)
    ax.add_collection(PolyCollection(polys, facecolor="#e6e3dc",
                                     edgecolor="#a8a29a", linewidth=0.35, zorder=1))

    for node in C.NODES:
        ax.scatter(node["lon"], node["lat"], **STYLE[node["role"]])
        if node["id"] in LABEL and node["role"] != "receptor":
            dx, dy = LABEL[node["id"]]
            ax.annotate(node["id"].replace("_", " "), (node["lon"], node["lat"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=6.6,
                        color="#333333", zorder=7)

    # Klang Valley: 8 candidate sites collapse onto 3 grid cells
    ax.add_patch(Rectangle((101.18, 2.62), 0.92, 0.72, fill=False, ec="#c0392b",
                           lw=0.9, ls="--", zorder=7))
    ax.annotate("see inset", xy=(102.16, 2.72), fontsize=6.8, color="#c0392b",
                va="center", ha="left", zorder=7)

    # south-west monsoon, the transport pathway the project is about
    ax.annotate("", xy=(101.45, 2.55), xytext=(99.55, -1.35),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#4a4a4a",
                                mutation_scale=13, connectionstyle="arc3,rad=-0.16"), zorder=8)
    ax.text(99.12, -0.72, "SW monsoon\n(Jun–Oct)", fontsize=7.2, color="#4a4a4a",
            rotation=56, ha="center", va="center", zorder=8)

    handles = [
        plt.Line2D([], [], linestyle="", marker=STYLE[k]["marker"], markersize=6,
                   markerfacecolor=STYLE[k]["c"], markeredgecolor="white",
                   label=f"{name} ({n})")
        for k, name, n in (("receptor", "Receptor", 3),
                           ("corridor", "Corridor", 10),
                           ("source", "Source", 13))
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.6, frameon=True,
              framealpha=0.92, borderpad=0.5)

    ax.set_xlim(94.6, 115.0)
    ax.set_ylim(-4.4, 7.2)
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.set_title("Study domain: 3 receptors, 10 corridor nodes and 13 upwind source cells",
                 fontsize=10.5)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", adjustable="box")

    # inset: the three receptor cells, which are 45 km apart at best
    ins = ax.inset_axes([0.635, 0.585, 0.235, 0.300])
    ins.set_facecolor("#f7f7f5")
    ins.add_collection(PolyCollection(polys, facecolor="#e6e3dc",
                                      edgecolor="#a8a29a", linewidth=0.3, zorder=1))
    ins.set_xlim(101.15, 102.12)
    ins.set_ylim(2.52, 3.42)
    for node in C.NODES:
        if node["role"] != "receptor":
            continue
        ins.scatter(node["lon"], node["lat"], **STYLE[node["role"]])
        dx, dy = {"KL_City": (-4, 7), "Port_Klang": (6, -7), "Sepang": (6, 3)}[node["id"]]
        ins.annotate(node["id"].replace("_", " "), (node["lon"], node["lat"]),
                     textcoords="offset points", xytext=(dx, dy), fontsize=5.8,
                     color="#333333", zorder=7,
                     ha="right" if dx < 0 else "left")
    ins.set_xticks([])
    ins.set_yticks([])
    # scale bar: 1 deg longitude at 3 N is 110.8 km, so 20 km is 0.18 deg
    ins.plot([101.24, 101.42], [2.585, 2.585], color="#333333", lw=1.2, zorder=8)
    ins.annotate("20 km", (101.33, 2.605), fontsize=5.2, ha="center", color="#333333")
    ins.set_title("Klang Valley: 8 candidate sites → 3 cells", fontsize=6.0, pad=2.5)
    for sp in ins.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#c0392b")

    fig.tight_layout()
    out = os.path.join(C.FIGURE_DIR, "fig_domain_map.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


def fig_seam() -> None:
    cols = ["station_id", "timestamp", "pm2_5"]
    df = pd.read_parquet(PANEL, columns=cols).sort_values(["station_id", "timestamp"])
    df["d1"] = df.groupby("station_id")["pm2_5"].diff()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    panel = df.dropna(subset=["d1"]).groupby("hour")["d1"].apply(
        lambda s: s.abs().mean())
    kl = df[(df.station_id == "KL_City")].dropna(subset=["d1"]).groupby("hour")["d1"].apply(
        lambda s: s.abs().mean())
    kl_signed = df[(df.station_id == "KL_City")].dropna(subset=["d1"]).groupby("hour")["d1"].mean()

    with open(SEAM, encoding="utf-8") as f:
        rep = json.load(f)
    seam = rep["seam_hours_local"]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), dpi=300, sharex=True)

    ax = axes[0]
    for h in seam:
        ax.axvspan(h - 0.5, h + 0.5, color="#c0392b", alpha=0.10, lw=0, zorder=0)
    ax.plot(panel.index, panel.values, "-o", ms=3.4, lw=1.3, color="#2f6f8f",
            label="26-node mean")
    ax.plot(kl.index, kl.values, "-s", ms=3.2, lw=1.1, color="#c0392b",
            label="KL_City")
    ax.set_ylabel("Mean |ΔPM$_{2.5}$| (µg/m³ h$^{-1}$)", fontsize=9)
    ax.set_xlabel("Hour of day (local, UTC+8)", fontsize=9)
    ax.set_title("(a) Hour-to-hour change by hour", fontsize=9.6)
    ax.legend(fontsize=7.6, frameon=False)
    ax.annotate(f"{rep['panel']['seam_ratio']:.2f}×", (20, panel.loc[20]),
                textcoords="offset points", xytext=(-26, 4), fontsize=8, color="#2f6f8f")
    ax.annotate(f"{rep['reference_station']['seam_ratio']:.2f}×", (20, kl.loc[20]),
                textcoords="offset points", xytext=(-22, 6), fontsize=8, color="#c0392b")

    ax = axes[1]
    for h in seam:
        ax.axvspan(h - 0.5, h + 0.5, color="#c0392b", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color="#666666", lw=0.7)
    ax.bar(kl_signed.index, kl_signed.values, width=0.72, color="#7f8c8d")
    ax.bar([20], [kl_signed.loc[20]], width=0.72, color="#c0392b")
    ax.set_title("(b) KL_City signed change", fontsize=9.6)
    ax.set_xlabel("Hour of day (local, UTC+8)", fontsize=9)
    ax.set_ylabel("Mean ΔPM$_{2.5}$ (µg/m³ h$^{-1}$)", fontsize=9)
    ax.annotate(f"+{kl_signed.loc[20]:.2f}", (20, kl_signed.loc[20]),
                textcoords="offset points", xytext=(-10, 5), fontsize=8, color="#c0392b")

    for ax in axes:
        ax.set_xticks(range(0, 24, 3))
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    out = os.path.join(C.FIGURE_DIR, "fig_run_seam.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


if __name__ == "__main__":
    os.makedirs(C.FIGURE_DIR, exist_ok=True)
    fig_domain()
    fig_seam()
