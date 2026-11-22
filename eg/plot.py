#!/usr/bin/env python3
"""
egplot: chart a forest.py report. One line (small markers) per
treatment; x = data sets, split into forest.py's DATA groups, each
sorted by lgbm; y = 0..100, labelled every 10, dashed every 5.
Usage: python3 eg/forest.py > out.txt; python3 eg/plot.py out.txt
(writes out.png).
(c) 2026, Tim Menzies <timm@ieee.org>, MIT license
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forest import DATA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read(file: str):
  "Treatment names; {dataset: {treatment: mean}} from a report"
  ts, d = None, {}
  for line in open(file):
    cells = [c.strip(" !\n") for c in line.replace("!", "!,").split(",")]
    cells = [c for c in cells if c]
    if not cells or line.startswith("#") or cells[-1][0] == "#": continue
    if "#x" in cells: ts = cells[:cells.index("#x")]; continue
    if ts is None: continue                   # preamble
    d[cells[-1]] = dict(zip(ts, map(float, cells)))
  return ts, d

def main(file: str) -> None:
  ts, d = read(file)
  key = "lgbm" if "lgbm" in ts else ts[-1]
  groups = [(g, sorted((f for f in fs.split() if f in d),
                       key=lambda f: d[f][key]))
            for g, fs in DATA.items()]
  xs = [f for _, fs in groups for f in fs]
  fig, ax = plt.subplots(figsize=(max(8, len(xs) * .3), 5))
  for t, m in zip(ts, "osD^v"):
    ax.plot(range(len(xs)), [d[f][t] for f in xs], marker=m,
            markersize=3, linewidth=1, label=t)
  i = 0
  for g, fs in groups:
    ax.axvline(i - .5, color="#999", linewidth=.6)
    ax.text(i + len(fs)/2 - .5, 2, g, ha="center", fontsize=9)
    i += len(fs)
  ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 10))
  for y in range(5, 100, 10):
    ax.axhline(y, color="#ccc", linewidth=.5, linestyle="--")
  ax.grid(axis="y", color="#ddd", linewidth=.5)
  ax.set_xticks(range(len(xs)), xs, rotation=90, fontsize=7)
  ax.set_ylabel("auc"); ax.legend(loc="lower right", fontsize=8)
  ax.set_title("sorted by %s within each group" % key, fontsize=9)
  fig.tight_layout()
  fig.savefig(out := os.path.splitext(file)[0] + ".png", dpi=150)
  print(out)

if __name__ == "__main__": main(sys.argv[1])
