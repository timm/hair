#!/usr/bin/env python3
"""
egforest: a fast random forest of class trees, vs one tree, vs
lightgbm (if installed). Each tree: a random half of train (at most
200 rows per class), deep (leaf=2, maxd=30), cut on rows binned
once into 8 equal-frequency bins, sqrt(#x) random cols per node.
Predict: mean of the leaves' class shares; score: aucs. Data: the
39 $MOOT/classify sets in DATA, grouped num, sym, mix, big (mixed,
many rows). Not worth adding (tried in lisp): MDL bins, 16 bins,
Laplace votes, best-of-3^d fast-frugal subtrees.
(c) 2026, Tim Menzies <timm@ieee.org>, MIT license
"""
import os, sys  # make hair importable whether run from source or installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hair import *

DATA = dict(
  num = """balance.scale blood-transfusion-service-center breast.w
           diabetes ecoli glass ionosphere sonar vehicle wdbc""",
  sym = """vote soybean tic-tac-toe german car solar.flare2
           primary.tumor breast.cancer spect.test audiology""",
  mix = """credit.a eucalyptus ilpd heart.c hepatitis colic autos
           cylinder.bands credit.g cmc""",
  big = """hypothyroid sick Internet-Advertisements churn COMPAS53
           nomao bank-marketing electricity adult""")

def capped(rows: list, Y, n: int) -> list:
  "At most n random rows per class"
  seen = {}
  def ok(r): seen[Y(r)] = seen.get(Y(r), 0) + 1; return seen[Y(r)] <= n
  return [r for r in random.sample(rows, len(rows)) if ok(r)]

def forest(tbl, rows, Y, n=10, cap=200, b4=True, rf=True, leaf=2,
           maxd=30, q=8) -> list:
  "n class trees, each on a random half of rows (n=1: all of them)"
  rows = capped(rows, Y, cap) if cap else rows
  cut  = binned(tbl, rows, Y, q, leaf, rf) if b4 else None
  return [tree(tbl, rows if n == 1 else random.sample(rows, len(rows)//2),
               Y, Sym, leaf, maxd, cut=cut) for _ in range(n)]

def vote(tbl, trees: list, row: list) -> dict:
  "class -> probability: mean over trees of row's leaf's class share"
  out = {}
  for t in trees:
    ys = node(tbl, t, row).ys; n = sum(ys.values())
    for k, v in ys.items(): out[k] = out.get(k, 0) + v / n / len(trees)
  return out

def lgbm(tbl, train: list, test: list) -> list:
  "lightgbm's {class: prob} per test row (defaults; syms categorical)"
  import numpy as np, lightgbm as lgb
  syms = {a: {v: i for i, v in enumerate(sorted(tbl.cols[a], key=str))}
          for a in tbl.x if isa(tbl.cols[a], Sym)}
  X = lambda rs: np.array([[np.nan if r[a] == "?" else
                            syms[a].get(r[a], np.nan) if a in syms else
                            float(r[a]) for a in tbl.x] for r in rs])
  ks = {str(k): k for k in tbl.cols[tbl.klass]}
  m  = lgb.LGBMClassifier(verbose=-1, n_jobs=1)
  m.fit(X(train), [str(r[tbl.klass]) for r in train],
        categorical_feature=[i for i, a in enumerate(tbl.x) if a in syms])
  return [{ks[c]: p for c, p in zip(m.classes_, ps)}
          for ps in m.predict_proba(X(test))]

TREATMENTS = {"1tree":  dict(n=1, b4=False, rf=False),
              "forest": dict()}

def learner(tbl, train, test):
  "Per treatment: (name, want, {class: prob}) per test row"
  Y = lambda r: r[tbl.klass]
  train = [r for r in train if Y(r) != "?"]
  test  = [r for r in test  if Y(r) != "?"]
  for name, kw in TREATMENTS.items():
    ts = forest(tbl, train, Y, **kw)
    for r in test: yield name, Y(r), vote(tbl, ts, r)
  try:
    for r, p in zip(test, lgbm(tbl, train, test)): yield "lgbm", Y(r), p
  except ImportError: pass

def main() -> None:
  the.pool = the.test = 10**9        # no row caps: forest caps per class
  cli(the)
  fs = ["$MOOT/classify/%s.csv" % f for g in DATA.values() for f in g.split()]
  print(conf(the, "forest.py", eval="auc", xval="5x2"))
  print("# auc (weighted, 0..100), bigger=better; ! marks rank-1; "
        "last col = data set\n")
  report(learn(fs, learner, aucs, n=2), reverse=True, decs=1)

if __name__ == "__main__": main()
