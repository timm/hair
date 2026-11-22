"""
tree: regression or classification trees over a Tab.
(c) 2026, Tim Menzies <timm@ieee.org>, MIT license
Trimmed from ezr-py/xai.py.

`bins` yields candidate (cost, at, v) splits for one column;
cost is the size-weighted spread of the two halves. `tree`
recurses the min-cost cut (default: `bin`; or `binned`, faster
for class trees); nodes keep their rows, their y summary `ys`
and its `mid`. `leaf` routes a row down; `leaves` yields them
all; `walk` lists the nodes; `show` prints them.
"""
import random
from bisect import bisect_left
from math import log2
from .lib import o, isa
from .data import Sym, Num, add, adds, fold, mid
from .stats import xpect

def bins(tbl: o, rows: list, at: int, Y, accum, leaf=3):
  "Yield (cost, at, v) bins; both sides hold at least leaf rows"
  xy  = [(r[at], Y(r)) for r in rows if r[at] != "?"]
  n   = len(xy)
  tot = adds((y for _,y in xy), accum())
  if isa(tbl.cols[at], Sym):
    for k in {x for x,_ in xy}:
      ys = [y for x,y in xy if x==k]
      if leaf <= len(ys) <= n - leaf:
        here = adds(ys, accum())
        yield xpect(here, fold(tot, here, -1)), at, k
  else:
    xy.sort(); here = accum()
    for j,(x,y) in enumerate(xy):
      here = add(here, y)
      if leaf <= j+1 <= n - leaf and x != xy[j+1][0]:
        yield xpect(here, fold(tot, here, -1)), at, x

def bin(tbl: o, rows: list, Y, accum, leaf=3):
  "The min-cost bin over every x column (None if no legal split)"
  return min((b for at in tbl.x
              for b in bins(tbl, rows, at, Y, accum, leaf)),
             default=None, key=lambda b: b[:2])  # v may mix int,str

def has(w, col, v) -> bool:
  "Does one cell fall on the yes-side of a bin? (? = yes)"
  return w == "?" or (v == w if isa(col, Sym) else w <= v)

def tree(tbl: o, rows: list, Y, accum=Num, leaf=3, maxd=8, lvl=0,
         cut=None) -> o:
  """Recursively split rows on the min-cost cut; accum=Num|Sym.
  cut(rows) -> (cost, at, v) or None; default: bin"""
  cut = cut or (lambda rs: bin(tbl, rs, Y, accum, leaf))
  all = adds((Y(r) for r in rows), accum())
  t = o(at=None, v=None, ys=all, mid=mid(all), n=len(rows), rows=rows,
        yes=None, no=None)
  if len(rows) >= 2*leaf and lvl < maxd:
    if b := cut(rows):
      _, at, v = b
      yes, no = [], []
      for r in rows:
        (yes if has(r[at], tbl.cols[at], v) else no).append(r)
      if yes and no:
        t.at, t.v = at, v
        t.yes = tree(tbl, yes, Y, accum, leaf, maxd, lvl+1, cut)
        t.no  = tree(tbl, no,  Y, accum, leaf, maxd, lvl+1, cut)
  return t

def node(tbl: o, t: o, row: list) -> o:
  "Walk a row down to its leaf node"
  while t.at is not None:
    t = t.yes if has(row[t.at], tbl.cols[t.at], t.v) else t.no
  return t

def leaf(tbl: o, t: o, row: list):
  "The mid of a row's leaf"
  return node(tbl, t, row).mid

def leaves(t: o):
  if t.at is None: yield t
  else: yield from leaves(t.yes); yield from leaves(t.no)

def walk(tbl: o, t: o, lvl=0, test=None):
  "Yield (lvl, test, mid, n) per node, parents first; test = the cut here"
  yield lvl, test, t.mid, t.n
  if t.at is not None:
    sym, at = isa(tbl.cols[t.at], Sym), tbl.names[t.at]
    yield from walk(tbl, t.yes, lvl+1, f"{at} {'==' if sym else '<='} {t.v}")
    yield from walk(tbl, t.no,  lvl+1, f"{at} {'!=' if sym else '>'} {t.v}")

def show(tbl: o, t: o) -> None:
  "One line per node: mid, n, then the cut that leads here"
  for lvl, test, m, n in walk(tbl, t):
    print("%6s %5s  %s%s" % (m if isa(m, str) else round(m, 2), n,
                            "|  " * lvl, test or ""))

#-- binned: class trees on rows coded once into bins --------------
def ent(cs: list) -> float:
  "Entropy of a list of counts"
  n = sum(cs)
  return -sum(c/n * log2(c/n) for c in cs if c)

def quantiles(rows: list, at: int, q: int) -> list:
  "Up to q-1 equal-frequency edges of column at"
  xs = sorted(r[at] for r in rows if r[at] != "?")
  return sorted({xs[j*len(xs)//q] for j in range(1, q)}) if xs else []

def binned(tbl: o, rows: list, Y, q=8, leaf=3, rf=False):
  """A cut for class trees. Once: code each row as ints (per x, its
  bin among q quantiles, or its sym value; -1 = ?), then its class.
  Per call: count classes per bin, sweep, min entropy (x <= edge,
  or sym == v). rf: just sqrt(#x) random cols per call"""
  ks   = {k: i for i, k in enumerate({Y(r) for r in rows})}
  vals = {at: (sorted({r[at] for r in rows if r[at] != "?"}, key=str)
               if isa(tbl.cols[at], Sym) else quantiles(rows, at, q))
          for at in tbl.x}
  def idx(at, x):
    if x == "?": return -1
    if isa(tbl.cols[at], Sym):
      return vals[at].index(x) if x in vals[at] else -1
    return bisect_left(vals[at], x)
  code = {id(r): [idx(at, r[at]) for at in tbl.x] + [ks[Y(r)]] for r in rows}
  m = max(1, round(len(tbl.x) ** .5))
  def cut(rs):
    best, cs = None, [code[id(r)] for r in rs]
    for j in (sorted(random.sample(range(len(tbl.x)), m)) if rf
              else range(len(tbl.x))):
      at, sym = tbl.x[j], isa(tbl.cols[tbl.x[j]], Sym)
      n, all = [[0]*len(ks) for _ in range(len(vals[at]) + 1)], [0]*len(ks)
      for c in cs:
        if c[j] >= 0: n[c[j]][c[-1]] += 1; all[c[-1]] += 1
      yes = [0]*len(ks)
      for b, v in enumerate(vals[at]):           # bins: add up; syms: just b
        yes = n[b] if sym else [y + k for y, k in zip(yes, n[b])]
        no  = [a - y for a, y in zip(all, yes)]
        ny, nn = sum(yes), sum(no)
        if min(ny, nn) >= leaf:
          e = (ny*ent(yes) + nn*ent(no)) / (ny + nn)
          if best is None or e < best[0]: best = (e, at, v)
    return best
  return cut
