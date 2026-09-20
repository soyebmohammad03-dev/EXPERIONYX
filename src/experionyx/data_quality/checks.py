"""The quality checks. Each is a pure function of a table (or two) and its normalized configuration.

Method notes (also in docs/data-quality.md):
- Status rule: FAIL only when a CONFIGURED rule is violated; WARNING when the phenomenon itself
  needs interpretation (duplicates, identifier-like columns, non-finite values, unseen categories,
  leakage indicators, order violations); otherwise PASS with observations recorded.
- Outliers: Tukey fences (q1 - k*IQR, q3 + k*IQR, k=1.5 by default) or the modified z-score
  0.6745*(x - median)/MAD (|z| > 3.5). They are observations of extremeness under the method, never
  evidence of error, and change a status only when `max_outlier_rate` is configured.
- Skewness g1 = m3/m2^1.5 and excess kurtosis m4/m2^2 - 3 use population moments.
- Leakage indicators: equality with the target, absolute Pearson correlation with a numeric target,
  and purity (share of rows whose feature value's majority target agrees, over values with at least
  two rows). Purity is only flagged when the feature has at least two values, the values cover at
  least half of the rows, and it beats the target's own majority share: a fact about these rows,
  not proof of leakage.
- Group comparisons reuse Phase 10 (`compare` on 0/1 indicators for rates, Wilson intervals,
  `adjust_pvalues` per family) and Phase 12 (`shift` for the target distribution)."""

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experionyx.data_quality.data import Table, cell_key, classify
from experionyx.data_quality.results import CheckResult, Status, viol, worst
from experionyx.data_quality.spec import CheckSpec, QualityConfig, QualitySpec, json_key
from experionyx.domain import to_jsonable
from experionyx.drift import measures as drift_measures
from experionyx.drift.spec import INDEX, DriftConfig, Role, TemporalWindow
from experionyx.evaluation.bootstrap import quantile
from experionyx.interactions.taxonomy import Pairing
from experionyx.stats import core as st


@dataclass(frozen=True)
class Ctx:
    spec: QualitySpec
    tables: Mapping[str | None, Table]
    tokens: frozenset[str]

    @property
    def cfg(self) -> QualityConfig:
        return self.spec.config

    @property
    def kinds(self) -> dict[str, str]:
        return {f.name: f.kind for f in self.spec.features}

    def names(self, cs: CheckSpec, t: Table, kinds: tuple[str, ...] | None = None) -> list[str]:
        """Explicit features of the check, else every declared feature, else (kind-free checks) every column."""
        chosen = list(cs.config.get("features") or [])
        base = chosen or (sorted(self.kinds) if self.kinds else list(t.columns))
        return [n for n in base if kinds is None or self.kinds.get(n) in kinds]

    def kind(self, name: str) -> str:
        return self.kinds.get(name, "CATEGORICAL")


class Acc:
    def __init__(self, examples: int) -> None:
        self.ex = examples
        self.fails: list[dict[str, Any]] = []
        self.warns: list[dict[str, Any]] = []

    def fail(self, rule: str, subject: str, ids: Sequence[object], **d: Any) -> None:
        self.fails.append(viol(rule, subject, ids, self.ex, severity="FAIL", **d))

    def warn(self, rule: str, subject: str, ids: Sequence[object], **d: Any) -> None:
        self.warns.append(viol(rule, subject, ids, self.ex, severity="WARNING", **d))

    @property
    def viols(self) -> tuple[dict[str, Any], ...]:
        return (*self.fails, *self.warns)

    @property
    def status(self) -> Status:
        return Status.FAIL if self.fails else Status.WARNING if self.warns else Status.PASS


def _res(
    cs: CheckSpec, scope: Mapping[str, Any], status: Status, reason: str | None = None, **kw: Any
) -> CheckResult:
    return CheckResult(cs.check_id, cs.type, dict(scope), status, reason, **kw)


def _thr(cs: CheckSpec, *keys: str) -> dict[str, Any]:
    return {k: to_jsonable(cs.config.get(k)) for k in keys if cs.config.get(k) is not None}


def _empty(cs: CheckSpec, scope: Mapping[str, Any], t: Table) -> CheckResult | None:
    if t.n == 0:
        return _res(
            cs, scope, Status.NOT_APPLICABLE, "the table has no rows", evidence={"n_rows": 0}
        )
    return None


def _wilson(k: int, n: int, ctx: Ctx) -> dict[str, Any]:
    d = to_jsonable(st.proportion_interval(k, n, ctx.cfg.confidence))
    assert isinstance(d, dict)  # noqa: S101
    return d


def _sorted_floats(vals: Sequence[float]) -> list[float]:
    return sorted(vals)


def outliers(vals: Sequence[float], method: str, k: float) -> tuple[list[bool], dict[str, Any]]:
    """Per value: is it extreme under the method; plus the method's parameters (documented above)."""
    s = _sorted_floats(vals)
    if method == "iqr":
        q1, q3 = quantile(s, 0.25), quantile(s, 0.75)
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        return [x < lo or x > hi for x in vals], {
            "method": "iqr",
            "k": k,
            "q1": q1,
            "q3": q3,
            "lower_fence": lo,
            "upper_fence": hi,
        }
    med = statistics.median(s)
    mad = statistics.median(abs(x - med) for x in s)
    if mad == 0:
        return [False] * len(vals), {
            "method": "mad",
            "k": k,
            "median": med,
            "mad": 0.0,
            "scoring_note": "MAD is zero: no value can be scored",
        }
    return [abs(0.6745 * (x - med) / mad) > k for x in vals], {
        "method": "mad",
        "k": k,
        "median": med,
        "mad": mad,
    }


def moments(vals: Sequence[float]) -> tuple[float | None, float | None]:
    """(skewness g1, excess kurtosis) with population moments; None when the variance is 0."""
    n = len(vals)
    m = math.fsum(vals) / n
    m2 = math.fsum((x - m) ** 2 for x in vals) / n
    if m2 == 0:
        return None, None
    return math.fsum((x - m) ** 3 for x in vals) / n / m2**1.5, math.fsum(
        (x - m) ** 4 for x in vals
    ) / n / m2**2 - 3


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = math.fsum(x) / n, math.fsum(y) / n
    sx, sy = math.fsum((a - mx) ** 2 for a in x), math.fsum((b - my) ** 2 for b in y)
    if sx == 0 or sy == 0:
        return None
    return math.fsum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / math.sqrt(sx * sy)


def _states(ctx: Ctx, t: Table, name: str, kind: str | None = None) -> list[tuple[str, Any]]:
    return classify(t.column(name), kind or ctx.kind(name), ctx.tokens)


# -- schema / sample count ---------------------------------------------------------------------------------------


def check_schema(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    missing = [n for n in cfg["required"] if n not in t.columns]
    if missing:
        a.fail("required_column_missing", ",".join(missing), missing, columns=missing)
    if not cfg["allow_extra"]:
        extra = [c for c in t.columns if c not in ctx.kinds and c not in cfg["required"]]
        if extra:
            a.fail("undeclared_column", ",".join(extra), extra, columns=extra)
    if t.ragged:
        a.fail("ragged_rows", "rows", t.ragged, expected_width=len(t.columns))
    per: dict[str, Any] = {}
    for name in ctx.names(cs, t):
        if name not in t.columns:
            if name not in cfg["required"]:
                a.fail("declared_feature_missing", name, [name])
            continue
        states = _states(ctx, t, name)
        bad = [i for i, (s, _) in zip(t.ids, states, strict=True) if s == "invalid"]
        per[name] = {"declared_type": ctx.kind(name), "n_invalid_for_type": len(bad)}
        if bad:
            a.fail("invalid_type", name, bad, declared_type=ctx.kind(name))
    obs = {
        "columns": list(t.columns),
        "n_columns": len(t.columns),
        "n_ragged_rows": len(t.ragged),
        "features": per,
    }
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "required", "allow_extra"),
        evidence={"n_rows": t.n},
    )


def check_sample_count(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    a, cfg = Acc(0), cs.config
    n = t.n
    if n == 0:
        a.warn("empty_table", "rows", [])
    for key, violated, what in (
        ("min", cfg["min"] is not None and n < cfg["min"], "fewer rows than min"),
        ("max", cfg["max"] is not None and n > cfg["max"], "more rows than max"),
        (
            "expected",
            cfg["expected"] is not None and n != cfg["expected"],
            "row count differs from expected",
        ),
    ):
        if violated:
            a.fail(f"{key}_violated", "rows", [], detail=what, configured=cfg[key], observed=n)
    if t.declared_n is not None and t.declared_n != n:
        a.fail("metadata_count_mismatch", "rows", [], declared=t.declared_n, observed=n)
    if ctx.spec.target is not None and t.target_len is not None and t.target_len != n:
        a.fail("feature_target_length_mismatch", "target", [], rows=n, targets=t.target_len)
    if ctx.spec.target is not None and t.target_len is None and n > 0:
        a.fail(
            "target_absent",
            "target",
            [],
            detail="a target is declared but the dataset serves none for this table",
        )
    if t.duplicate_ids:
        a.warn("duplicate_sample_ids", "ids", t.duplicate_ids)
    obs = {
        "n_rows": n,
        "declared_n": t.declared_n,
        "target_len": t.target_len,
        "n_unique_ids": len(set(t.ids)),
        "n_columns": len(t.columns),
    }
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "min", "max", "expected"),
        evidence={"n_rows": n},
    )


# -- missingness -------------------------------------------------------------------------------------------------


def check_missingness(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    a, cfg = Acc(int(ctx.cfg.max_examples)), cs.config
    names = ctx.names(cs, t)
    per: dict[str, Any] = {}
    row_missing = [set() for _ in range(t.n)]  # type: list[set[str]]
    for name in names:
        miss = [s == "missing" for s, _ in _states(ctx, t, name)]
        for i, m in enumerate(miss):
            if m:
                row_missing[i].add(name)
        k = sum(miss)
        per[name] = {
            "n_missing": k,
            "n_total": t.n,
            "rate": k / t.n,
            "interval": _wilson(k, t.n, ctx),
            "all_missing": k == t.n,
        }
        if cfg["max_feature_rate"] is not None and k / t.n > cfg["max_feature_rate"]:
            a.fail(
                "feature_missing_rate_exceeded",
                name,
                [],
                rate=k / t.n,
                configured=cfg["max_feature_rate"],
            )
    counts = [len(r) for r in row_missing]
    if cfg["max_row_rate"] is not None and names:
        bad = [
            i for i, c in zip(t.ids, counts, strict=True) if c / len(names) > cfg["max_row_rate"]
        ]
        if bad:
            a.fail("row_missing_rate_exceeded", "rows", bad, configured=cfg["max_row_rate"])
    pats = Counter(tuple(sorted(r)) for r in row_missing if r)
    top = [
        {"features": list(p), "rows": n}
        for p, n in sorted(pats.items(), key=lambda kv: (-kv[1], kv[0]))[: int(cfg["patterns"])]
    ]
    obs = {
        "features": per,
        "rows_with_any_missing": sum(c > 0 for c in counts),
        "rows_all_missing": sum(c == len(names) for c in counts) if names else 0,
        "missing_per_row_histogram": dict(sorted(Counter(counts).items())),
        "patterns": top,
        "n_patterns": len(pats),
        "note": "missingness can be informative (it may relate to the target, a slice or time); it is measured here, not judged",
    }
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "max_feature_rate", "max_row_rate"),
        evidence={"n_rows": t.n, "n_features": len(names)},
    )


# -- duplicates and identity -------------------------------------------------------------------------------------


def check_duplicates(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    names = ctx.names(cs, t)
    cols = [t.column(n) for n in names]
    tgt = t.target if t.target is not None else None
    fkeys = [tuple(cell_key(c[i]) for c in cols) for i in range(t.n)]
    keys = [
        (fk, cell_key(tgt[i])) if cfg["include_target"] and tgt is not None else fk
        for i, fk in enumerate(fkeys)
    ]
    groups: dict[Any, list[int]] = {}
    for i, k in zip(t.ids, keys, strict=True):
        groups.setdefault(k, []).append(i)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    dup_rows = sum(len(v) - 1 for v in dup_groups.values())
    conflicts: list[list[int]] = []
    if tgt is not None:
        by_vec: dict[Any, list[tuple[int, str]]] = {}
        for i, fk, tv in zip(t.ids, fkeys, tgt, strict=True):
            by_vec.setdefault(fk, []).append((i, cell_key(tv)))
        conflicts = [
            [i for i, _ in v] for v in by_vec.values() if len(v) > 1 and len({k for _, k in v}) > 1
        ]
    conf_rows = sum(len(g) for g in conflicts)
    rate, crate = dup_rows / t.n, conf_rows / t.n
    if cfg["max_duplicate_rate"] is not None and rate > cfg["max_duplicate_rate"]:
        a.fail(
            "duplicate_rate_exceeded", "rows", [], rate=rate, configured=cfg["max_duplicate_rate"]
        )
    if cfg["max_conflicting_rate"] is not None and crate > cfg["max_conflicting_rate"]:
        a.fail(
            "conflicting_rate_exceeded",
            "rows",
            [],
            rate=crate,
            configured=cfg["max_conflicting_rate"],
        )
    if dup_rows and not a.fails:
        a.warn(
            "duplicate_rows",
            "rows",
            [i for v in dup_groups.values() for i in v[1:]],
            n_groups=len(dup_groups),
        )
    if conflicts and not any(f["rule"] == "conflicting_rate_exceeded" for f in a.fails):
        a.warn(
            "identical_features_different_target",
            "rows",
            [i for g in conflicts for i in g],
            n_groups=len(conflicts),
        )
    ex = int(cfg["examples"])
    obs = {
        "columns_compared": names,
        "include_target": cfg["include_target"],
        "n_rows": t.n,
        "n_unique_rows": len(groups),
        "n_duplicate_rows": dup_rows,
        "n_duplicate_groups": len(dup_groups),
        "duplicate_rate": rate,
        "example_groups": [v[:10] for v in list(dup_groups.values())[:ex]],
        "conflicting_groups": len(conflicts),
        "conflicting_rows": conf_rows,
        "conflicting_rate": crate if tgt is not None else None,
        "conflicts_available": tgt is not None,
        "note": "a repeated row is reported, not judged: it can be a legitimate repeated measurement, a join artifact or a copy",
    }
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "max_duplicate_rate", "max_conflicting_rate"),
        evidence={"n_rows": t.n},
    )


def check_identifiers(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    obs: dict[str, Any] = {}
    idc = ctx.spec.id_column
    if idc is not None:
        vals = t.column(idc)
        st_ = classify(vals, "CATEGORICAL", ctx.tokens)
        groups: dict[str, list[int]] = {}
        for i, (s, v) in zip(t.ids, st_, strict=True):
            if s == "valid":
                groups.setdefault(v, []).append(i)
        dup = {k: v for k, v in groups.items() if len(v) > 1}
        pos = {i: p for p, i in enumerate(t.ids)}
        others = [c for c in t.columns if c != idc]
        conflicting = 0
        conf_ids: list[int] = []
        for members in dup.values():
            sigs = {
                tuple(cell_key(t.rows[pos[i]][t.index(c)]) for c in others)
                + ((cell_key(t.target[pos[i]]),) if t.target is not None else ())
                for i in members
            }
            if len(sigs) > 1:
                conflicting += 1
                conf_ids += members
        n_missing = sum(s == "missing" for s, _ in st_)
        obs["id_column"] = {
            "name": idc,
            "n_missing": n_missing,
            "n_distinct": len(groups),
            "n_duplicated_values": len(dup),
            "n_rows_with_duplicated_id": sum(len(v) for v in dup.values()),
            "n_conflicting_groups": conflicting,
        }
        (a.fail if cfg["require_unique"] else a.warn)(
            "duplicate_ids", idc, [i for v in dup.values() for i in v], n_groups=len(dup)
        )
        if not dup:
            a.warns = [w for w in a.warns if w["rule"] != "duplicate_ids"]
            a.fails = [w for w in a.fails if w["rule"] != "duplicate_ids"]
        if conf_ids:
            (a.fail if cfg["require_unique"] else a.warn)(
                "conflicting_rows_share_id", idc, conf_ids, n_groups=conflicting
            )
        if n_missing:
            a.warn(
                "missing_ids",
                idc,
                [i for i, (s, _) in zip(t.ids, st_, strict=True) if s == "missing"],
            )
    if t.duplicate_ids:
        a.warn("adapter_served_ids_twice", "sample_ids", t.duplicate_ids)
    cands: dict[str, Any] = {}
    for name in ctx.names(cs, t):
        kind = ctx.kind(name)
        valid = [v for s, v in _states(ctx, t, name) if s == "valid"]
        if len(valid) < int(cfg["min_samples"]):
            cands[name] = {"status": "INCONCLUSIVE", "n_valid": len(valid)}
            continue
        integral = kind == "NUMERIC" and all(float(v).is_integer() for v in valid)
        ratio = len(set(valid)) / len(valid)
        cands[name] = {
            "unique_ratio": ratio,
            "n_valid": len(valid),
            "eligible": kind != "NUMERIC" or integral,
        }
        if (kind != "NUMERIC" or integral) and ratio >= float(cfg["unique_ratio"]):
            a.warn(
                "identifier_like",
                name,
                [],
                unique_ratio=ratio,
                note="nearly every row has a distinct value; such a column can identify rows rather than describe them",
            )
    obs["candidates"] = cands
    obs["note"] = (
        "continuous numeric columns are not considered identifier candidates; integer-valued and categorical columns are"
    )
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "unique_ratio", "min_samples", "require_unique"),
        evidence={"n_rows": t.n},
    )


# -- numeric / categorical / distribution ------------------------------------------------------------------------


def _num_values(
    ctx: Ctx, t: Table, name: str
) -> tuple[list[int], list[float], dict[str, list[int]]]:
    ids, vals = [], []
    bad: dict[str, list[int]] = {"missing": [], "nonfinite": [], "invalid": []}
    for i, (s, v) in zip(t.ids, _states(ctx, t, name, "NUMERIC"), strict=True):
        if s == "valid":
            ids.append(i)
            vals.append(float(v))
        else:
            bad[s].append(i)
    return ids, vals, bad


def check_numeric(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    per: dict[str, Any] = {}
    stat: list[Status] = []
    for name in ctx.names(cs, t, ("NUMERIC",)):
        ids, vals, bad = _num_values(ctx, t, name)
        rec: dict[str, Any] = {
            "n_total": t.n,
            "n_valid": len(vals),
            "n_missing": len(bad["missing"]),
            "n_nonfinite": len(bad["nonfinite"]),
            "n_invalid": len(bad["invalid"]),
        }
        f = Acc(a.ex)
        if bad["nonfinite"]:
            f.warn("nonfinite_values", name, bad["nonfinite"])
        if bad["invalid"]:
            f.fail("invalid_type", name, bad["invalid"], declared_type="NUMERIC")
        if not vals:
            rec["status"] = "INCONCLUSIVE"
            rec["reason"] = "no valid values"
            per[name] = rec
            stat.append(Status.INCONCLUSIVE if not f.fails else Status.FAIL)
            a.fails += f.fails
            a.warns += f.warns
            continue
        s = st.summarize(vals)
        counts = Counter(vals)
        modal = counts.most_common(1)[0][1] / len(vals)
        var = st.summarize(vals).variance.value if len(vals) > 1 else None
        rec |= {
            "summary": to_jsonable(s),
            "n_unique": len(counts),
            "modal_fraction": modal,
            "constant": len(counts) == 1,
            "variance": var,
        }
        if len(counts) == 1:
            f.warn("constant_feature", name, [], value=vals[0])
        elif modal >= float(cfg["near_constant_ratio"]):
            f.warn(
                "near_constant_feature",
                name,
                [],
                modal_fraction=modal,
                configured=cfg["near_constant_ratio"],
            )
        if cfg["min_variance"] is not None and var is not None and var < cfg["min_variance"]:
            f.fail("variance_below_minimum", name, [], variance=var, configured=cfg["min_variance"])
        lo_hi = cfg["bounds"].get(name)
        if lo_hi is not None:
            lo, hi = lo_hi
            out = [
                i
                for i, x in zip(ids, vals, strict=True)
                if (lo is not None and x < lo) or (hi is not None and x > hi)
            ]
            rec["bounds"] = {
                "low": lo,
                "high": hi,
                "n_outside": len(out),
                "observed_min": min(vals),
                "observed_max": max(vals),
            }
            if out:
                f.fail("outside_bounds", name, out, low=lo, high=hi)
        flags, params = outliers(vals, str(cfg["outlier_method"]), float(cfg["outlier_k"]))
        out_ids = [i for i, fl in zip(ids, flags, strict=True) if fl]
        rec["outliers"] = {
            **params,
            "n_outliers": len(out_ids),
            "rate": len(out_ids) / len(vals),
            "affected_rows": sorted(out_ids)[: a.ex],
            "note": "extreme under this method; not evidence of error",
        }
        if (
            cfg["max_outlier_rate"] is not None
            and len(out_ids) / len(vals) > cfg["max_outlier_rate"]
        ):
            f.fail(
                "outlier_rate_exceeded",
                name,
                out_ids,
                rate=len(out_ids) / len(vals),
                configured=cfg["max_outlier_rate"],
            )
        rec["status"] = f.status.value
        per[name] = rec
        stat.append(f.status)
        a.fails += f.fails
        a.warns += f.warns
    if not per:
        return _res(cs, scope, Status.NOT_APPLICABLE, "no NUMERIC feature selected")
    status = worst(stat)
    return _res(
        cs,
        scope,
        status,
        observations={"features": per},
        violations=a.viols,
        thresholds=_thr(
            cs,
            "bounds",
            "near_constant_ratio",
            "min_variance",
            "max_outlier_rate",
            "outlier_method",
            "outlier_k",
        ),
        evidence={"n_rows": t.n, "n_features": len(per)},
    )


def _allowed_keys(cfg: Mapping[str, Any], name: str, kind: str) -> set[str] | None:
    vals = (cfg.get("allowed") or {}).get(name)
    if vals is None:
        return None
    return {
        ("true" if v else "false") if kind == "BOOLEAN" and isinstance(v, bool) else json_key(v)
        for v in vals
    }


def check_categorical(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    per: dict[str, Any] = {}
    stat: list[Status] = []
    for name in ctx.names(cs, t, ("CATEGORICAL", "BOOLEAN")):
        kind = ctx.kind(name)
        st_ = _states(ctx, t, name)
        valid = [(i, v) for i, (s, v) in zip(t.ids, st_, strict=True) if s == "valid"]
        counts = Counter(v for _, v in valid)
        f = Acc(a.ex)
        inv = [i for i, (s, _) in zip(t.ids, st_, strict=True) if s in ("invalid", "nonfinite")]
        if inv:
            f.fail("invalid_type", name, inv, declared_type=kind)
        allowed = _allowed_keys(cfg, name, kind)
        outside = sorted(k for k in counts if allowed is not None and k not in allowed)
        if outside:
            f.fail(
                "invalid_category",
                name,
                [i for i, v in valid if v in outside],
                categories=outside,
                note="not in the declared allowed set (distinct from a merely rare or newly observed category)",
            )
        n = len(valid)
        rare = sorted(
            k
            for k, c in counts.items()
            if cfg["rare_fraction"] is not None and n and c / n < cfg["rare_fraction"]
        )
        if cfg["max_cardinality"] is not None and len(counts) > cfg["max_cardinality"]:
            f.fail(
                "cardinality_exceeded",
                name,
                [],
                cardinality=len(counts),
                configured=cfg["max_cardinality"],
            )
        per[name] = {
            "declared_type": kind,
            "n_valid": n,
            "n_missing": sum(s == "missing" for s, _ in st_),
            "n_invalid": len(inv),
            "cardinality": len(counts),
            "counts": dict(sorted(counts.items())),
            "proportions": {k: c / n for k, c in sorted(counts.items())} if n else {},
            "rare_categories": rare,
            "rare_note": "rare means below rare_fraction of valid values; it is not invalid",
            "invalid_categories": outside,
            "allowed_declared": allowed is not None,
            "status": (Status.INCONCLUSIVE if n == 0 and not f.fails else f.status).value,
        }
        stat.append(Status.INCONCLUSIVE if n == 0 and not f.fails else f.status)
        a.fails += f.fails
        a.warns += f.warns
    if not per:
        return _res(cs, scope, Status.NOT_APPLICABLE, "no CATEGORICAL or BOOLEAN feature selected")
    return _res(
        cs,
        scope,
        worst(stat),
        observations={"features": per},
        violations=a.viols,
        thresholds=_thr(cs, "allowed", "rare_fraction", "max_cardinality"),
        evidence={"n_rows": t.n, "n_features": len(per)},
    )


def check_distribution(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    per: dict[str, Any] = {}
    stat: list[Status] = []
    for name in ctx.names(cs, t):
        kind = ctx.kind(name)
        f = Acc(a.ex)
        valid = [v for s, v in _states(ctx, t, name) if s == "valid"]
        if not valid:
            per[name] = {"status": "INCONCLUSIVE", "reason": "no valid values"}
            stat.append(Status.INCONCLUSIVE)
            continue
        counts = Counter(valid)
        modal_v, modal_n = counts.most_common(1)[0]
        rec: dict[str, Any] = {
            "kind": kind,
            "n_valid": len(valid),
            "n_unique": len(counts),
            "modal_value": modal_v,
            "modal_fraction": modal_n / len(valid),
        }
        if kind == "NUMERIC":
            fl = [float(v) for v in valid]
            skew, kurt = moments(fl)
            rec |= {
                "skewness": skew,
                "excess_kurtosis": kurt,
                "integer_valued_fraction": sum(x.is_integer() for x in fl) / len(fl),
                "min": min(fl),
                "max": max(fl),
            }
            if (
                cfg["max_abs_skew"] is not None
                and skew is not None
                and abs(skew) > cfg["max_abs_skew"]
            ):
                f.warn(
                    "skew_exceeds_configured",
                    name,
                    [],
                    skewness=skew,
                    configured=cfg["max_abs_skew"],
                )
        if (
            cfg["max_modal_fraction"] is not None
            and rec["modal_fraction"] > cfg["max_modal_fraction"]
        ):
            f.warn(
                "dominant_value",
                name,
                [],
                modal_fraction=rec["modal_fraction"],
                configured=cfg["max_modal_fraction"],
            )
        if cfg["min_unique"] is not None and len(counts) < cfg["min_unique"]:
            f.warn(
                "few_distinct_values", name, [], n_unique=len(counts), configured=cfg["min_unique"]
            )
        rec["status"] = f.status.value
        per[name] = rec
        stat.append(f.status)
        a.warns += f.warns
    if not per:
        return _res(cs, scope, Status.NOT_APPLICABLE, "no feature selected")
    return _res(
        cs,
        scope,
        worst(stat),
        observations={
            "features": per,
            "note": "distribution characteristics are descriptive; a heavy skew or a dominant value is not an error",
        },
        violations=a.viols,
        thresholds=_thr(cs, "max_abs_skew", "max_modal_fraction", "min_unique"),
        evidence={"n_rows": t.n, "n_features": len(per)},
    )


# -- target ------------------------------------------------------------------------------------------------------


def _type_name(x: Any) -> str:
    return (
        "bool"
        if isinstance(x, bool)
        else "number"
        if isinstance(x, int | float)
        else "str"
        if isinstance(x, str)
        else type(x).__name__
    )


def check_target(cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    if t.target is None:
        return _res(
            cs,
            scope,
            Status.UNAVAILABLE,
            "no target is available for this table (none served, or its length differs from the rows)",
            evidence={"n_rows": t.n, "target_len": t.target_len},
        )
    assert ctx.spec.target is not None  # noqa: S101
    a, cfg = Acc(int(cs.config["examples"])), cs.config
    task = ctx.spec.target.task
    raw = list(t.target)
    miss = [
        i
        for i, x in zip(t.ids, raw, strict=True)
        if x is None
        or (isinstance(x, float) and math.isnan(x))
        or (isinstance(x, str) and x in ctx.tokens)
    ]
    present = [(i, x) for i, x in zip(t.ids, raw, strict=True) if i not in set(miss)]
    obs: dict[str, Any] = {
        "task": task,
        "n_rows": t.n,
        "n_missing": len(miss),
        "missing_rate": len(miss) / t.n,
    }
    if miss:
        a.warn("missing_targets", "target", miss)
    if cfg["max_missing_rate"] is not None and len(miss) / t.n > cfg["max_missing_rate"]:
        a.fail(
            "missing_target_rate_exceeded",
            "target",
            [],
            rate=len(miss) / t.n,
            configured=cfg["max_missing_rate"],
        )
    stats_block: dict[str, Any] | None = None
    if task == "CLASSIFICATION":
        types = sorted({_type_name(x) for _, x in present})
        obs["value_types"] = types
        if len(types) > 1:
            a.fail("inconsistent_target_types", "target", [], types=types)
        nonfinite = [i for i, x in present if isinstance(x, float) and math.isinf(x)]
        if nonfinite:
            a.fail("nonfinite_target", "target", nonfinite)
        keyed = [(i, json_key(x)) for i, x in present if i not in set(nonfinite)]
        counts = Counter(k for _, k in keyed)
        n = len(keyed)
        allowed = None if cfg["allowed"] is None else {json_key(v) for v in cfg["allowed"]}
        if allowed is not None:
            out = sorted(k for k in counts if k not in allowed)
            if out:
                a.fail(
                    "invalid_target_value", "target", [i for i, k in keyed if k in out], values=out
                )
        rows = (
            {
                k: {"count": c, "proportion": c / n, "interval": _wilson(c, n, ctx)}
                for k, c in sorted(counts.items())
            }
            if n
            else {}
        )
        obs |= {
            "n_classes": len(counts),
            "classes": rows,
            "singleton_classes": sorted(k for k, c in counts.items() if c == 1),
        }
        if counts:
            hi, lo = max(counts.values()), min(counts.values())
            obs["imbalance_ratio"] = hi / lo
            obs["imbalance_note"] = (
                "class imbalance is descriptive and context-dependent; it is not a defect by itself"
            )
        if obs.get("singleton_classes"):
            a.warn(
                "singleton_class",
                "target",
                [i for i, k in keyed if k in obs["singleton_classes"]],
                classes=obs["singleton_classes"],
            )
        if cfg["min_class_count"] is not None:
            low = sorted(k for k, c in counts.items() if c < cfg["min_class_count"])
            if low:
                a.fail(
                    "class_below_min_count",
                    "target",
                    [i for i, k in keyed if k in low],
                    classes=low,
                    configured=cfg["min_class_count"],
                )
        if cfg["min_class_proportion"] is not None and n:
            low = sorted(k for k, c in counts.items() if c / n < cfg["min_class_proportion"])
            if low:
                a.fail(
                    "class_below_min_proportion",
                    "target",
                    [i for i, k in keyed if k in low],
                    classes=low,
                    configured=cfg["min_class_proportion"],
                )
        ok_n = n
    else:
        vals, ids = [], []
        inv, nonf = [], []
        for i, x in present:
            if isinstance(x, bool) or not isinstance(x, int | float):
                inv.append(i)
            elif math.isinf(x):
                nonf.append(i)
            else:
                vals.append(float(x))
                ids.append(i)
        if inv:
            a.fail("non_numeric_target", "target", inv)
        if nonf:
            a.warn("nonfinite_target", "target", nonf)
        ok_n = len(vals)
        if vals:
            flags, params = outliers(vals, str(cfg["outlier_method"]), float(cfg["outlier_k"]))
            oi = [i for i, fl in zip(ids, flags, strict=True) if fl]
            obs |= {
                "summary": to_jsonable(st.summarize(vals)),
                "outliers": {
                    **params,
                    "n_outliers": len(oi),
                    "affected_rows": sorted(oi)[: a.ex],
                    "note": "extreme under this method; not evidence of error",
                },
            }
    status = a.status
    if ok_n == 0 and not a.fails:
        status = Status.INCONCLUSIVE
    return _res(
        cs,
        scope,
        status,
        "no valid target values" if status is Status.INCONCLUSIVE else None,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(
            cs, "allowed", "min_class_count", "min_class_proportion", "max_missing_rate"
        ),
        evidence={"n_rows": t.n, "n_valid_targets": ok_n},
        statistics=stats_block,
    )


# -- leakage indicators ------------------------------------------------------------------------------------------


def check_target_leakage(
    cs: CheckSpec, ctx: Ctx, t: Table, scope: Mapping[str, Any]
) -> CheckResult:
    if (e := _empty(cs, scope, t)) is not None:
        return e
    if t.target is None:
        return _res(cs, scope, Status.UNAVAILABLE, "no target is available for this table")
    assert ctx.spec.target is not None  # noqa: S101
    cfg, a = cs.config, Acc(int(cs.config["examples"]))
    per: dict[str, Any] = {}
    stat: list[Status] = []
    tkind = "NUMERIC" if ctx.spec.target.task == "REGRESSION" else "CATEGORICAL"
    tstates = classify(list(t.target), tkind, ctx.tokens)
    for name in cfg["candidates"]:
        kind = ctx.kind(name)
        fstates = _states(ctx, t, name)
        pairs = [
            (i, fv, tv)
            for i, (fs, fv), (ts, tv) in zip(t.ids, fstates, tstates, strict=True)
            if fs == "valid" and ts == "valid"
        ]
        if len(pairs) < int(cfg["min_rows"]):
            per[name] = {
                "status": "INCONCLUSIVE",
                "n_pairs": len(pairs),
                "reason": f"fewer than min_rows={cfg['min_rows']} rows with both values",
            }
            stat.append(Status.INCONCLUSIVE)
            continue
        rec: dict[str, Any] = {"n_pairs": len(pairs)}
        flagged: list[str] = []
        raw_t = dict(zip(t.ids, t.target, strict=True))
        eq = sum(_same(fv, raw_t[i]) for i, fv, _ in pairs) / len(pairs)
        rec["equality_rate"] = eq
        if eq >= float(cfg["max_abs_correlation"]):
            flagged.append("identical_to_target")
        if kind == "NUMERIC" and tkind == "NUMERIC":
            r = pearson([float(fv) for _, fv, _ in pairs], [float(tv) for _, _, tv in pairs])
            rec["pearson_r"] = r
            if r is not None and abs(r) >= float(cfg["max_abs_correlation"]):
                flagged.append("near_perfect_correlation")
        if kind != "NUMERIC" or all(float(fv).is_integer() for _, fv, _ in pairs):
            by_val: dict[Any, Counter[Any]] = {}
            for _, fv, tv in pairs:
                by_val.setdefault(fv, Counter())[tv] += 1
            supported = {k: c for k, c in by_val.items() if sum(c.values()) >= 2}
            cover = sum(sum(c.values()) for c in supported.values()) / len(pairs)
            purity = sum(max(c.values()) for c in supported.values()) / max(
                1, sum(sum(c.values()) for c in supported.values())
            )
            base = Counter(tv for _, _, tv in pairs).most_common(1)[0][1] / len(pairs)
            rec |= {
                "n_distinct_values": len(by_val),
                "support_fraction": cover,
                "purity": purity,
                "target_majority_share": base,
            }
            if (
                tkind == "CATEGORICAL"
                and len(by_val) >= 2
                and cover >= 0.5
                and purity >= float(cfg["min_purity"])
                and base < float(cfg["min_purity"])
            ):
                flagged.append("value_determines_target")
        rec["indicators"] = flagged
        rec["status"] = "WARNING" if flagged else "PASS"
        per[name] = rec
        stat.append(Status.WARNING if flagged else Status.PASS)
        if flagged:
            a.warn(
                "leakage_indicator",
                name,
                [],
                indicators=flagged,
                note="an indicator in these rows that needs interpretation; it is not proof of leakage",
            )
    return _res(
        cs,
        scope,
        worst(stat),
        observations={
            "candidates": per,
            "note": "only the explicitly named candidates were examined; indicators are evidence, not conclusions",
        },
        violations=a.viols,
        thresholds=_thr(cs, "max_abs_correlation", "min_purity", "min_rows"),
        evidence={"n_rows": t.n, "n_candidates": len(per)},
    )


def _same(fv: Any, tv: Any) -> bool:
    try:
        return (bool(fv == tv) and _type_name(fv) == _type_name(tv)) or (
            isinstance(fv, int | float)
            and isinstance(tv, int | float)
            and not isinstance(tv, bool)
            and float(fv) == float(tv)
        )
    except (TypeError, ValueError):
        return False


# -- cross-split checks --------------------------------------------------------------------------------------------


def check_split_overlap(cs: CheckSpec, ctx: Ctx) -> CheckResult:
    cfg = cs.config
    ref, cmp = ctx.tables[cfg["reference"]], ctx.tables[cfg["comparison"]]
    scope = {"reference": cfg["reference"], "comparison": cfg["comparison"]}
    a = Acc(int(cfg["examples"]))
    names = list(
        cfg["features"]
        or (ctx.spec.features and [f.name for f in ctx.spec.features])
        or ref.columns
    )
    common = [n for n in names if n in ref.columns and n in cmp.columns]
    idov = sorted(set(ref.ids) & set(cmp.ids))
    rcols, ccols = [ref.column(n) for n in common], [cmp.column(n) for n in common]
    rvec: dict[tuple[str, ...], list[int]] = {}
    for p in range(ref.n):
        rvec.setdefault(tuple(cell_key(c[p]) for c in rcols), []).append(p)
    hits: list[int] = []
    conflict: list[int] = []
    agree = 0
    for p, i in enumerate(cmp.ids):
        k = tuple(cell_key(c[p]) for c in ccols)
        if k in rvec:
            hits.append(i)
            if ref.target is not None and cmp.target is not None:
                ts = {cell_key(ref.target[q]) for q in rvec[k]}
                if cell_key(cmp.target[p]) not in ts:
                    conflict.append(i)
                else:
                    agree += 1
    mx = cfg["max_overlap"]
    if idov:
        (a.fail if mx is not None and len(idov) > mx else a.warn)("sample_id_overlap", "ids", idov)
    if hits:
        (a.fail if mx is not None and len(hits) > mx else a.warn)(
            "identical_feature_vector_across_splits", "rows", hits
        )
    if conflict:
        a.warn("identical_features_different_target_across_splits", "rows", conflict)
    if not common:
        return _res(cs, scope, Status.NOT_APPLICABLE, "the splits share no compared column")
    obs = {
        "n_reference": ref.n,
        "n_comparison": cmp.n,
        "columns_compared": common,
        "sample_id_overlap": len(idov),
        "comparison_rows_with_identical_vector_in_reference": len(hits),
        "label_conflicts": len(conflict),
        "label_agreements": agree,
        "labels_compared": ref.target is not None and cmp.target is not None,
        "note": "overlap between declared splits is a leakage INDICATOR: it may be intended (repeated measurements) or a split defect; interpretation is required",
    }
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "max_overlap"),
        evidence={"n_reference": ref.n, "n_comparison": cmp.n},
    )


def order_keys(t: Table, ordering_field: str) -> tuple[dict[int, float], dict[str, list[int]]]:
    """Ordering key per sample ID and the IDs with a missing / non-finite / invalid key."""
    keys: dict[int, float] = {}
    bad: dict[str, list[int]] = {"MISSING": [], "NON_FINITE": [], "INVALID": []}
    if ordering_field == INDEX:
        return {i: float(i) for i in t.ids}, bad
    for i, x in zip(t.ids, t.column(ordering_field[len("feature:") :]), strict=True):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            bad["MISSING"].append(i)
        elif isinstance(x, bool) or not isinstance(x, int | float):
            bad["INVALID"].append(i)
        elif not math.isfinite(x):
            bad["NON_FINITE"].append(i)
        else:
            keys[i] = float(x)
    return keys, bad


def check_temporal_order(cs: CheckSpec, ctx: Ctx) -> CheckResult:
    cfg = cs.config
    assert ctx.spec.ordering is not None  # noqa: S101
    ref, cmp = ctx.tables[cfg["reference"]], ctx.tables[cfg["comparison"]]
    scope = {"reference": cfg["reference"], "comparison": cfg["comparison"]}
    a = Acc(int(cfg["examples"]))
    rk, rbad = order_keys(ref, ctx.spec.ordering.field)
    ck, cbad = order_keys(cmp, ctx.spec.ordering.field)
    if not rk or not ck:
        return _res(
            cs,
            scope,
            Status.INCONCLUSIVE,
            "a split has no usable ordering keys",
            observations={
                "reference_missing_keys": {k: len(v) for k, v in rbad.items()},
                "comparison_missing_keys": {k: len(v) for k, v in cbad.items()},
            },
        )
    rmax, cmin = max(rk.values()), min(ck.values())
    tie = bool(cfg["allow_ties"])
    late_ref = [i for i, k in rk.items() if (k > cmin if tie else k >= cmin)]
    early_cmp = [i for i, k in ck.items() if (k < rmax if tie else k <= rmax)]
    if early_cmp:
        a.warn("comparison_rows_not_after_reference", "comparison", early_cmp, reference_max=rmax)
    if late_ref:
        a.warn("reference_rows_not_before_comparison", "reference", late_ref, comparison_min=cmin)
    for label, bad in (("reference", rbad), ("comparison", cbad)):
        ids = [i for v in bad.values() for i in v]
        if ids:
            a.warn(
                "unusable_ordering_key", label, ids, kinds={k: len(v) for k, v in bad.items() if v}
            )
    obs = {
        "ordering": ctx.spec.ordering.field,
        "reference_range": [min(rk.values()), rmax],
        "comparison_range": [cmin, max(ck.values())],
        "n_comparison_not_after_reference": len(early_cmp),
        "n_reference_not_before_comparison": len(late_ref),
        "allow_ties": tie,
        "note": "the ordering is the declared one; a violation means the splits are not separated in that order, which may or may not matter for the intended use",
    }
    return _res(
        cs,
        scope,
        a.status,
        observations=obs,
        violations=a.viols,
        thresholds=_thr(cs, "allow_ties"),
        evidence={"n_reference": len(rk), "n_comparison": len(ck)},
    )


# -- group comparison ---------------------------------------------------------------------------------------------


def resolve_group(ctx: Ctx, g: Mapping[str, Any], role: Role) -> Table:
    t = ctx.tables[g["split"]]
    if g["window"] is None:
        return t
    assert ctx.spec.ordering is not None  # noqa: S101
    w = TemporalWindow.from_dict(g["window"], role)
    keys, _ = order_keys(t, ctx.spec.ordering.field)
    return t.select([p for p, i in enumerate(t.ids) if i in keys and w.contains(keys[i])])


def _rate_compare(x: Sequence[int], y: Sequence[int], ctx: Ctx) -> dict[str, Any]:
    c = st.compare(
        [float(v) for v in x],
        [float(v) for v in y],
        pairing=Pairing.UNPAIRED,
        method=ctx.cfg.method,
        confidence=ctx.cfg.confidence,
        resamples=ctx.cfg.resamples,
        permutations=ctx.cfg.permutations,
        seed=ctx.cfg.seed,
    )
    d = to_jsonable(c)
    assert isinstance(d, dict)  # noqa: S101
    return d


def _gname(g: Mapping[str, Any]) -> str:
    w = g["window"]
    return str(g["split"]) + ("" if w is None else f" [{w['start']}, {w['end']})")


def check_group_comparison(cs: CheckSpec, ctx: Ctx) -> CheckResult:
    cfg = cs.config
    ref = resolve_group(ctx, cfg["reference"], Role.REFERENCE)
    cmp = resolve_group(ctx, cfg["comparison"], Role.COMPARISON)
    scope = {"reference": _gname(cfg["reference"]), "comparison": _gname(cfg["comparison"])}
    n = ctx.cfg.min_members
    same_split = cfg["reference"]["split"] == cfg["comparison"]["split"]
    if same_split and set(ref.ids) & set(cmp.ids):
        return _res(
            cs,
            scope,
            Status.INCONCLUSIVE,
            "the groups share samples; an independent-groups comparison would be invalid",
            evidence={"n_reference": ref.n, "n_comparison": cmp.n},
        )
    if ref.n == 0 or cmp.n == 0:
        return _res(
            cs,
            scope,
            Status.INCONCLUSIVE,
            "a group has no rows",
            evidence={"n_reference": ref.n, "n_comparison": cmp.n},
        )
    a = Acc(int(cfg["examples"]))
    aspects = list(cfg["aspects"])
    names = list(cfg["features"]) or (sorted(ctx.kinds) if ctx.kinds else list(ref.columns))
    obs: dict[str, Any] = {}
    families: dict[str, dict[str, float | None]] = {}
    enough = ref.n >= n and cmp.n >= n
    if "missingness" in aspects:
        per = {}
        for name in names:
            xs = [int(s == "missing") for s, _ in _states(ctx, ref, name)]
            ys = [int(s == "missing") for s, _ in _states(ctx, cmp, name)]
            rec: dict[str, Any] = {
                "reference": {
                    "n_missing": sum(xs),
                    "rate": sum(xs) / ref.n,
                    "interval": _wilson(sum(xs), ref.n, ctx),
                },
                "comparison": {
                    "n_missing": sum(ys),
                    "rate": sum(ys) / cmp.n,
                    "interval": _wilson(sum(ys), cmp.n, ctx),
                },
                "difference": sum(ys) / cmp.n - sum(xs) / ref.n,
            }
            if enough:
                rec["comparison_statistics"] = _rate_compare(xs, ys, ctx)
                families.setdefault("missingness", {})[name] = rec["comparison_statistics"]["test"][
                    "p_value"
                ]
            else:
                rec["status"] = "INCONCLUSIVE"
                rec["reason"] = (
                    f"fewer than min_members={n} rows in a group; rates are reported, inference is withheld"
                )
            per[name] = rec
        obs["missingness"] = per
    if "validity" in aspects:
        per = {}
        for name in [x for x in names if ctx.kinds.get(x)]:
            vx = [int(s in ("invalid", "nonfinite")) for s, _ in _states(ctx, ref, name)]
            vy = [int(s in ("invalid", "nonfinite")) for s, _ in _states(ctx, cmp, name)]
            rec = {
                "reference_invalid_rate": sum(vx) / ref.n,
                "comparison_invalid_rate": sum(vy) / cmp.n,
                "difference": sum(vy) / cmp.n - sum(vx) / ref.n,
                "n_invalid": [sum(vx), sum(vy)],
            }
            if enough and (any(vx) or any(vy)):
                rec["comparison_statistics"] = _rate_compare(vx, vy, ctx)
                families.setdefault("validity", {})[name] = rec["comparison_statistics"]["test"][
                    "p_value"
                ]
            per[name] = rec
        obs["validity"] = per
    if "categories" in aspects:
        per = {}
        allowed_map: dict[str, set[str]] = {}
        for c in ctx.spec.checks:
            if c.type == "categorical":
                for nm in c.config["allowed"]:
                    allowed_map[nm] = _allowed_keys(c.config, nm, ctx.kind(nm)) or set()
        rare_f = next(
            (c.config["rare_fraction"] for c in ctx.spec.checks if c.type == "categorical"), 0.01
        )
        for name in [x for x in names if ctx.kinds.get(x) in ("CATEGORICAL", "BOOLEAN")]:
            rv = Counter(v for s, v in _states(ctx, ref, name) if s == "valid")
            cv = Counter(v for s, v in _states(ctx, cmp, name) if s == "valid")
            nc = sum(cv.values())
            new = [
                {
                    "category": k,
                    "count": c,
                    "fraction": c / nc,
                    "declared_invalid": None
                    if name not in allowed_map
                    else k not in allowed_map[name],
                    "rare": bool(rare_f) and c / nc < rare_f,
                }
                for k, c in sorted(cv.items())
                if k not in rv
            ]
            per[name] = {
                "reference_cardinality": len(rv),
                "comparison_cardinality": len(cv),
                "observed_new_in_comparison": new,
                "absent_from_comparison": sorted(k for k in rv if k not in cv),
                "note": "new means observed only in the comparison group; declared_invalid says whether the schema forbids it; rare says it is below the rare fraction; these are three different facts",
            }
            if new:
                a.warn("observed_new_category", name, [], categories=[x["category"] for x in new])
        obs["categories"] = per
    stats_block: dict[str, Any] = {}
    if "target" in aspects and ctx.spec.target is not None:
        if ref.target is None or cmp.target is None:
            obs["target"] = {"status": "UNAVAILABLE", "reason": "no aligned target in a group"}
        else:
            dc = DriftConfig(
                min_samples=n,
                confidence=ctx.cfg.confidence,
                resamples=ctx.cfg.resamples,
                permutations=ctx.cfg.permutations,
                seed=ctx.cfg.seed,
                method=ctx.cfg.method,
                correction=ctx.cfg.correction,
                alpha=ctx.cfg.alpha,
                dimensions=("label", "performance", "prediction"),
            )
            kind = "CATEGORICAL" if ctx.spec.target.task == "CLASSIFICATION" else "NUMERIC"
            r = drift_measures.shift(
                "TARGET", "target", kind, list(ref.target), list(cmp.target), dc
            ).to_dict()
            obs["target"] = r
            if r.get("test"):
                families.setdefault("target", {})["target"] = r["test"]["p_value"]
    for aspect, ps in sorted(families.items()):
        corr = st.adjust_pvalues(dict(ps), method=ctx.cfg.correction, alpha=ctx.cfg.alpha)
        cdoc = to_jsonable(corr)
        assert isinstance(cdoc, dict)  # noqa: S101
        stats_block[aspect] = {
            **cdoc,
            "family": f"{aspect}: one hypothesis per feature between these two groups",
            "raw_p": ps,
        }
    status = Status.WARNING if a.warns else Status.INCONCLUSIVE if not enough else Status.PASS
    reason = (
        None if enough else f"a group has fewer than min_members={n} rows: descriptive values only"
    )
    return _res(
        cs,
        scope,
        status,
        reason,
        observations=obs,
        violations=a.viols,
        thresholds={"min_members": n},
        evidence={
            "n_reference": ref.n,
            "n_comparison": cmp.n,
            "reference_digest": ref.digest(),
            "comparison_digest": cmp.digest(),
        },
        statistics=stats_block or None,
    )


PER_TABLE_FUNCS = {
    "schema": check_schema,
    "sample_count": check_sample_count,
    "missingness": check_missingness,
    "duplicates": check_duplicates,
    "identifiers": check_identifiers,
    "numeric": check_numeric,
    "categorical": check_categorical,
    "target": check_target,
    "distribution": check_distribution,
    "target_leakage": check_target_leakage,
}
CROSS_FUNCS = {
    "split_overlap": check_split_overlap,
    "temporal_order": check_temporal_order,
    "group_comparison": check_group_comparison,
}
