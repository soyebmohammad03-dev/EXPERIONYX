"""Benchmark specification identity, protocol expansion and refusal of invalid definitions."""

from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_helpers import EVAL, FR, BenchmarkLimits, ids, small_world, spec_for
from eval_helpers import EvalWorld
from experionyx.benchmark.protocol import expand, validation_report
from experionyx.benchmark.spec import BenchmarkSpec, FaultGrid, InteractionPair
from experionyx.benchmark.taxonomy import UnitKind
from experionyx.errors import BenchmarkRefusal, ValidationError
from experionyx.evaluation.config import EvaluationConfig
from experionyx.interactions.config import InteractionConfig


@pytest.fixture(scope="module")
def w(tmp_path_factory: pytest.TempPathFactory) -> EvalWorld:
    return small_world(tmp_path_factory.mktemp("bm"))


def codes(w: EvalWorld, spec: BenchmarkSpec) -> set[str]:
    with pytest.raises(BenchmarkRefusal) as exc:
        expand(w.registry, spec, FR)
    return {i.code for i in exc.value.issues}  # type: ignore[attr-defined]


# -- identity ---------------------------------------------------------------------------------------


def test_spec_identity_is_deterministic_and_roundtrips(w: EvalWorld) -> None:
    s = spec_for(w)
    assert s.spec_id == spec_for(w).spec_id == BenchmarkSpec.from_dict(s.to_dict()).spec_id
    assert BenchmarkSpec.from_dict(s.to_dict()) == s
    assert (
        spec_for(w, faults=tuple(reversed(s.faults))).spec_id == s.spec_id
    )  # listing order is not identity
    assert spec_for(w, seeds=(3, 1, 2)).spec_id == s.spec_id


def test_every_scientifically_meaningful_change_changes_the_identity(w: EvalWorld) -> None:
    s = spec_for(w)
    drop, noise = s.faults  # canonical order is by name
    changes = {
        "version": spec_for(w, version="1.0.1"), "name": spec_for(w, name="other"), "seeds": spec_for(w, seeds=(1, 2, 4)),
        "one more seed": spec_for(w, seeds=(1, 2, 3, 4)), "grid value": spec_for(w, faults=(replace(noise, values=(2.0, 7.0)), drop)),
        "extra grid point": spec_for(w, faults=(replace(noise, values=(2.0, 4.0, 6.0)), drop)),
        "fixed parameter": spec_for(w, faults=(FaultGrid("noise", "gaussian_noise", parameters={"sigma": 1.0}), drop), interactions=()),
        "fault version pin": spec_for(w, faults=(replace(noise, fault_version="1"), drop)), "scope": spec_for(w, faults=(replace(noise, scope_fraction=0.5), drop)),
        "evaluation": spec_for(w, evaluation=EvaluationConfig(split="test", batch_size=7)), "primary metric": spec_for(w, primary_metric="f1"),
        "aggregation seed": spec_for(w, aggregation_seed=1), "aggregation resamples": spec_for(w, aggregation_resamples=201),
        "interaction bootstrap": spec_for(w, interaction_config=InteractionConfig(bootstrap_resamples=201)),
        "interaction point": spec_for(w, interactions=(InteractionPair("noise", "dropout", 2.0, 0.5),)), "order analysis": spec_for(w, interactions=(InteractionPair("noise", "dropout", 6.0, 0.5, True),)),
        "no interactions": spec_for(w, interactions=()), "discovery off": spec_for(w, discovery=False), "profile off": spec_for(w, profile=False),
        "limits": spec_for(w, limits=BenchmarkLimits(max_units=99)),
    }  # fmt: skip
    ids_ = {s.spec_id}
    for name, c in changes.items():
        assert c.spec_id not in ids_, f"{name} did not change the identity"
        ids_.add(c.spec_id)


def test_protocol_key_ignores_only_the_model(w: EvalWorld) -> None:
    s = spec_for(w)
    other_model = replace(s, model="mdl_" + "0" * 32)
    assert other_model.spec_id != s.spec_id and other_model.protocol_key() == s.protocol_key()
    assert replace(s, version="2.0.0").protocol_key() != s.protocol_key()


def test_spec_validation_is_strict() -> None:
    model, data = "mdl_" + "0" * 32, "dst_" + "0" * 32
    grid = FaultGrid("g", "gaussian_noise", sweep_parameter="sigma", values=(1.0,))

    def make(**kw):  # type: ignore[no-untyped-def]
        base = {
            "name": "b",
            "version": "1.0.0",
            "model": model,
            "dataset": data,
            "faults": (grid,),
            "seeds": (1,),
        }
        base.update(kw)
        return BenchmarkSpec(**base)

    for bad in (
        dict(faults=()),
        dict(faults=(grid, grid)),
        dict(seeds=()),
        dict(seeds=(1, 1)),
        dict(seeds=(-1,)),
        dict(version="one"),
        dict(model="x"),
        dict(aggregation_confidence=1.0),
        dict(interactions=(InteractionPair("g", "h"), InteractionPair("g", "h"))),
    ):
        with pytest.raises(ValidationError):
            make(**bad)
    for bad_grid in (
        dict(values=(1.0, 1.0)),
        dict(values=(float("nan"),)),
        dict(sweep_parameter=None),
        dict(parameters={"sigma": 1.0}),
        dict(scope_fraction=0.0),
    ):
        with pytest.raises(ValidationError):
            FaultGrid(
                "g", "gaussian_noise", **{"sweep_parameter": "sigma", "values": (1.0,), **bad_grid}
            )  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        InteractionPair("a", "a")
    with pytest.raises(ValidationError):
        BenchmarkSpec.from_dict({**make().to_dict(), "surprise": 1})
    with pytest.raises(ValidationError, match="missing"):
        BenchmarkSpec.from_dict({"name": "b"})
    with pytest.raises(ValidationError):
        BenchmarkSpec.from_dict({**make().to_dict(), "aggregation": {"confidence": 0.9, "typo": 1}})


# -- expansion --------------------------------------------------------------------------------------------


def test_expansion_makes_every_requested_point_an_explicit_traceable_unit(w: EvalWorld) -> None:
    s = spec_for(w)
    p = expand(w.registry, s, FR)
    assert p.spec_id == s.spec_id and p.engine_version
    by = {k: p.by_kind(k) for k in UnitKind}
    assert len(by[UnitKind.BASELINE]) == 1
    assert (
        len(by[UnitKind.FAULT_TRIAL]) == (2 + 1) * 3
    )  # (sigma x2 + p x1) x 3 seeds: nothing skipped, nothing interpolated
    assert (
        len(by[UnitKind.INTERACTION_TRIAL]) == 3
    )  # only the AB compound is new: cells A and B reuse the grid trials at their points
    assert p.pairs == {
        "noise+dropout": {
            "a_grid": "noise",
            "a_point_index": 1,
            "b_grid": "dropout",
            "b_point_index": 0,
            "order_analysis": False,
            "reuses_grid_trials": True,
        }
    }
    assert {(u.grid, u.value, u.seed) for u in by[UnitKind.FAULT_TRIAL]} == {
        (g, v, sd)
        for g, vals in (("noise", (2.0, 6.0)), ("dropout", (0.5,)))
        for v in vals
        for sd in (1, 2, 3)
    }
    assert len({u.key for u in p.units}) == len(p.units) and len(
        {u.fault_id for u in p.units if u.fault_id}
    ) == len([u for u in p.units if u.fault_id])
    u = next(x for x in p.units if x.key == "fault:noise:1:2")
    assert (
        u.fault is not None
        and u.fault["parameters"]["sigma"] == 6.0
        and u.fault["seed"] == 2
        and u.fault_id
        and u.point_index == 1
        and u.parameter == "sigma"
    )
    cell = next(x for x in p.units if x.key == "interaction:noise+dropout:AB:1")
    assert cell.fault is not None and [c["type"] for c in cell.fault["components"]] == [
        "gaussian_noise",
        "feature_dropout",
    ]  # ordered composition
    assert p.resolved == {
        "noise": {"type": "gaussian_noise", "version": "1"},
        "dropout": {"type": "feature_dropout", "version": "1"},
    }
    ba = expand(
        w.registry,
        spec_for(w, interactions=(InteractionPair("noise", "dropout", 6.0, 0.5, True),)),
        FR,
    )
    assert len(ba.by_kind(UnitKind.INTERACTION_TRIAL)) == 2 * 3  # AB and BA
    assert [
        c["type"] for c in next(x for x in ba.units if x.key.endswith(":BA:1")).fault["components"]
    ] == ["feature_dropout", "gaussian_noise"]  # type: ignore[index]


def test_protocol_hash_is_deterministic_model_independent_and_sensitive(w: EvalWorld) -> None:
    s = spec_for(w)
    h = expand(w.registry, s, FR).protocol_hash
    assert expand(w.registry, spec_for(w), FR).protocol_hash == h
    assert expand(w.registry, spec_for(w, seeds=(1, 2, 4)), FR).protocol_hash != h
    assert (
        expand(w.registry, spec_for(w, version="1.0.1"), FR).protocol_hash != h
    )  # a definition change is detectable
    assert expand(w.registry, spec_for(w, interactions=()), FR).protocol_hash != h


def test_a_fault_version_change_is_detectable_in_the_protocol(w: EvalWorld) -> None:
    from experionyx.faults.spec import FaultRegistry

    latest = expand(w.registry, spec_for(w), FR)
    bumped = FaultRegistry()
    from dataclasses import replace as dc_replace

    for ft in FR.all():
        bumped.register(dc_replace(ft, version="2") if ft.name == "gaussian_noise" else ft)
    p2 = expand(w.registry, spec_for(w), bumped)
    assert (
        p2.resolved["noise"]["version"] == "2" and p2.protocol_hash != latest.protocol_hash
    )  # same spec, different fault implementation
    with pytest.raises(BenchmarkRefusal):
        expand(
            w.registry,
            spec_for(
                w,
                faults=(
                    FaultGrid(
                        "noise",
                        "gaussian_noise",
                        fault_version="1",
                        sweep_parameter="sigma",
                        values=(2.0,),
                    ),
                ),
                interactions=(),
            ),
            bumped,
        )  # a pinned version that no longer exists is refused


# -- refusal -----------------------------------------------------------------------------------------------


def test_invalid_definitions_are_refused_before_anything_runs_with_every_issue(
    w: EvalWorld,
) -> None:
    model, data = ids(w)
    assert codes(w, spec_for(w, model="mdl_" + "0" * 32)) == {"MODEL_MISSING"}
    assert codes(w, spec_for(w, dataset="dst_" + "0" * 32)) == {"DATASET_MISSING"}
    assert codes(w, spec_for(w, faults=(FaultGrid("x", "no_such_fault"),), interactions=())) == {
        "UNKNOWN_FAULT"
    }
    assert codes(w, spec_for(w, faults=(FaultGrid("x", "covariate_shift"),), interactions=())) == {
        "FAULT_NOT_IMPLEMENTED"
    }  # reserved vocabulary
    assert codes(
        w,
        spec_for(
            w,
            faults=(FaultGrid("x", "gaussian_noise", sweep_parameter="sigma", values=(-1.0,)),),
            interactions=(),
        ),
    ) == {"INVALID_GRID"}  # negative sigma
    assert codes(
        w,
        spec_for(
            w,
            faults=(FaultGrid("x", "gaussian_noise", sweep_parameter="nope", values=(1.0,)),),
            interactions=(),
        ),
    ) == {"INVALID_GRID"}
    assert codes(w, spec_for(w, primary_metric="not_a_metric")) == {"UNKNOWN_METRIC"}
    assert codes(w, spec_for(w, interaction_config=InteractionConfig(order_analysis=True))) == {
        "ORDER_ANALYSIS_ON_CONFIG"
    }
    assert codes(w, spec_for(w, limits=BenchmarkLimits(max_units=5))) == {"MAX_UNITS"}
    assert codes(w, spec_for(w, interactions=(InteractionPair("noise", "ghost", 6.0),))) == {
        "INTERACTION_UNKNOWN_GRID"
    }
    assert codes(w, spec_for(w, interactions=(InteractionPair("noise", "dropout", 3.0, 0.5),))) == {
        "INTERACTION_VALUE"
    }  # 3.0 is not a tested point
    assert codes(
        w, spec_for(w, interactions=(InteractionPair("noise", "dropout", None, 0.5),))
    ) == {"INTERACTION_VALUE"}
    both = codes(
        w,
        spec_for(
            w, primary_metric="nope", faults=(FaultGrid("x", "no_such_fault"),), interactions=()
        ),
    )
    assert both == {
        "UNKNOWN_METRIC",
        "UNKNOWN_FAULT",
    }  # every problem is listed, not just the first
    assert model and data


def test_unsupported_grids_are_refused_by_default_and_recorded_when_allowed(w: EvalWorld) -> None:
    grid = FaultGrid(
        "labels", "label_flip", sweep_parameter="rate", values=(0.1,)
    )  # label faults need a class-labelled dataset; this one is fine
    grid2 = FaultGrid(
        "blur", "random_occlusion", sweep_parameter="fraction", values=(0.5,)
    )  # image-only fault on tabular data
    assert "UNSUPPORTED_FAULT" in codes(w, spec_for(w, faults=(grid, grid2), interactions=()))
    p = expand(
        w.registry,
        spec_for(
            w, faults=(grid, grid2), interactions=(), limits=BenchmarkLimits(allow_unsupported=True)
        ),
        FR,
    )
    assert (
        [u["grid"] for u in p.unsupported] == ["blur"]
        and "blur" not in {u.grid for u in p.units}
        and p.unsupported[0]["reason"]
    )
    assert {u.grid for u in p.units if u.grid} == {
        "labels"
    }  # the supported grid is unaffected and the unsupported one is on record
    assert "UNSUPPORTED_FAULT" not in codes(
        w,
        spec_for(
            w,
            primary_metric="nope",
            faults=(grid2,),
            interactions=(),
            limits=BenchmarkLimits(allow_unsupported=True),
        ),
    )
    assert "INTERACTION_GRID_UNUSABLE" in codes(
        w,
        spec_for(
            w,
            faults=(grid2, FaultGrid("n", "gaussian_noise")),
            interactions=(InteractionPair("blur", "n", 1.0, None),),
            limits=BenchmarkLimits(allow_unsupported=True),
        ),
    )


def test_validation_report_is_structured_and_non_raising(w: EvalWorld) -> None:
    ok = validation_report(w.registry, spec_for(w), FR)
    assert (
        ok["valid"] is True
        and ok["units"] == 1 + 9 + 3
        and ok["units_by_kind"]["FAULT_TRIAL"] == 9
        and "fault:noise:0:1" in ok["unit_keys"]
    )
    few = validation_report(w.registry, spec_for(w, seeds=(1, 2)), FR)
    assert few["valid"] is True and any(
        "no interval can be produced" in x for x in few["warnings"]
    )  # honest about what will be inconclusive
    bad = validation_report(w.registry, spec_for(w, primary_metric="nope"), FR)
    assert (
        bad["valid"] is False
        and bad["issues"][0]["required"]
        and bad["issues"][0]["found"]
        and bad["issues"][0]["why"]
    )


def test_unused_imports_are_real(tmp_path: Path) -> None:
    assert EVAL.split == "test" and tmp_path
