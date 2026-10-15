from pathlib import Path

import pytest

from seq_align import run_jack


class FakeHit:
    def __init__(self, name: str, *, included: bool = False, reported: bool = False, new: bool = False, dropped: bool = False, duplicate: bool = False):
        self.name = name
        self.included = included
        self.reported = reported
        self.new = new
        self.dropped = dropped
        self.duplicate = duplicate


class FakeTopHits:
    searched_models = 3
    searched_nodes = 100
    searched_sequences = 50
    searched_residues = 5000

    def __init__(self, hits):
        self._hits = list(hits)

    def __len__(self):
        return len(self._hits)

    def __iter__(self):
        return iter(self._hits)

    def sort(self):
        return None

    @property
    def reported(self):
        return [hit for hit in self._hits if hit.reported]

    @property
    def included(self):
        return [hit for hit in self._hits if hit.included]


class FakeIterationResult:
    def __init__(self, *, iteration: int, converged: bool, hits, hmm=None):
        self.iteration = iteration
        self.converged = converged
        self.hits = hits
        self.hmm = hmm


def test_tophits_counts_does_not_expose_hit_names():
    hits = FakeTopHits([
        FakeHit("target_secret_a", included=True, reported=True, new=True),
        FakeHit("target_secret_b", reported=True, dropped=True),
        FakeHit("target_secret_c", duplicate=True),
    ])

    counts = run_jack._tophits_counts(hits)

    assert counts["total"] == 3
    assert counts["reported"] == 2
    assert counts["included"] == 1
    assert counts["new"] == 1
    assert counts["dropped"] == 1
    assert counts["duplicate"] == 1
    assert counts["searched_sequences"] == 50
    assert "target_secret" not in str(counts)


def test_iteration_profile_records_counts_only():
    hits = FakeTopHits([
        FakeHit("target_a", included=True, reported=True),
        FakeHit("target_b", reported=False),
    ])
    result = FakeIterationResult(iteration=2, converged=True, hits=hits)

    profile = run_jack._iteration_profile(result)

    assert profile["iteration"] == 2
    assert profile["converged"] is True
    assert profile["hits"]["total"] == 2
    assert profile["hits"]["included"] == 1
    assert "target_a" not in str(profile)


def test_aggregate_profiles_computes_phase_dominance_and_stats():
    profiles = [
        {
            "query_idx": 1,
            "query_id": "q1",
            "status": "ok",
            "cost_s": 10.0,
            "iterations": 3,
            "nhits": 20,
            "phase_wall_s": {"load_targets": 2.0, "jackhmmer_iterations": 7.0, "final_hmmsearch": 1.0},
            "phase_wall_fraction": {"jackhmmer_iterations": 0.7},
            "dominant_phase": "jackhmmer_iterations",
            "iteration_results": [{"iteration": 1, "hits": {"total": 10, "included": 4}}],
            "final_hits": {"total": 20},
        },
        {
            "query_idx": 2,
            "query_id": "q2",
            "status": "ok",
            "cost_s": 5.0,
            "iterations": 2,
            "nhits": 10,
            "phase_wall_s": {"load_targets": 1.0, "jackhmmer_iterations": 3.0, "final_hmmsearch": 1.0},
            "phase_wall_fraction": {"jackhmmer_iterations": 0.6},
            "dominant_phase": "jackhmmer_iterations",
            "iteration_results": [{"iteration": 1, "hits": {"total": 8, "included": 3}}],
            "final_hits": {"total": 10},
        },
    ]

    aggregate = run_jack._aggregate_profiles(
        profiles,
        stage="smoke",
        status="ok",
        cpus=2,
        max_iter=3,
        query_count=2,
        cost_s=15.0,
    )

    assert aggregate["schema_version"] == "pyhmmer_eval_profile.v1"
    assert aggregate["dominant_phase"] == "jackhmmer_iterations"
    assert aggregate["dominant_algorithm_phase"] == "jackhmmer_iterations"
    assert aggregate["phase_wall_s"]["jackhmmer_iterations"] == 10.0
    assert aggregate["phase_wall_fraction"]["jackhmmer_iterations"] == round(10.0 / 15.0, 6)
    assert aggregate["query_cost_stats"]["max"] == 10.0
    assert aggregate["iteration_stats"]["mean"] == 2.5
    assert aggregate["iteration_hit_stats"]["included"]["max"] == 4.0
    assert aggregate["top_slowest_queries"][0]["query_id"] == "q1"


def test_run_variant_records_phase_and_hit_count_profile(
    tmp_path: Path, monkeypatch,
):
    # pyhmmer is an OPTIONAL dependency: the default evaluation path shells out to
    # a compiled HMMER CLI and never imports it. This is the only test in the suite
    # that needs the library itself, so skip rather than fail when it is absent.
    # Enable with:  pip install pyhmmer
    pytest.importorskip("pyhmmer", reason="optional dependency 'pyhmmer' is not installed")

    import pyhmmer.easel
    import pyhmmer.hmmer
    import pyhmmer.plan7

    query = tmp_path / "query.fasta"
    uniref = tmp_path / "uniref.fasta"
    astral = tmp_path / "astral.fasta"
    scop = tmp_path / "scop.txt"
    for path in (query, uniref, astral):
        path.write_text("placeholder\n", encoding="utf-8")
    scop.write_text(
        "dquery a.1.1.1\n"
        "dfinal_secret a.1.1.2\n",
        encoding="utf-8",
    )

    class FakeSequence:
        def __init__(self, name):
            self.name = name

    class FakeAlphabet:
        @staticmethod
        def amino():
            return object()

    class FakeSequenceFile:
        def __init__(self, path, **kwargs):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            if self.path == query:
                return iter([FakeSequence(b"dquery")])
            return iter([FakeSequence(b"db_sequence")])

    class FakeBlock:
        def __init__(self, alphabet, sequences):
            self.sequences = list(sequences)

    class FakeBuilder:
        def __init__(self, *args, **kwargs):
            pass

    class FakeBackground:
        def __init__(self, *args, **kwargs):
            pass

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def search_hmm(self, hmm, targets):
            return FakeTopHits([FakeHit("dfinal_secret", included=True, reported=True)])

    iteration_hits = FakeTopHits([
        FakeHit("iteration_secret", included=True, reported=True, new=True),
    ])

    def fake_jackhmmer(*args, **kwargs):
        return [FakeIterationResult(
            iteration=1,
            converged=True,
            hits=iteration_hits,
            hmm=object(),
        )]

    monkeypatch.setattr(pyhmmer.easel, "Alphabet", FakeAlphabet)
    monkeypatch.setattr(pyhmmer.easel, "SequenceFile", FakeSequenceFile)
    monkeypatch.setattr(pyhmmer.easel, "DigitalSequenceBlock", FakeBlock)
    monkeypatch.setattr(pyhmmer.plan7, "Builder", FakeBuilder)
    monkeypatch.setattr(pyhmmer.plan7, "Background", FakeBackground)
    monkeypatch.setattr(pyhmmer.plan7, "Pipeline", FakePipeline)
    monkeypatch.setattr(pyhmmer.hmmer, "jackhmmer", fake_jackhmmer)

    class FakeConfig:
        specs = {}

        @staticmethod
        def builder_kwargs():
            return {}

        @staticmethod
        def pipeline_kwargs():
            return {}

        @staticmethod
        def select_hits():
            return None

        @staticmethod
        def should_stop_factory():
            return None

    result = run_jack.run_variant(
        str(query),
        FakeConfig(),
        max_iter=1,
        astral_db=astral,
        uniref_db=uniref,
        scop_txt=scop,
    )

    assert result.status == "ok"
    assert result.cost_s >= 0.0
    profile = result.meta["profile"]
    assert profile["schema_version"] == "pyhmmer_query_profile.v1"
    assert set(profile["phase_wall_s"]) == {
        "pipeline_setup", "load_targets", "jackhmmer_iterations",
        "load_astral", "final_hmmsearch",
    }
    assert profile["iteration_results"][0]["hits"]["included"] == 1
    assert profile["final_hits"]["total"] == 1
    assert "iteration_secret" not in str(profile)
    assert "final_secret" not in str(profile)


def test_run_all_queries_aggregates_query_profiles(monkeypatch):
    profiles = [
        {
            "schema_version": "pyhmmer_query_profile.v1",
            "query_id": "q1",
            "status": "ok",
            "cost_s": 3.0,
            "iterations": 1,
            "phase_wall_s": {"jackhmmer_iterations": 2.0},
            "iteration_results": [{"hits": {"included": 1}}],
        },
        {
            "schema_version": "pyhmmer_query_profile.v1",
            "query_id": "q2",
            "status": "ok",
            "cost_s": 2.0,
            "iterations": 1,
            "phase_wall_s": {"jackhmmer_iterations": 1.0},
            "iteration_results": [{"hits": {"included": 2}}],
        },
    ]
    monkeypatch.setattr(run_jack, "query_fastas", lambda data_dir: ["q1.fa", "q2.fa"])

    def fake_run_variant(path, cfg, **kwargs):
        index = 0 if path == "q1.fa" else 1
        return run_jack.EvalResult(
            stage="smoke",
            status="ok",
            tp=1,
            fp=0,
            cost_s=profiles[index]["cost_s"],
            n_seqs=1,
            iterations=1,
            meta={"profile": profiles[index]},
        )

    monkeypatch.setattr(run_jack, "run_variant", fake_run_variant)
    result = run_jack.run_all_queries(object(), data_dir=Path("unused"), max_iter=1)

    aggregate = result.meta["profile"]
    assert aggregate["schema_version"] == "pyhmmer_eval_profile.v1"
    assert aggregate["profiled_query_count"] == 2
    assert aggregate["phase_wall_s"]["jackhmmer_iterations"] == 3.0
    assert aggregate["top_slowest_queries"][0]["query_idx"] == 1
