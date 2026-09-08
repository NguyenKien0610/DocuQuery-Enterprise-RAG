import json

import numpy as np
import pytest

from scripts import compare_embeddings as comparison
from scripts.compare_embeddings import (
    _encode,
    _retrieval_rows,
    document_token_lengths,
    run_experiment,
    validate_question_sets,
)


class FakeEncoder:
    revision = "fake-revision"
    max_seq_length = 4

    class _Tokenizer:
        def __call__(self, texts, **_kwargs):
            return {"input_ids": [[0] * (len(text.split()) + 2) for text in texts]}

    tokenizer = _Tokenizer()

    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [1.0, 0.0, 0.0]
                if "alpha" in lowered
                else [0.0, 1.0, 0.0]
                if "beta" in lowered
                else [0.0, 0.0, 1.0]
            )
        return np.asarray(vectors, dtype=float)


def write_fixture(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha policy", encoding="utf-8")
    (corpus / "b.txt").write_text("beta policy", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            [
                {"id": "c1", "language": "vi", "query": "alpha", "sources": ["a.txt"]},
                {"id": "c2", "language": "en", "query": "unknown", "sources": []},
            ]
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {"id": "t1", "language": "vi", "query": "beta", "sources": ["b.txt"]},
                {"id": "t2", "language": "en", "query": "unknown test", "sources": []},
            ]
        ),
        encoding="utf-8",
    )
    return corpus, calibration, questions


def test_run_experiment_uses_fake_models_and_freezes_calibrated_threshold(tmp_path):
    corpus, calibration, questions = write_fixture(tmp_path)
    loaded = []

    def load_model(model_id):
        loaded.append(model_id)
        return FakeEncoder()

    report = run_experiment(
        corpus=corpus,
        calibration=calibration,
        questions=questions,
        model_ids=["baseline", "candidate"],
        k=1,
        load_model=load_model,
    )

    assert loaded == ["baseline", "candidate"]
    assert report["document_level"] is True
    assert report["fingerprints"]["corpus_sha256"]
    assert len(report["models"]) == 2
    result = report["models"][0]
    assert result["model_id"] == "baseline"
    assert result["revision"] == "fake-revision"
    assert result["max_seq_length"] == 4
    assert result["calibration"]["threshold"] == 0.6
    assert result["calibration"]["selection_score"] == 1.0
    assert result["test"]["rows"][0]["retrieved_filenames"] == ["b.txt"]
    assert result["test"]["rows"][0]["retrieved"][0]["score"] == 1.0
    assert result["test"]["rows"][1]["retrieved_filenames"] == []
    assert result["test"]["by_language"]["vi"]["document_recall_at_k"] == 1.0


def test_validate_question_sets_rejects_overlapping_query_strings():
    calibration = [{"query": "  Same   query ", "sources": []}]
    questions = [{"query": "same query", "sources": []}]

    with pytest.raises(ValueError, match="disjoint"):
        validate_question_sets(calibration, questions)


def test_threshold_filters_each_document_score_not_only_the_best():
    rows = _retrieval_rows(
        [{"id": "q", "language": "vi", "query": "q", "sources": ["right.txt"]}],
        np.array([[1.0, 0.0]]),
        np.array([[0.9, 0.0], [0.1, 0.0]]),
        ["wrong.txt", "right.txt"],
        k=2,
        threshold=0.5,
    )

    assert rows[0]["retrieved_filenames"] == ["wrong.txt"]
    assert rows[0]["recall"] == 0.0


def test_document_window_validation_rejects_untruncated_long_documents():
    encoder = FakeEncoder()
    encoder.max_seq_length = 3

    with pytest.raises(ValueError, match="a.txt.*4 tokens.*3"):
        document_token_lengths(encoder, ["alpha policy"], ["a.txt"])


@pytest.mark.parametrize("values", [[[float("nan"), 1]], [[float("inf"), 1]], [[0, 0]], [[1, 0], [0, 1]]])
def test_encode_rejects_invalid_vectors_and_row_counts(values):
    class BrokenEncoder:
        def encode(self, texts, **kwargs):
            return values

    with pytest.raises(ValueError):
        _encode(BrokenEncoder(), ["query"])


def test_calibration_only_records_curve_and_e5_prefixes(tmp_path):
    corpus, calibration, _ = write_fixture(tmp_path)
    seen = []

    class RecordingEncoder(FakeEncoder):
        max_seq_length = 8

        def encode(self, texts, **kwargs):
            seen.extend(texts)
            return super().encode(texts, **kwargs)

    report = run_experiment(
        corpus=corpus, calibration=calibration, questions=None,
        model_ids=["intfloat/multilingual-e5-small"], load_model=lambda _: RecordingEncoder(),
    )
    result = report["models"][0]
    assert result["test"] is None
    assert report["fingerprints"]["test_questions_sha256"] is None
    assert len(result["calibration"]["curve"]) == len(report["threshold_grid"])
    assert seen == ["passage: alpha policy", "passage: beta policy", "query: alpha", "query: unknown"]


def test_diagnostics_separate_ranking_from_threshold_rejection():
    rows = _retrieval_rows(
        [{"query": "q", "sources": ["right.txt"], "language": "vi"}],
        np.array([[1.0, 0.0]]), np.array([[0.9, 0.0], [0.1, 0.0]]),
        ["wrong.txt", "right.txt"], k=2, threshold=0.5,
    )
    assert rows[0]["ranking_recall"] == 1.0
    assert rows[0]["ranking_reciprocal_rank"] == 0.5
    assert rows[0]["error_type"] == "threshold_rejection"
    assert rows[0]["candidates"][1]["filename"] == "right.txt"
    missed = _retrieval_rows(
        [{"query": "q", "sources": ["right.txt"], "language": "vi"}],
        np.array([[1.0, 0.0]]), np.array([[0.9, 0.0], [0.1, 0.0]]),
        ["wrong.txt", "right.txt"], k=1, threshold=0.5,
    )
    assert missed[0]["error_type"] == "ranking_miss"


def frozen_fixture(tmp_path):
    corpus, calibration, questions = write_fixture(tmp_path)
    selection = run_experiment(
        corpus=corpus, calibration=calibration, model_ids=["baseline", "candidate"],
        k=1, load_model=lambda _: FakeEncoder(),
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    # Confirmation documents must not include the positive calibration document.
    (corpus / "a.txt").unlink()
    return corpus, questions, selection_path


def test_confirmation_pins_configuration_without_recalibrating(tmp_path, monkeypatch):
    corpus, questions, selection = frozen_fixture(tmp_path)
    loaded = []

    def load(model_id, revision):
        loaded.append((model_id, revision))
        return FakeEncoder()

    def forbidden(*args, **kwargs):
        pytest.fail("Confirmation must not select thresholds")

    monkeypatch.setattr(comparison, "select_threshold", forbidden)
    report = comparison.run_confirmation(
        corpus=corpus, questions=questions, selection_report=selection, load_model=load,
    )
    assert loaded == [("baseline", "fake-revision"), ("candidate", "fake-revision")]
    assert report["k"] == 1
    assert report["models"][0]["threshold"] == 0.6
    assert report["models"][0]["test"]["aggregate"]["document_recall_at_k"] == 1.0
    assert report["promotion_gate"]["passed"] is False  # Equal recall is not improvement.
    assert report["fingerprints"]["selection_report_sha256"]


@pytest.mark.parametrize("mutation", ["revision", "threshold", "test", "query_overlap", "document_overlap"])
def test_confirmation_rejects_unfrozen_or_leaking_inputs_before_loading(tmp_path, mutation):
    corpus, questions, selection = frozen_fixture(tmp_path)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    if mutation == "revision":
        payload["models"][0]["revision"] = None
    elif mutation == "threshold":
        payload["models"][0]["calibration"]["threshold"] = float("nan")
    elif mutation == "test":
        payload["models"][0]["test"] = {"rows": []}
    elif mutation == "query_overlap":
        rows = json.loads(questions.read_text(encoding="utf-8"))
        rows[0]["query"] = " ALPHA "
        questions.write_text(json.dumps(rows), encoding="utf-8")
    else:
        (corpus / "a.txt").write_text("alpha policy", encoding="utf-8")
    selection.write_text(json.dumps(payload), encoding="utf-8")

    def forbidden(*args):
        pytest.fail("Invalid input must be rejected before model loading")

    with pytest.raises(ValueError):
        comparison.run_confirmation(
            corpus=corpus, questions=questions, selection_report=selection, load_model=forbidden,
        )


def test_cli_rejects_output_alias_even_with_overwrite(tmp_path, monkeypatch):
    corpus, calibration, _ = write_fixture(tmp_path)
    before = calibration.read_bytes()
    monkeypatch.setattr("sys.argv", [
        "compare", "--corpus", str(corpus), "--calibration", str(calibration),
        "--output", str(calibration), "--overwrite",
    ])
    monkeypatch.setattr(comparison, "run_experiment", lambda **kwargs: {})
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 2
    assert calibration.read_bytes() == before


@pytest.mark.parametrize("regresses,has_negatives,expected", [(False, True, True), (True, True, False), (False, False, False)])
def test_confirmation_gate_requires_improvement_and_nonregression(tmp_path, regresses, has_negatives, expected):
    corpus, questions, selection = frozen_fixture(tmp_path)
    if not has_negatives:
        rows = json.loads(questions.read_text(encoding="utf-8"))
        questions.write_text(json.dumps(rows[:1]), encoding="utf-8")

    class Encoder(FakeEncoder):
        def __init__(self, model_id):
            self.model_id = model_id

        def encode(self, texts, **kwargs):
            vectors = super().encode(texts, **kwargs)
            for index, text in enumerate(texts):
                if self.model_id == "baseline" and text == "beta":
                    vectors[index] = [0, 0, 1]
                if self.model_id == "candidate" and regresses and text == "unknown test":
                    vectors[index] = [0, 1, 0]
            return vectors

    report = comparison.run_confirmation(
        corpus=corpus, questions=questions, selection_report=selection,
        load_model=lambda model, revision: Encoder(model),
    )
    gate = report["promotion_gate"]
    assert gate["baseline_vi_recall"] == 0
    assert gate["candidate_vi_recall"] == 1
    assert gate["evaluable"] is has_negatives
    assert gate["passed"] is expected


def test_confirmation_rejects_loaded_revision_drift(tmp_path):
    corpus, questions, selection = frozen_fixture(tmp_path)
    encoder = FakeEncoder()
    encoder.revision = "different-revision"
    with pytest.raises(ValueError, match="revision differs"):
        comparison.run_confirmation(
            corpus=corpus, questions=questions, selection_report=selection,
            load_model=lambda model, revision: encoder,
        )


@pytest.mark.parametrize("override", [["--k", "1"], ["--models", "other"], ["--threshold-grid", "0.9"]])
def test_confirmation_cli_forbids_overrides(tmp_path, monkeypatch, override):
    corpus, questions, selection = frozen_fixture(tmp_path)
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", [
        "compare", "--corpus", str(corpus), "--questions", str(questions),
        "--selection-report", str(selection), "--output", str(output), *override,
    ])
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 2
    assert not output.exists()


def test_confirmation_cli_writes_utf8_and_preserves_existing_output(tmp_path, monkeypatch):
    corpus, questions, selection = frozen_fixture(tmp_path)
    rows = json.loads(questions.read_text(encoding="utf-8"))
    rows[0]["query"] = "beta tiếng Việt"
    questions.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    encoder = FakeEncoder()
    encoder.max_seq_length = 8
    monkeypatch.setattr(comparison, "load_encoder", lambda model, revision: encoder)
    output = tmp_path / "result.json"
    monkeypatch.setattr("sys.argv", [
        "compare", "--corpus", str(corpus), "--questions", str(questions),
        "--selection-report", str(selection), "--output", str(output),
    ])
    comparison.main()
    before = output.read_bytes()
    assert "tiếng Việt" in before.decode("utf-8")
    assert json.loads(before)["models"][0]["test"]["rows"][0]["recall"] == 1
    with pytest.raises(SystemExit) as exc:
        comparison.main()
    assert exc.value.code == 2
    assert output.read_bytes() == before


def test_vietnamese_unicode_forms_cannot_bypass_query_disjointness():
    import unicodedata

    query = "Chính sách nghỉ phép"
    with pytest.raises(ValueError, match="disjoint"):
        validate_question_sets(
            [{"query": query}], [{"query": unicodedata.normalize("NFD", query)}],
        )


def test_document_window_guard_rejects_missing_tokenizer_rows():
    encoder = FakeEncoder()
    encoder.tokenizer = lambda texts, **kwargs: {"input_ids": []}
    with pytest.raises(ValueError, match="number of token"):
        document_token_lengths(encoder, ["alpha"])
