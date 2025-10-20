import json
from pathlib import Path

from tools.pretrain_utils import ExperimentTracker


def test_experiment_tracker_logs_metrics_and_artifacts(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path, history_filename="metrics.jsonl")

    tracker.log_params({"foo": "bar"})
    tracker.log_metrics(1, {"loss": 0.5, "epoch": 1}, context="train_step")

    artifact_path = tmp_path / "model.pt"
    artifact_path.write_bytes(b"weights")

    tracker.register_artifact(artifact_path, name="checkpoint_last", metadata={"epoch": 1})
    tracker.register_artifact(artifact_path, name="checkpoint_last", metadata={"epoch": 2})

    params = json.loads((tmp_path / "params.json").read_text(encoding="utf8"))
    assert params["foo"] == "bar"

    history_lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf8").strip().splitlines()
    assert len(history_lines) == 1
    history_entry = json.loads(history_lines[0])
    assert history_entry["step"] == 1
    assert history_entry["context"] == "train_step"
    assert history_entry["loss"] == 0.5

    artifacts = json.loads((tmp_path / "artifacts.json").read_text(encoding="utf8"))
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "checkpoint_last"
    assert artifacts[0]["metadata"]["epoch"] == 2
