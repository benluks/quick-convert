import json

from quick_convert.pipelines.benchmarking.calibration import calibrate


def test_calibration_continues_after_oom(tmp_path):
    output = tmp_path / "calibration"

    def trial(command, directory):
        directory.mkdir()
        if directory.name == "3s_b64":
            return {"status": "oom", "returncode": 1}
        assert json.loads((output / "results.json").read_text())[0]["status"] == "oom"
        assert "trainer.trainer_kwargs.batch_size=128" in command
        return {"status": "completed", "metrics": [{"audio_seconds_per_second": 300}]}

    results = calibrate(output, [3], [192, 384], [], run_trial_fn=trial)
    assert [result["status"] for result in results] == ["oom", "completed"]
    assert results[1]["quadratic_cost"] == 1152
    assert json.loads((output / "results.json").read_text()) == results
