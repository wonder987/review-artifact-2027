import importlib.util
import json
import os
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "build_gold138_references.py")


def load_module():
    spec = importlib.util.spec_from_file_location("build_gold138", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(
    signal_hash="aaa",
    time_hash="bbb",
    derived_hash=None,
    task_id="task",
    repeat_id=1,
):
    value = {
        "task_id": task_id,
        "repeat_id": repeat_id,
        "status": "success",
        "resource_decisions": [{"allow_full_read": True}],
        "targets": [
            {
                "target": {"shot": 1, "node": "\\IP"},
                "expressions": {"signal": "\\IP", "time": "dim_of(\\IP)"},
                "window": None,
                "signal": {"sha256": signal_hash, "shape": [10], "size": 10},
                "time_axis": {"sha256": time_hash, "shape": [10], "size": 10},
                "validation": {
                    "non_empty": True,
                    "same_size": True,
                    "time_strictly_increasing": True,
                },
            }
        ],
    }
    if derived_hash is not None:
        value["derived"] = {
            "common_time_axis": {"sha256": derived_hash},
            "aligned_first": {"sha256": "first"},
            "aligned_second": {"sha256": "second"},
        }
    return value


class GoldReferenceBatchTests(unittest.TestCase):
    def setUp(self):
        self.batch = load_module()

    def test_equal_repeats_are_stable(self):
        status = self.batch.compare_repeats(
            reference(repeat_id=1), reference(repeat_id=2)
        )
        self.assertTrue(status["stable"])
        self.assertEqual(status["differences"], [])

    def test_nine_task_pilot_covers_all_levels_and_subtypes(self):
        tasks = []
        layout = (
            ("L1", "simple_retrieval", 4),
            ("L2", "exact_window", 2),
            ("L2", "sampling_constrained", 2),
            ("L2", "subtree_discovery", 2),
            ("L3", "adaptive_access", 2),
            ("L3", "cross_shot_comparison", 2),
            ("L3", "same_shot_alignment", 2),
        )
        for level, subtype, count in layout:
            for position in range(count):
                tasks.append(
                    {
                        "id": "{}_{}_{}".format(level, subtype, position),
                        "level": level,
                        "subtype": subtype,
                    }
                )
        selected = self.batch.balanced_limit(tasks, 9)
        counts = __import__("collections").Counter(
            task["level"] for task in selected
        )
        self.assertEqual({"L1": 3, "L2": 3, "L3": 3}, dict(counts))
        self.assertEqual(
            set(subtype for _, subtype, _ in layout),
            set(task["subtype"] for task in selected),
        )

    def test_hash_difference_marks_reference_unstable(self):
        status = self.batch.compare_repeats(reference("aaa"), reference("zzz"))
        self.assertFalse(status["stable"])
        self.assertIn("target_0.signal.sha256", status["differences"])

    def test_task_or_target_difference_marks_reference_unstable(self):
        changed_task = self.batch.compare_repeats(
            reference(task_id="task"), reference(task_id="other", repeat_id=2)
        )
        self.assertIn("task_id", changed_task["differences"])
        second = reference(repeat_id=2)
        second["targets"][0]["target"]["node"] = "\\OTHER"
        changed_target = self.batch.compare_repeats(reference(), second)
        self.assertIn("target_0.identity", changed_target["differences"])

    def test_failed_targets_do_not_crash_comparison(self):
        first = {"task_id": "task", "status": "error", "targets": [None]}
        second = {"task_id": "task", "status": "error", "targets": [None]}
        status = self.batch.compare_repeats(first, second)
        self.assertFalse(status["stable"])

    def test_resume_skips_completed_stable_task(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "task.repeat1.json")
            second = os.path.join(directory, "task.repeat2.json")
            for repeat_id, path in enumerate((first, second), 1):
                with open(path, "w") as handle:
                    json.dump(reference(repeat_id=repeat_id), handle)
            self.assertTrue(
                self.batch.can_resume_task("task", directory, repeats=2)
            )

    def test_timeout_writes_synthetic_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "timeout.json")

            def timed_out(command, timeout):
                return 124

            status = self.batch.run_reference_command(
                ["python", "runner.py"], output, 90, caller=timed_out
            )
            self.assertEqual(status, 124)
            with open(output, "r") as handle:
                data = json.load(handle)
            self.assertEqual(data["status"], "timeout")
            self.assertEqual(data["exit_status"], 124)

    def test_timeout_overwrites_stale_success_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "timeout.json")
            with open(output, "w") as handle:
                json.dump(reference(), handle)

            self.batch.run_reference_command(
                ["python", "runner.py"], output, 90, caller=lambda c, t: 124
            )
            with open(output, "r") as handle:
                data = json.load(handle)
            self.assertEqual("timeout", data["status"])

    def test_completed_repeat_requires_matching_task_and_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "repeat.json")
            with open(path, "w") as handle:
                json.dump(reference(task_id="task", repeat_id=1), handle)
            self.assertTrue(
                self.batch.is_completed_repeat(path, "task", repeat_id=1)
            )
            self.assertFalse(
                self.batch.is_completed_repeat(path, "other", repeat_id=1)
            )
            self.assertFalse(
                self.batch.is_completed_repeat(path, "task", repeat_id=2)
            )

    def test_resume_keeps_one_completed_repeat_and_runs_only_missing_one(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = os.path.join(directory, "manifest.json")
            results_dir = os.path.join(directory, "results")
            os.makedirs(results_dir)
            with open(manifest_path, "w") as handle:
                json.dump({"tasks": [{"id": "task", "level": "L1"}]}, handle)
            first_path = self.batch.repeat_path("task", results_dir, 1)
            with open(first_path, "w") as handle:
                json.dump(reference(repeat_id=1), handle)

            calls = []
            original = self.batch.run_reference_command

            def fake_run(command, output_path, timeout_seconds):
                calls.append(output_path)
                with open(output_path, "w") as handle:
                    json.dump(reference(repeat_id=2), handle)
                return 0

            self.batch.run_reference_command = fake_run
            try:
                result = self.batch.build_references(
                    manifest_path, results_dir, resume=True
                )
            finally:
                self.batch.run_reference_command = original
            self.assertEqual(
                [self.batch.repeat_path("task", results_dir, 2)], calls
            )
            self.assertTrue(result["tasks"]["task"]["stability"]["stable"])
            self.assertTrue(
                all(
                    not os.path.isabs(path)
                    for path in result["tasks"]["task"]["repeat_files"]
                )
            )


if __name__ == "__main__":
    unittest.main()
