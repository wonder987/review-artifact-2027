import hashlib
import json
import os
import tempfile
import unittest

from scripts.evaluate_generated_batch import _group_summary, evaluate_batch

from tests.test_semantic_evaluator import gold, success_report


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


class BatchEvaluatorTests(unittest.TestCase):
    def test_generation_failure_counts_in_model_evaluable_denominator(self):
        records = [{
            "category": "generation_failure",
            "code": "generation_timeout",
            "model_alias": "deepseek",
            "level": "L1",
            "subtype": "simple_retrieval",
            "semantic_success": False,
        }]

        summary = _group_summary(records)

        self.assertEqual(1, summary["generation_failure"])
        self.assertEqual(1, summary["model_evaluable"])
        self.assertEqual(0.0, summary["semantic_accuracy"])

    def build_fixture(self, root):
        execution_root = os.path.join(root, "executions")
        gold_root = os.path.join(root, "gold")
        task_manifest = os.path.join(root, "manifest.json")
        protocol = os.path.join(root, "protocol.json")
        package_manifest = os.path.join(root, "package.json")
        task = {
            "id": "task",
            "level": "L1",
            "split": "pilot",
            "subtype": "simple_retrieval",
        }
        write_json(task_manifest, {"tasks": [task]})
        write_json(protocol, {"prompt_version": "primary-zero-shot-v2"})
        write_json(package_manifest, {"schema_version": "2.0"})
        write_json(os.path.join(gold_root, "task.repeat1.json"), gold())
        write_json(os.path.join(gold_root, "task.repeat2.json"), gold())
        write_json(os.path.join(gold_root, "reference_index.json"), {
            "tasks": {
                "task": {
                    "repeat_files": ["task.repeat1.json", "task.repeat2.json"],
                    "stability": {"stable": True, "differences": []},
                }
            }
        })
        write_json(
            os.path.join(execution_root, "model-a", "task", "execution.json"),
            {
                "model_alias": "model-a",
                "task_id": "task",
                "status": "execution_success",
                "exit_status": 0,
                "stdout_json_valid": True,
                "reported_result": success_report(),
            },
        )
        return {
            "execution_root": execution_root,
            "gold_root": gold_root,
            "reference_index": os.path.join(gold_root, "reference_index.json"),
            "manifest": task_manifest,
            "protocol": protocol,
            "package_manifest": package_manifest,
        }

    def run_fixture(self, fixture, output):
        return evaluate_batch(
            execution_dir=fixture["execution_root"],
            gold_dir=fixture["gold_root"],
            reference_index_path=fixture["reference_index"],
            manifest_path=fixture["manifest"],
            output_dir=output,
            models=["model-a", "model-b"],
            task_ids=["task"],
            protocol_path=fixture["protocol"],
            package_manifest_path=fixture["package_manifest"],
        )

    def test_evaluates_success_and_missing_execution_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            output = os.path.join(root, "evaluation")
            summary = self.run_fixture(fixture, output)
            with open(os.path.join(output, "evaluation.jsonl"), "r") as handle:
                records = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(2, summary["total"])
        self.assertEqual(1, summary["model_evaluable"])
        self.assertEqual(1, summary["semantic_success"])
        self.assertEqual(1.0, summary["semantic_accuracy"])
        self.assertEqual(1, summary["by_model"]["model-a"]["model_evaluable"])
        self.assertEqual(1.0, summary["by_model"]["model-a"]["semantic_accuracy"])
        self.assertEqual(0, summary["by_model"]["model-b"]["model_evaluable"])
        self.assertIsNone(summary["by_model"]["model-b"]["semantic_accuracy"])
        self.assertEqual(
            [("model-a", "success"), ("model-b", "infrastructure_failure")],
            [(r["model_alias"], r["category"]) for r in records],
        )
        self.assertEqual("missing_execution", records[1]["code"])

    def test_rejects_unstable_or_missing_gold_as_infrastructure(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            with open(fixture["reference_index"], "r") as handle:
                index = json.load(handle)
            index["tasks"]["task"]["stability"]["stable"] = False
            write_json(fixture["reference_index"], index)
            output = os.path.join(root, "evaluation")
            summary = evaluate_batch(
                fixture["execution_root"], fixture["gold_root"],
                fixture["reference_index"], fixture["manifest"], output,
                ["model-a"], ["task"], fixture["protocol"],
                fixture["package_manifest"],
            )
            with open(os.path.join(output, "evaluation.jsonl"), "r") as handle:
                record = json.loads(handle.readline())

        self.assertEqual(1, summary["infrastructure_failure"])
        self.assertEqual("unstable_gold", record["code"])

    def test_outputs_are_deterministic_and_manifest_is_hashed(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            first = os.path.join(root, "first")
            second = os.path.join(root, "second")
            self.run_fixture(fixture, first)
            self.run_fixture(fixture, second)

            for name in ("evaluation.jsonl", "evaluation_summary.json"):
                self.assertEqual(
                    file_hash(os.path.join(first, name)),
                    file_hash(os.path.join(second, name)),
                )
            with open(
                os.path.join(first, "evaluation_manifest.json"), "r"
            ) as handle:
                manifest = json.load(handle)

        self.assertEqual("semantic-evaluator-v2", manifest["evaluator_version"])
        for field in (
            "task_manifest_sha256", "protocol_sha256",
            "package_manifest_sha256", "gold_reference_index_sha256",
            "evaluation_jsonl_sha256",
        ):
            self.assertEqual(64, len(manifest[field]))

    def test_diagnostic_mode_is_recorded_and_emits_normalizations(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            execution_path = os.path.join(
                fixture["execution_root"], "model-a", "task", "execution.json"
            )
            with open(execution_path, "r") as handle:
                execution_value = json.load(handle)
            item = execution_value["reported_result"]["targets"][0]
            item["identity"] = item["target"]
            item["target"] = "\\IP"
            write_json(execution_path, execution_value)
            output = os.path.join(root, "diagnostic")
            summary = evaluate_batch(
                execution_dir=fixture["execution_root"],
                gold_dir=fixture["gold_root"],
                reference_index_path=fixture["reference_index"],
                manifest_path=fixture["manifest"],
                output_dir=output,
                models=["model-a"],
                task_ids=["task"],
                protocol_path=fixture["protocol"],
                package_manifest_path=fixture["package_manifest"],
                evaluation_mode="diagnostic-v2",
            )
            with open(os.path.join(output, "evaluation.jsonl"), "r") as handle:
                record = json.loads(handle.readline())
            with open(
                os.path.join(output, "evaluation_manifest.json"), "r"
            ) as handle:
                manifest = json.load(handle)

        self.assertEqual(1, summary["semantic_success"])
        self.assertEqual("diagnostic-v2", manifest["evaluation_mode"])
        self.assertEqual(
            "identity_to_target", record["normalizations"][0]["normalization"]
        )

    def test_rejects_duplicate_model_or_task_scope(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            with self.assertRaises(ValueError):
                evaluate_batch(
                    fixture["execution_root"], fixture["gold_root"],
                    fixture["reference_index"], fixture["manifest"],
                    os.path.join(root, "duplicate-model"),
                    ["model-a", "model-a"], ["task"],
                    fixture["protocol"], fixture["package_manifest"],
                )
            with self.assertRaises(ValueError):
                evaluate_batch(
                    fixture["execution_root"], fixture["gold_root"],
                    fixture["reference_index"], fixture["manifest"],
                    os.path.join(root, "duplicate-task"),
                    ["model-a"], ["task", "task"],
                    fixture["protocol"], fixture["package_manifest"],
                )


if __name__ == "__main__":
    unittest.main()
