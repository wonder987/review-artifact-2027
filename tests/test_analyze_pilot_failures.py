import json
import os
import tempfile
import unittest

from scripts.analyze_pilot_failures import analyze_pilot


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAXONOMY = os.path.join(ROOT, "experiments", "config", "failure_taxonomy_v0.json")


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path, values):
    with open(path, "w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


class PilotFailureAnalysisTests(unittest.TestCase):
    def build_fixture(self, root):
        execution_dir = os.path.join(root, "execution")
        evaluation_path = os.path.join(root, "evaluation.jsonl")
        manifest_path = os.path.join(root, "tasks.json")
        output_dir = os.path.join(root, "analysis")
        tasks = [
            {"id": "task-1", "level": "L1", "subtype": "simple_retrieval"},
            {"id": "task-2", "level": "L2", "subtype": "exact_window"},
            {"id": "task-3", "level": "L3", "subtype": "cross_shot_comparison"},
        ]
        write_json(manifest_path, {"tasks": tasks})
        executions = [
            {
                "model_alias": "model-a",
                "task_id": "task-1",
                "status": "generation_failure",
                "failure_code": "generation_timeout",
                "error_type": "ClientError",
                "error_message": "generation request timed out after 600 seconds",
            },
            {
                "model_alias": "model-a",
                "task_id": "task-2",
                "status": "policy_rejected",
                "policy_violations": ["syntax_error:4:invalid syntax"],
            },
            {
                "model_alias": "model-a",
                "task_id": "task-3",
                "status": "execution_success",
                "stdout_json_valid": True,
                "reported_result": {"status": "success"},
            },
        ]
        for execution in executions:
            write_json(
                os.path.join(
                    execution_dir,
                    execution["model_alias"],
                    execution["task_id"],
                    "execution.json",
                ),
                execution,
            )
        evaluations = [
            {
                "model_alias": "model-a",
                "task_id": "task-1",
                "category": "generation_failure",
                "code": "generation_timeout",
                "semantic_success": False,
                "normalizations": [],
            },
            {
                "model_alias": "model-a",
                "task_id": "task-2",
                "category": "execution_failure",
                "code": "policy_rejected",
                "semantic_success": False,
                "normalizations": [],
            },
            {
                "model_alias": "model-a",
                "task_id": "task-3",
                "category": "success",
                "code": "semantic_success",
                "semantic_success": True,
                "normalizations": [
                    {"path": "targets[0].target", "normalization": "identity_to_target"}
                ],
            },
        ]
        write_jsonl(evaluation_path, evaluations)
        return {
            "execution_dir": execution_dir,
            "evaluation_path": evaluation_path,
            "manifest_path": manifest_path,
            "output_dir": output_dir,
            "evaluations": evaluations,
        }

    def run_fixture(self, fixture):
        return analyze_pilot(
            fixture["execution_dir"],
            fixture["evaluation_path"],
            fixture["manifest_path"],
            TAXONOMY,
            fixture["output_dir"],
        )

    def test_mixed_stages_have_exact_denominators(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            summary = self.run_fixture(fixture)

        self.assertEqual(3, summary["total"])
        self.assertEqual(2, summary["generation_completed"])
        self.assertEqual(1, summary["syntax_valid"])
        self.assertEqual(1, summary["policy_safe"])
        self.assertEqual(1, summary["execution_success"])
        self.assertEqual(1, summary["protocol_valid"])
        self.assertEqual(1, summary["semantic_success"])

    def test_grouping_root_causes_and_normalizations_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            summary = self.run_fixture(fixture)
            with open(
                os.path.join(fixture["output_dir"], "failure_records.jsonl"), "r"
            ) as handle:
                records = [json.loads(line) for line in handle if line.strip()]

        self.assertEqual(3, summary["by_model"]["model-a"]["count"])
        self.assertEqual(1, summary["by_level"]["L1"]["count"])
        self.assertEqual(
            1, summary["by_task_subtype"]["cross_shot_comparison"]["count"]
        )
        self.assertEqual(2, summary["by_major_category"]["code_generation"]["count"])
        self.assertEqual(1, summary["by_major_category"]["success"]["count"])
        self.assertEqual(1, summary["by_failure_subtype"]["generation_timeout"]["count"])
        self.assertEqual(1, summary["by_failure_subtype"]["syntax_error"]["count"])
        self.assertEqual(
            "identity_to_target", records[2]["normalizations"][0]["normalization"]
        )

    def test_analysis_manifest_hashes_all_inputs_and_output(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            self.run_fixture(fixture)
            with open(
                os.path.join(fixture["output_dir"], "analysis_manifest.json"), "r"
            ) as handle:
                manifest = json.load(handle)

        for name in (
            "execution_directory_sha256",
            "evaluation_jsonl_sha256",
            "task_manifest_sha256",
            "taxonomy_sha256",
            "failure_records_sha256",
        ):
            self.assertEqual(64, len(manifest[name]))
        self.assertEqual("pilot-failure-taxonomy-v0", manifest["taxonomy_version"])
        self.assertEqual(3, manifest["record_count"])

    def test_duplicate_evaluation_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            write_jsonl(
                fixture["evaluation_path"],
                fixture["evaluations"] + [fixture["evaluations"][0]],
            )
            with self.assertRaises(ValueError):
                self.run_fixture(fixture)

    def test_missing_execution_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            os.remove(os.path.join(
                fixture["execution_dir"], "model-a", "task-2", "execution.json"
            ))
            with self.assertRaises(ValueError):
                self.run_fixture(fixture)

    def test_infrastructure_record_is_excluded_from_model_denominator(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)
            execution_path = os.path.join(
                fixture["execution_dir"], "model-a", "task-1", "execution.json"
            )
            execution = {
                "model_alias": "model-a",
                "task_id": "task-1",
                "status": "generation_failure",
                "failure_code": "provider_outage",
                "error_type": "ClientError",
                "error_message": "HTTP 503 UNAVAILABLE due to high demand",
            }
            write_json(execution_path, execution)
            summary = self.run_fixture(fixture)

        self.assertEqual(3, summary["total"])
        self.assertEqual(2, summary["model_evaluable"])
        self.assertEqual(1, summary["infrastructure_failure"])
        self.assertEqual(2, summary["generation_completed"])
        self.assertEqual(1.0, summary["generation_completed_rate"])

    def test_analyzer_uses_stdout_and_program_as_diagnostic_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = self.build_fixture(root)

            first_dir = os.path.join(
                fixture["execution_dir"], "model-a", "task-1"
            )
            write_json(
                os.path.join(first_dir, "execution.json"),
                {
                    "model_alias": "model-a",
                    "task_id": "task-1",
                    "status": "program_failure",
                    "reported_result": None,
                },
            )
            with open(os.path.join(first_dir, "stdout.log"), "w") as handle:
                handle.write(
                    '{"status":"failure","error_type":"RuntimeError",'
                    '"error_message":"MDS operation failed"}\n'
                    "Insufficient virtual memory\n"
                )

            second_dir = os.path.join(
                fixture["execution_dir"], "model-a", "task-2"
            )
            write_json(
                os.path.join(second_dir, "execution.json"),
                {
                    "model_alias": "model-a",
                    "task_id": "task-2",
                    "status": "program_failure",
                    "reported_result": {
                        "status": "failure",
                        "error_type": "RuntimeError",
                        "error_message": "Could not locate node in any subtree",
                    },
                },
            )
            with open(os.path.join(second_dir, "program.py"), "w") as handle:
                handle.write(
                    "conn = MDSplus.Connection(SERVER)\n"
                    "conn.openTree(TREE, SHOT)\n"
                    "node = conn.getNode(candidate)\n"
                )

            fixture["evaluations"][0].update({
                "category": "execution_failure",
                "code": "runtime_error",
            })
            write_jsonl(fixture["evaluation_path"], fixture["evaluations"])
            summary = self.run_fixture(fixture)

        self.assertEqual(
            1, summary["by_failure_subtype"]["memory_exhaustion"]["count"]
        )
        self.assertEqual(
            1, summary["by_failure_subtype"]["hallucinated_api"]["count"]
        )


if __name__ == "__main__":
    unittest.main()
