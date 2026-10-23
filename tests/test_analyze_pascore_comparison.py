import hashlib
import json
import os
import tempfile
import unittest
import zipfile

from scripts.analyze_pascore_comparison import (
    analyze_files,
    analyze_outcomes,
    cluster_bootstrap_difference,
    cluster_bootstrap_macro_level_difference,
    exact_mcnemar,
    load_diagnostics_zip,
    merge_round_outcomes,
)


class PascoreComparisonTests(unittest.TestCase):
    def write_diagnostics(self, path, evaluations, corrupt_hash=False):
        entries = []
        with zipfile.ZipFile(path, "w") as archive:
            for evaluation in evaluations:
                model = evaluation["model_alias"]
                task = evaluation["task_id"]
                member = "evaluation/{}/{}/evaluation.json".format(model, task)
                payload = json.dumps(evaluation, sort_keys=True).encode("utf-8")
                archive.writestr(member, payload)
                digest = hashlib.sha256(payload).hexdigest()
                entries.append({
                    "model_alias": model,
                    "task_id": task,
                    "files": {
                        "evaluation": {
                            "path": member,
                            "sha256": "0" * 64 if corrupt_hash else digest,
                        }
                    },
                })
            archive.writestr("manifest.json", json.dumps({
                "entry_count": len(entries),
                "entries": entries,
            }))

    def test_load_diagnostics_zip_returns_paired_evaluations(self):
        evaluations = [{
            "model_alias": "model-a",
            "task_id": "task-1",
            "level": "L1",
            "subtype": "simple_retrieval",
            "category": "success",
            "code": "semantic_success",
            "semantic_success": True,
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostics.zip")
            self.write_diagnostics(path, evaluations)

            loaded = load_diagnostics_zip(path)

        self.assertEqual(evaluations[0], loaded[("model-a", "task-1")])

    def test_load_diagnostics_zip_rejects_hash_mismatch(self):
        evaluations = [{
            "model_alias": "model-a",
            "task_id": "task-1",
            "semantic_success": False,
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostics.zip")
            self.write_diagnostics(path, evaluations, corrupt_hash=True)

            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_diagnostics_zip(path)

    def test_load_diagnostics_zip_supports_evaluation_jsonl_package(self):
        evaluation = {
            "model_alias": "model-a",
            "task_id": "task-1",
            "level": "L1",
            "subtype": "simple_retrieval",
            "category": "success",
            "code": "semantic_success",
            "semantic_success": True,
        }
        payload = (json.dumps(evaluation, sort_keys=True) + "\n").encode("utf-8")
        manifest = {
            "record_count": 1,
            "evaluation_jsonl_sha256": hashlib.sha256(payload).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "diagnostics.zip")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("evaluation/evaluation.jsonl", payload)
                archive.writestr(
                    "evaluation/evaluation_manifest.json",
                    json.dumps(manifest),
                )

            loaded = load_diagnostics_zip(path)

        self.assertEqual(evaluation, loaded[("model-a", "task-1")])

    def test_merge_rounds_replaces_only_failed_pairs(self):
        round_zero = {
            ("model-a", "task-1"): {
                "semantic_success": True,
                "level": "L1",
            },
            ("model-a", "task-2"): {
                "semantic_success": False,
                "level": "L2",
            },
        }
        round_one = {
            ("model-a", "task-2"): {
                "semantic_success": True,
                "level": "L2",
            },
        }

        final, success_round = merge_round_outcomes([round_zero, round_one])

        self.assertTrue(final[("model-a", "task-1")]["semantic_success"])
        self.assertTrue(final[("model-a", "task-2")]["semantic_success"])
        self.assertEqual(0, success_round[("model-a", "task-1")])
        self.assertEqual(1, success_round[("model-a", "task-2")])

    def test_merge_rounds_rejects_retry_of_successful_pair(self):
        pair = ("model-a", "task-1")
        round_zero = {pair: {"semantic_success": True, "level": "L1"}}
        round_one = {pair: {"semantic_success": False, "level": "L1"}}

        with self.assertRaisesRegex(ValueError, "already-successful"):
            merge_round_outcomes([round_zero, round_one])

    def test_exact_mcnemar_uses_two_sided_exact_binomial(self):
        direct = {("model", "task-{}".format(i)): False for i in range(10)}
        method = dict((key, True) for key in direct)

        result = exact_mcnemar(direct, method)

        self.assertEqual(0, result["direct_only"])
        self.assertEqual(10, result["method_only"])
        self.assertAlmostEqual(0.001953125, result["p_value"])

    def test_task_cluster_bootstrap_keeps_models_from_one_task_together(self):
        direct = {}
        method = {}
        for task_id, level in (("task-1", "L1"), ("task-2", "L2")):
            for model in ("model-a", "model-b"):
                key = (model, task_id)
                direct[key] = {"semantic_success": False, "level": level}
                method[key] = {"semantic_success": True, "level": level}

        micro = cluster_bootstrap_difference(
            direct, method, iterations=100, seed=7
        )
        macro = cluster_bootstrap_macro_level_difference(
            direct, method, iterations=100, seed=7
        )

        for result in (micro, macro):
            self.assertEqual(1.0, result["estimate"])
            self.assertEqual(1.0, result["confidence_low"])
            self.assertEqual(1.0, result["confidence_high"])
            self.assertEqual("task", result["resampling_unit"])

    def test_analyze_outcomes_reports_terminal_and_paired_counts(self):
        direct = {
            ("model-a", "task-1"): self.outcome(True, "L1", "simple"),
            ("model-b", "task-1"): self.outcome(False, "L1", "simple"),
            ("model-a", "task-2"): self.outcome(False, "L2", "exact"),
            ("model-b", "task-2"): self.outcome(False, "L2", "exact"),
        }
        round_zero = {
            key: self.outcome(False, value["level"], value["subtype"])
            for key, value in direct.items()
        }
        round_one = {
            ("model-a", "task-1"): self.outcome(True, "L1", "simple"),
            ("model-b", "task-1"): self.outcome(True, "L1", "simple"),
            ("model-a", "task-2"): self.outcome(True, "L2", "exact"),
        }

        result = analyze_outcomes(
            direct, [round_zero, round_one], iterations=100, seed=11
        )

        self.assertEqual(1, result["direct"]["semantic_success"])
        self.assertEqual(3, result["pascore"]["semantic_success"])
        self.assertEqual(3, result["rounds"][1]["newly_successful"])
        self.assertEqual(1, result["terminal_failure_count"])
        self.assertEqual(0, result["paired"]["mcnemar"]["direct_only"])
        self.assertEqual(2, result["paired"]["mcnemar"]["method_only"])

    def test_analyze_files_records_source_hashes_and_round_order(self):
        direct = [{
            "model_alias": "model-a",
            "task_id": "task-1",
            "level": "L1",
            "subtype": "simple",
            "category": "semantic_failure",
            "code": "mismatch",
            "semantic_success": False,
        }]
        round_zero = [dict(direct[0])]
        round_one = [dict(direct[0], category="success",
                          code="semantic_success", semantic_success=True)]
        with tempfile.TemporaryDirectory() as directory:
            direct_path = os.path.join(directory, "direct.zip")
            round_zero_path = os.path.join(directory, "round-0.zip")
            round_one_path = os.path.join(directory, "round-1.zip")
            self.write_diagnostics(direct_path, direct)
            self.write_diagnostics(round_zero_path, round_zero)
            self.write_diagnostics(round_one_path, round_one)

            result = analyze_files(
                direct_path,
                [round_zero_path, round_one_path],
                iterations=20,
                seed=3,
            )

        self.assertEqual(64, len(result["artifacts"]["direct"]["sha256"]))
        self.assertEqual(
            [0, 1],
            [item["round"] for item in result["artifacts"]["pascore_rounds"]],
        )
        self.assertEqual(1, result["analysis"]["pascore"]["semantic_success"])

    @staticmethod
    def outcome(success, level, subtype):
        return {
            "semantic_success": success,
            "category": "success" if success else "semantic_failure",
            "code": "semantic_success" if success else "mismatch",
            "level": level,
            "subtype": subtype,
        }


if __name__ == "__main__":
    unittest.main()
