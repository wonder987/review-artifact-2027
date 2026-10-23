import hashlib
import json
import os
import tempfile
import unittest
import zipfile

from scripts.audit_baseline_ablation_reuse import audit_reuse


class BaselineAblationReuseAuditTests(unittest.TestCase):
    def write_diagnostics(self, path, evaluations):
        entries = []
        with zipfile.ZipFile(path, "w") as archive:
            for evaluation in evaluations:
                model = evaluation["model_alias"]
                task = evaluation["task_id"]
                member = "evaluation/{}/{}/evaluation.json".format(model, task)
                payload = json.dumps(evaluation, sort_keys=True).encode("utf-8")
                archive.writestr(member, payload)
                entries.append({
                    "model_alias": model,
                    "task_id": task,
                    "files": {
                        "evaluation": {
                            "path": member,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                    },
                })
            archive.writestr("manifest.json", json.dumps({
                "entry_count": len(entries),
                "entries": entries,
            }))

    def test_audit_reports_reusable_conditions_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            round_paths = [os.path.join(directory, "round-{}.zip".format(i))
                           for i in range(3)]
            self.write_diagnostics(round_paths[0], [
                self.outcome("qwen", "task-1", True, "L1"),
                self.outcome("qwen", "task-2", False, "L2"),
                self.outcome("other", "task-1", False, "L1"),
                self.outcome("other", "task-2", False, "L2"),
            ])
            self.write_diagnostics(round_paths[1], [
                self.outcome("qwen", "task-2", False, "L2"),
            ])
            self.write_diagnostics(round_paths[2], [
                self.outcome("qwen", "task-2", True, "L2"),
            ])
            split_path = os.path.join(directory, "splits.json")
            with open(split_path, "w") as handle:
                json.dump({"ablation": ["task-1", "task-2"]}, handle)

            result = audit_reuse(round_paths, split_path, "qwen")

        self.assertEqual(2, result["prompt_only_heldout"]["total"])
        self.assertEqual(1, result["prompt_only_heldout"]["semantic_success"])
        self.assertEqual(1, result["single_round_ablation"]["semantic_success"])
        self.assertEqual(2, result["full_ablation"]["semantic_success"])
        self.assertEqual([2, 1, 1], result["eligible_entries_by_round"])
        self.assertEqual(64, len(result["artifacts"]["split"]["sha256"]))
        self.assertEqual(3, len(result["artifacts"]["diagnostic_rounds"]))

    def test_audit_rejects_ablation_task_missing_from_round_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            round_path = os.path.join(directory, "round-0.zip")
            self.write_diagnostics(round_path, [
                self.outcome("qwen", "task-1", True, "L1"),
            ])
            split_path = os.path.join(directory, "splits.json")
            with open(split_path, "w") as handle:
                json.dump({"ablation": ["task-1", "task-2"]}, handle)

            with self.assertRaisesRegex(ValueError, "missing ablation tasks"):
                audit_reuse([round_path], split_path, "qwen")

    @staticmethod
    def outcome(model, task, success, level):
        return {
            "model_alias": model,
            "task_id": task,
            "level": level,
            "subtype": "simple_retrieval",
            "category": "success" if success else "semantic_failure",
            "code": "semantic_success" if success else "mismatch",
            "semantic_success": success,
        }


if __name__ == "__main__":
    unittest.main()
