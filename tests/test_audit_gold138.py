import copy
import json
import os
import tempfile
import unittest

from scripts.audit_gold138 import (
    audit_manifest_structure,
    audit_prompts,
    audit_references,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(name):
    with open(os.path.join(ROOT, "benchmark", name), "r") as handle:
        return json.load(handle)


def issue_codes(issues):
    return set(issue["code"] for issue in issues)


def one_task_manifest():
    return {
        "tasks": [
            {
                "id": "task_1",
                "level": "L1",
                "subtype": "simple_retrieval",
                "split": "held_out",
                "targets": [
                    {
                        "server": "server",
                        "tree": "east",
                        "shot": 1,
                        "subtree": "t1",
                        "node": "\\IP",
                        "signal_name": "IP",
                        "sample_num": 3,
                        "estimated_total_mib": 0.001,
                    }
                ],
                "operation": {"kind": "full_retrieval"},
                "prompt_spec": {"disclose_subtree": True},
                "safety_policy": {
                    "max_points": 5000000,
                    "max_total_mib": 256.0,
                    "timeout_seconds": 90,
                },
            }
        ]
    }


def success_reference(
    window=None,
    allow_full=True,
    increasing=True,
    task_id="task_1",
    repeat_id=1,
):
    return {
        "task_id": task_id,
        "repeat_id": repeat_id,
        "status": "success",
        "elapsed_seconds": 1.25,
        "resource_decisions": [{"allow_full_read": allow_full}],
        "targets": [
            {
                "target": copy.deepcopy(one_task_manifest()["tasks"][0]["targets"][0]),
                "window": window,
                "signal": {"sha256": "a", "size": 3},
                "time_axis": {"sha256": "b", "size": 3},
                "validation": {
                    "non_empty": True,
                    "same_size": True,
                    "signal_all_finite": True,
                    "time_all_finite": True,
                    "time_strictly_increasing": increasing,
                    "expected_point_count_match": True,
                },
            }
        ],
    }


class GoldAuditTests(unittest.TestCase):
    def test_real_manifest_has_exact_counts_and_no_split_leakage(self):
        issues, report = audit_manifest_structure(
            load_json("gold138_manifest.json"), load_json("splits.json")
        )
        self.assertEqual([], issues)
        self.assertEqual({"L1": 30, "L2": 48, "L3": 60}, report["by_level"])

    def test_manifest_detects_pilot_shot_in_held_out(self):
        manifest = load_json("gold138_manifest.json")
        splits = load_json("splits.json")
        manifest["tasks"][0]["targets"][0]["shot"] = splits["pilot_shots"][0]
        issues, _ = audit_manifest_structure(manifest, splits)
        self.assertIn("split_shot_leakage", issue_codes(issues))

    def test_manifest_detects_wrong_frozen_pilot_shots_and_level_mapping(self):
        manifest = load_json("gold138_manifest.json")
        splits = load_json("splits.json")
        bad_splits = copy.deepcopy(splits)
        bad_splits["pilot_shots"] = [70000, 106915, 140000]
        issues, _ = audit_manifest_structure(manifest, bad_splits)
        self.assertIn("wrong_pilot_shots", issue_codes(issues))

        bad_manifest = copy.deepcopy(manifest)
        l2 = next(t for t in bad_manifest["tasks"] if t["level"] == "L2")
        l3 = next(t for t in bad_manifest["tasks"] if t["level"] == "L3")
        l2["level"], l3["level"] = l3["level"], l2["level"]
        issues, _ = audit_manifest_structure(bad_manifest, splits)
        self.assertIn("subtype_level_mismatch", issue_codes(issues))

    def test_prompt_audit_detects_missing_and_forbidden_disclosure(self):
        manifest = one_task_manifest()
        prompts = [{"task_id": "task_1", "prompt": "contains sha256"}]
        issues, _ = audit_prompts(manifest, prompts)
        self.assertIn("prompt_oracle_disclosure", issue_codes(issues))
        issues, _ = audit_prompts(manifest, [])
        self.assertIn("missing_prompt", issue_codes(issues))

    def test_prompt_audit_detects_oracle_slice_values(self):
        manifest = one_task_manifest()
        manifest["tasks"][0]["operation"] = {
            "kind": "index_window",
            "index_start": 7000,
            "index_end": 17000,
        }
        prompts = [{"task_id": "task_1", "prompt": "use slice 7000:17000"}]
        issues, _ = audit_prompts(manifest, prompts)
        self.assertIn("prompt_oracle_value_disclosure", issue_codes(issues))

    def test_reference_audit_detects_missing_task(self):
        issues, _ = audit_references(one_task_manifest(), {"tasks": {}})
        self.assertIn("missing_reference", issue_codes(issues))

    def test_reference_audit_detects_unstable_repeats(self):
        index = {
            "tasks": {
                "task_1": {
                    "repeat_files": [],
                    "stability": {"stable": False, "differences": ["hash"]},
                }
            }
        }
        issues, _ = audit_references(one_task_manifest(), index)
        self.assertIn("unstable_reference", issue_codes(issues))

    def test_reference_audit_detects_unsafe_full_read_and_bad_time(self):
        manifest = one_task_manifest()
        manifest["tasks"][0]["targets"][0]["sample_num"] = 6000000
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for repeat in (1, 2):
                value = success_reference(
                    window=None,
                    allow_full=False,
                    increasing=False,
                    repeat_id=repeat,
                )
                value["targets"][0]["target"] = copy.deepcopy(
                    manifest["tasks"][0]["targets"][0]
                )
                path = os.path.join(directory, "repeat{}.json".format(repeat))
                with open(path, "w") as handle:
                    json.dump(value, handle)
                files.append(path)
            index = {
                "tasks": {
                    "task_1": {
                        "repeat_files": files,
                        "stability": {"stable": True, "differences": []},
                    }
                }
            }
            issues, _ = audit_references(manifest, index)
        codes = issue_codes(issues)
        self.assertIn("unsafe_full_read", codes)
        self.assertIn("invalid_time_axis", codes)

    def test_reference_audit_detects_non_finite_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for repeat in (1, 2):
                value = success_reference(repeat_id=repeat)
                value["targets"][0]["validation"]["signal_all_finite"] = False
                path = os.path.join(directory, "repeat{}.json".format(repeat))
                with open(path, "w") as handle:
                    json.dump(value, handle)
                files.append(path)
            index = {
                "tasks": {
                    "task_1": {
                        "repeat_files": files,
                        "stability": {"stable": True, "differences": []},
                    }
                }
            }
            issues, _ = audit_references(one_task_manifest(), index)
        self.assertIn("invalid_signal_values", issue_codes(issues))

    def test_reference_audit_rejects_stale_exit_status_and_wrong_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for repeat in (1, 2):
                value = success_reference(
                    task_id="wrong_task", repeat_id=repeat
                )
                path = os.path.join(directory, "repeat{}.json".format(repeat))
                with open(path, "w") as handle:
                    json.dump(value, handle)
                files.append(path)
            index = {
                "timeout_seconds": 90,
                "tasks": {
                    "task_1": {
                        "repeat_files": files,
                        "exit_statuses": [124, 124],
                        "stability": {"stable": True, "differences": []},
                    }
                },
            }
            issues, _ = audit_references(one_task_manifest(), index)
        codes = issue_codes(issues)
        self.assertIn("nonzero_reference_exit", codes)
        self.assertIn("reference_task_mismatch", codes)

    def test_reference_audit_recomputes_resource_guard(self):
        manifest = one_task_manifest()
        manifest["tasks"][0]["targets"][0]["sample_num"] = 6000000
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for repeat in (1, 2):
                value = success_reference(allow_full=True, repeat_id=repeat)
                value["targets"][0]["target"] = copy.deepcopy(
                    manifest["tasks"][0]["targets"][0]
                )
                path = os.path.join(directory, "repeat{}.json".format(repeat))
                with open(path, "w") as handle:
                    json.dump(value, handle)
                files.append(path)
            index = {
                "timeout_seconds": 90,
                "tasks": {
                    "task_1": {
                        "repeat_files": files,
                        "exit_statuses": [0, 0],
                        "stability": {"stable": True, "differences": []},
                    }
                },
            }
            issues, _ = audit_references(manifest, index)
        self.assertIn("resource_decision_mismatch", issue_codes(issues))

    def test_reference_audit_rejects_wrong_operation_window(self):
        manifest = one_task_manifest()
        manifest["tasks"][0]["subtype"] = "exact_window"
        manifest["tasks"][0]["operation"] = {
            "kind": "index_window",
            "index_start": 0,
            "index_end": 2,
            "expected_points": 3,
            "time_start": 0.0,
            "time_end": 0.2,
            "time_step": 0.1,
        }
        wrong_window = dict(manifest["tasks"][0]["operation"])
        wrong_window["index_start"] = 1
        wrong_window["index_end"] = 3
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for repeat in (1, 2):
                value = success_reference(
                    window=wrong_window, repeat_id=repeat
                )
                value["targets"][0]["validation"].update(
                    {
                        "window_start_match": True,
                        "window_end_match": True,
                        "time_step_match": True,
                    }
                )
                path = os.path.join(directory, "repeat{}.json".format(repeat))
                with open(path, "w") as handle:
                    json.dump(value, handle)
                files.append(path)
            index = {
                "timeout_seconds": 90,
                "tasks": {
                    "task_1": {
                        "repeat_files": files,
                        "exit_statuses": [0, 0],
                        "stability": {"stable": True, "differences": []},
                    }
                },
            }
            issues, _ = audit_references(manifest, index)
        self.assertIn("reference_window_mismatch", issue_codes(issues))

    def test_reference_audit_requires_pair_derived_outputs(self):
        manifest = one_task_manifest()
        task = manifest["tasks"][0]
        second_target = copy.deepcopy(task["targets"][0])
        second_target["node"] = "\\IP2"
        second_target["signal_name"] = "IP2"
        task["targets"].append(second_target)
        task["level"] = "L3"
        task["subtype"] = "same_shot_alignment"
        window = {
            "index_start": 0,
            "index_end": 2,
            "expected_points": 3,
            "time_start": 0.0,
            "time_end": 0.2,
            "time_step": 0.1,
        }
        task["operation"] = {
            "kind": "same_shot_alignment",
            "common_points": 3,
            "per_target_windows": [window, window],
        }
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for repeat in (1, 2):
                value = success_reference(window=window, repeat_id=repeat)
                value["targets"][0]["validation"].update(
                    {
                        "window_start_match": True,
                        "window_end_match": True,
                        "time_step_match": True,
                    }
                )
                second_result = copy.deepcopy(value["targets"][0])
                second_result["target"] = copy.deepcopy(second_target)
                value["targets"].append(second_result)
                value["resource_decisions"].append({"allow_full_read": True})
                path = os.path.join(directory, "repeat{}.json".format(repeat))
                with open(path, "w") as handle:
                    json.dump(value, handle)
                files.append(path)
            index = {
                "timeout_seconds": 90,
                "tasks": {
                    "task_1": {
                        "repeat_files": files,
                        "exit_statuses": [0, 0],
                        "stability": {"stable": True, "differences": []},
                    }
                },
            }
            issues, _ = audit_references(manifest, index)
        self.assertIn("missing_derived_result", issue_codes(issues))


if __name__ == "__main__":
    unittest.main()
