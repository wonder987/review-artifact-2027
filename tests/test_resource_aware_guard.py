import json
import unittest
from pathlib import Path

from scripts.resource_aware_guard import guard_program


TASK = {
    "id": "task-large",
    "operation": {"kind": "exact_window"},
    "safety_policy": {
        "max_points": 5000000,
        "max_total_mib": 256.0,
        "timeout_seconds": 90,
    },
    "targets": [{
        "estimated_total_mib": 600.0,
        "node": "\\BIG1",
        "sample_num": 50000000,
        "signal_name": "BIG1",
    }],
}


class ResourceAwareGuardTests(unittest.TestCase):
    def test_requires_bounded_access_for_oversized_full_read(self):
        source = "from MDSplus import Connection\nc=Connection('s')\ny=c.get(r'\\BIG1').data()\n"

        result = guard_program(source, TASK, enabled=True)

        self.assertEqual("require_bounded_access", result["decision"])
        self.assertTrue(result["checks"]["full_read_detected"])

    def test_allows_bounded_read(self):
        source = "from MDSplus import Connection\nc=Connection('s')\ny=c.get(r'data(\\BIG1)[0:1000]').data()\n"

        result = guard_program(source, TASK, enabled=True)

        self.assertEqual("allow", result["decision"])
        self.assertTrue(result["checks"]["bounded_read_detected"])

    def test_ablation_disables_only_task_aware_decision(self):
        source = "c.get(r'\\BIG1').data()\n"

        result = guard_program(source, TASK, enabled=False)

        self.assertEqual("allow", result["decision"])
        self.assertFalse(result["enabled"])

    def test_literal_access_decisions_on_pilot_metadata(self):
        manifest_path = Path(__file__).resolve().parents[1] / "benchmark" / "gold138_manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            tasks = [task for task in json.load(handle)["tasks"] if task["split"] == "pilot"]

        oversized_count = 0
        for task in tasks:
            policy = task["safety_policy"]
            oversized = [
                target for target in task["targets"]
                if target.get("sample_num", 0) > policy["max_points"]
                or target.get("estimated_total_mib", 0) > policy["max_total_mib"]
            ]
            oversized_count += bool(oversized)
            node = (oversized or task["targets"])[0]["node"]
            full_read = "conn.get({!r}).data()".format(node)
            bounded_read = "conn.get({!r}).data()".format(
                "data({})[0:9]".format(node)
            )

            with self.subTest(task_id=task["id"]):
                enabled = guard_program(full_read, task, enabled=True)
                disabled = guard_program(full_read, task, enabled=False)
                bounded = guard_program(bounded_read, task, enabled=True)
                bounded_disabled = guard_program(bounded_read, task, enabled=False)
                self.assertEqual(
                    "require_bounded_access" if oversized else "allow",
                    enabled["decision"],
                )
                self.assertEqual("allow", disabled["decision"])
                self.assertEqual("allow", bounded["decision"])
                self.assertEqual("allow", bounded_disabled["decision"])

        self.assertEqual(30, len(tasks))
        self.assertEqual(17, oversized_count)


if __name__ == "__main__":
    unittest.main()
