import unittest

from scripts.audit_table6_cost import case_cost, paired_summary


class Table6CostTests(unittest.TestCase):
    def test_cumulative_cost_across_repair_rounds(self):
        key = ("qwen3.8-chat", "gold138_l3_001")
        phases = [
            ({key: {"input_tokens": 100, "output_tokens": 10,
                    "generation_seconds": 2.0}},
             {key: {"status": "program_failure", "elapsed_seconds": 3.0}}),
            ({key: {"input_tokens": 20, "output_tokens": 5,
                    "generation_seconds": 6.0}},
             {key: {"status": "execution_success", "elapsed_seconds": 4.0}}),
        ]
        cost = case_cost(key, phases)
        self.assertTrue(cost["complete"])
        self.assertEqual(cost["model_calls"], 2)
        self.assertEqual(cost["input_tokens"], 120)
        self.assertEqual(cost["output_tokens"], 15)
        self.assertEqual(cost["generation_seconds"], 8.0)
        self.assertEqual(cost["execution_seconds"], 7.0)

    def test_timeout_is_missing_and_matched_cohort_excludes_case(self):
        ok = ("qwen3.8-chat", "gold138_l1_001")
        timed_out = ("qwen3.8-chat", "gold138_l1_002")
        baseline = {
            key: case_cost(key, [
                ({key: {"input_tokens": 10, "output_tokens": 2,
                        "generation_seconds": 1.0}},
                 {key: {"status": "execution_success", "elapsed_seconds": 2.0}})
            ]) for key in (ok, timed_out)
        }
        full = {
            ok: case_cost(ok, [
                ({ok: {"input_tokens": 20, "output_tokens": 4,
                       "generation_seconds": 2.0}},
                 {ok: {"status": "execution_success", "elapsed_seconds": 3.0}})
            ]),
            timed_out: case_cost(timed_out, [
                ({}, {timed_out: {"status": "generation_failure"}})
            ]),
        }
        self.assertFalse(full[timed_out]["complete"])
        self.assertIsNone(full[timed_out]["input_tokens"])
        rows = paired_summary(baseline, full)
        self.assertEqual(rows["n"], 1)
        self.assertEqual(rows["excluded_cases"], [timed_out])
        self.assertEqual(rows["direct"]["mean_input_tokens"], 10)
        self.assertEqual(rows["full"]["mean_input_tokens"], 20)

    def test_policy_rejection_has_no_program_execution_time(self):
        key = ("qwen3.8-chat", "gold138_l2_001")
        cost = case_cost(key, [
            ({key: {"input_tokens": 50, "output_tokens": 10,
                    "generation_seconds": 2.5}},
             {key: {"status": "policy_rejected"}})
        ])
        self.assertTrue(cost["complete"])
        self.assertEqual(cost["execution_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
