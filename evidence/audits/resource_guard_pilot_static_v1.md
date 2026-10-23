# Pilot Resource Guard Static Check (Post-hoc)

Date: 2026-10-09. This is a supplementary engineering check designed after
the Held-out ablation was inspected, not a preregistered evaluation or a new
Held-out result.

## Inputs and Procedure

- Source: `benchmark/gold138_manifest.json`, SHA-256
  `5c00943dd49ff5e7ff67d318e9dda3fcb394164b407c02ecaf261e8b6eff6fac`.
- Scope: all 30 Pilot tasks; 17 contain at least one target whose declared
  `sample_num` or `estimated_total_mib` exceeds that task's safety policy;
  13 do not.
- For each task, select the first oversized target when present, otherwise
  the first target. Construct a Python source string containing one literal
  `Connection.get(<node>)` full read and another with literal
  `Connection.get('data(<node>)[0:9]')` bounded access. Call
  `guard_program(source, task, enabled=True/False)` directly. No generated
  program is executed, and no MDSplus server or model API is contacted.
- Reproduce with `python -m unittest tests.test_resource_aware_guard -v`.

## Observations

| Case | Guard enabled | Guard disabled |
| --- | ---: | ---: |
| Literal full read, oversized target: require bounded access | 17/17 | 0/17 |
| Literal bounded read, oversized target: allowed | 17/17 | 17/17 |
| Literal full read, within-budget target: allowed | 13/13 | 13/13 |

The check exercises the static policy on real Pilot metadata, not safety under
live execution. It does not measure peak memory, execution time, false
positives on model outputs, or whether a blocked program is successfully
repaired. The implementation reads only constant-string arguments to `.get()`;
for Pilot task `gold138_l2_015`, the literal full-read expression is blocked,
but an equivalent expression stored in a variable is allowed because no
literal `.get()` argument is extracted. This is a known coverage limitation,
not evidence that the dynamic read is safe.

The formal Qwen 60-task `w/o Resource-Aware Guard` ablation remains 58/60 in
both conditions because Full never issued a blocking decision on those task
instances. Keep this offline check separate from the Held-out accuracy table.
