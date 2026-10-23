# Experiment Results Registry

Last updated: 2026-10-13

This file is the only manuscript-facing registry for quantitative results. Raw
artifacts remain under `experiments/`; values enter manuscript prose only after
their source hashes and denominators are recorded here.

## A. Development Evidence

### A1. Direct Zero-shot Pilot Failure Study

- Tasks: 30.
- Models: 3.
- Task-model cases: 90.
- Semantic successes: 1.
- Failures classified: 89.
- Failure denominator for taxonomy percentages: 89.

| Major category | Count | Fraction of 89 failures |
| --- | ---: | ---: |
| MDSplus interaction | 55 | 61.80% |
| Code generation | 15 | 16.85% |
| Semantic result | 10 | 11.24% |
| Resource/runtime | 9 | 10.11% |

Source: `figures/source_data/pilot_failure_taxonomy_v1.csv`.

### A2. Full PA-SCoRe Pilot Gate

- Round 0 semantic successes: 59/90.
- Round 1 newly repaired: 19/31.
- Round 2 newly repaired: 4/12.
- Final cumulative semantic successes: 82/90 (91.11%).
- Terminal failures: 8.
- Infrastructure failures: 0.

Source: `experiments/config/pascore_v2_freeze.json` and
`experiments/audits/pascore_pilot_full_v2_round2_terminal_audit.md`.

Use: development gate and method-design evidence only.

## B. Formal Held-out Evidence

### B1. Direct Zero-shot Model Survey

Scope: 108 tasks x 3 models = 324 task-model cases.

| Model | Generated | Syntax valid | Execution success | Semantic success | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.8 Chat | 108 | 106 | 0/108 (0.00%) | 0/108 (0.00%) | 0.00% |
| DeepSeek V4 Flash Ascend1 | 108 | 103 | 21/108 (19.44%) | 3/108 (2.78%) | 2.78% |
| Gemini 3.5 Flash Lite | 108 | 78 | 54/108 (50.00%) | 9/108 (8.33%) | 8.33% |
| **Overall** | **324** | **287** | **75/324 (23.15%)** | **12/324 (3.70%)** | **3.70%** |

The syntax counts above describe local generation artifacts only and are not
formal semantic results.

The level-macro semantic accuracy is 4.36%: L1 is 5/72 (6.94%), L2 is
7/114 (6.14%), and L3 is 0/138 (0.00%). Of the 75 execution-success cases, 68
are protocol-valid and 12 are semantically correct. Thus protocol compliance
among executed programs is 68/75 (90.67%), while semantic correctness among
protocol-valid executions is 12/68 (17.65%).

Generation coverage is complete: all three models contain exactly the 108
Held-out task IDs, with no missing or unexpected task. The packaged batch
contains 324 entries. Its ZIP SHA-256 is
`aaaeb6b9c16f96e766c6ebad727a2d0ba45f97948a796feea47bcba37ed80969`.
The 2 Qwen, 5 DeepSeek, and 30 Gemini syntax failures remain unmodified model
outputs and will be counted as generation or execution failures by the frozen
evaluation pipeline.

Formal source: `experiments/packages/direct-zero-shot-heldout-v2-diagnostics.zip`
(SHA-256 `d81711b0fca40475667a1d05ff92237e07b1b2924a1fcbae164299154cd6bb84`).
The archive contains 324 unique execution records and 324 unique evaluation
records. The evaluation JSONL SHA-256 is
`f16ead9b1ce51e6beae159189eae7e8aae37e4d4908e59da7e2d7a26ba6d0865`.
Primary evaluation uses `diagnostic-v2` under the documented prompt/evaluator
alignment amendment; the original strict result is retained only as a
sensitivity result. See
`experiments/audits/direct_zero_shot_heldout_result_audit.json` and
`experiments/audits/direct_zero_shot_v2_evaluation_mode_amendment.json`.

### B2. Full PA-SCoRe

Scope: 108 tasks x 3 models, Round 0 plus at most two repair rounds.

| Stage | Eligible cases | Newly successful | Cumulative success | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Round 0 | 324 | 197 | 197/324 | 60.80% |
| Round 1 | 127 | 100 | 297/324 | 91.67% |
| Round 2 | 27 | 18 | 315/324 | 97.22% |

Round 0 level-macro semantic accuracy is 65.99%. Per-model semantic success is
76/108 (70.37%) for Qwen, 55/108 (50.93%) for DeepSeek, and 66/108 (61.11%)
for Gemini. Per-level success is 62/72 (86.11%) for L1, 92/114 (80.70%) for
L2, and 43/138 (31.16%) for L3. These are initial grounded-generation results,
not the final repaired PA-SCoRe outcome.

Formal Round 0 source:
`experiments/packages/pascore-heldout-full-v2-diagnostics-round-0.zip`
(SHA-256 `a4f26935b0692c6649607f16d10bf84f4b21e95c0d65239279d62ba961152c07`).
All 324 task-model pairs and all 2057 packaged files passed the manifest hash
audit. See
`experiments/audits/pascore_heldout_full_v2_round0_result_audit.json`.

Round 1 repaired 100/127 eligible failures (78.74%), raising cumulative
semantic success to 297/324 (91.67%). Cumulative per-model success is 104/108
(96.30%) for Qwen, 91/108 (84.26%) for DeepSeek, and 102/108 (94.44%) for
Gemini. Cumulative per-level success is 72/72 (100.00%) for L1, 114/114
(100.00%) for L2, and 111/138 (80.43%) for L3. The remaining 27 cases are
eligible for the frozen second repair round.

Formal Round 1 source:
`experiments/packages/pascore-heldout-full-v2-diagnostics-round-1.zip`
(SHA-256 `5fece80ac52d1aa95e28f7c103663f9f02311d433a0f101d5dcef2a2cbc6585d`).
All 127 task-model pairs and all 789 manifest-listed files passed the hash
audit. See
`experiments/audits/pascore_heldout_full_v2_round1_result_audit.json`.

Round 2 repaired 18/27 eligible failures (66.67%), producing a terminal result
of 315/324 semantic successes (97.22%) and 9 terminal failures. Final per-model
success is 105/108 (97.22%) for Qwen, 102/108 (94.44%) for DeepSeek, and
108/108 (100.00%) for Gemini. Final per-level success is 72/72 (100.00%) for
L1, 114/114 (100.00%) for L2, and 129/138 (93.48%) for L3. Terminal execution
success is 319/324 (98.46%).

Formal Round 2 source:
`experiments/packages/pascore-heldout-full-v2-diagnostics-round-2.zip`
(SHA-256 `b533056e9b0fa9a80de8c940b3447f9b5a266945062025be8420385d4e6affd1`).
All 27 task-model pairs and all 149 manifest-listed files passed the hash
audit. See
`experiments/audits/pascore_heldout_full_v2_round2_terminal_audit.json`.

### B3. Paired Primary Comparison

| Metric | Direct Zero-shot | Full PA-SCoRe | Absolute delta | Paired test |
| --- | ---: | ---: | ---: | --- |
| Micro semantic accuracy | 12/324 (3.70%) | 315/324 (97.22%) | +93.52 pp | Exact McNemar p = 1.23e-91; task-cluster bootstrap 95% CI [+90.74, +95.99] pp |
| Macro L1/L2/L3 accuracy | 4.36% | 97.83% | +93.46 pp | Stratified task-cluster bootstrap 95% CI [+90.63, +96.04] pp |
| Execution success | 75/324 (23.15%) | 319/324 (98.46%) | +75.31 pp | Descriptive |

For semantic accuracy, the paired contingency contains 303 PA-SCoRe-only
successes, zero Direct-only successes, and 303 discordant task-model pairs.
Bootstrap intervals use 10,000 deterministic resamples (seed 20260930), with
task as the resampling unit so that the three model observations for a task
remain clustered. The macro interval additionally stratifies tasks by L1/L2/L3.
Formal machine-readable source:
`experiments/audits/pascore_heldout_full_v2_final_comparison.json`.

### B4. Breakdown

Record per-model, per-level, and per-subtype numerators before drafting Results.

| Breakdown | Direct Zero-shot | Full PA-SCoRe | Delta |
| --- | ---: | ---: | ---: |
| Qwen | 0/108 (0.00%) | 105/108 (97.22%) | +97.22 pp |
| DeepSeek | 3/108 (2.78%) | 102/108 (94.44%) | +91.67 pp |
| Gemini | 9/108 (8.33%) | 108/108 (100.00%) | +91.67 pp |
| L1 | 5/72 (6.94%) | 72/72 (100.00%) | +93.06 pp |
| L2 | 7/114 (6.14%) | 114/114 (100.00%) | +93.86 pp |
| L3 | 0/138 (0.00%) | 129/138 (93.48%) | +93.48 pp |

Direct Zero-shot subtype results are 5/72 (6.94%) for simple retrieval, 3/39
(7.69%) for subtree discovery, 0/39 (0.00%) for exact windows, 4/36 (11.11%)
for sampling-constrained access, and 0% for all three L3 subtypes: same-shot
alignment (0/45), cross-shot comparison (0/48), and adaptive access (0/45).

Full PA-SCoRe subtype results are 72/72 (100.00%) for simple retrieval,
39/39 (100.00%) for subtree discovery, 39/39 (100.00%) for exact windows,
36/36 (100.00%) for sampling-constrained access, 39/45 (86.67%) for same-shot
alignment, 46/48 (95.83%) for cross-shot comparison, and 44/45 (97.78%) for
adaptive access.

### B5. Token Use and Stage Time

Table 6 reports means of per-task cumulative cost over all attempted rounds.
Within each model, Direct Zero-shot and Full PA-SCoRe use the same complete-case
task subset. Qwen and Gemini use 108/108 tasks each. DeepSeek uses 102/108;
the omitted task IDs are `gold138_l3_030`, `gold138_l3_035`,
`gold138_l3_043`, `gold138_l3_045`, `gold138_l3_046`, and
`gold138_l3_051`. Two lack a returned Round-0 API response and four lack a
returned Round-2 response. Their missing usage/time values are not zero.

| Model | Method | Paired n | Calls/task | Input tokens/task | Output tokens/task | API s/task | Execution s/task |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.8 Chat | Direct Zero-shot | 108 | 1.00 | 417.00 | 1914.67 | 86.88 | 1.62 |
| Qwen 3.8 Chat | Full PA-SCoRe | 108 | 1.33 | 2171.03 | 1972.32 | 97.37 | 6.93 |
| DeepSeek V4 Flash Ascend1 | Direct Zero-shot | 102 | 1.00 | 385.58 | 1219.82 | 116.68 | 6.52 |
| DeepSeek V4 Flash Ascend1 | Full PA-SCoRe | 102 | 1.57 | 2579.80 | 1804.79 | 226.56 | 9.07 |
| Gemini 3.5 Flash Lite | Direct Zero-shot | 108 | 1.00 | 418.20 | 1857.69 | 8.63 | 6.47 |
| Gemini 3.5 Flash Lite | Full PA-SCoRe | 108 | 1.44 | 2835.08 | 2504.98 | 10.51 | 8.02 |

API time is the Windows request wall time, including network and retries.
Execution time is the cs1 sandbox subprocess wall time; a policy rejection or
generation failure has no launched subprocess and contributes no execution
time. Neither measure includes manual host transfer, package preparation, or
feedback assembly. Provider token counts are comparable within a model, not
across provider tokenizers. These figures are not billed cost or end-to-end
latency. The 324 Direct Zero-shot generation records and 322, 127, and 23
returned Full generation records provide token/time values. Execution archives
contain 324 Direct and 324, 127, and 27 Full per-round records.

The executable audit is `scripts/audit_table6_cost.py`; its machine-readable
summary is `experiments/audits/table6_cost_latency_v1.json` and the 648
per-task/condition records are
`experiments/audits/table6_cost_latency_cases_v1.csv` (SHA-256
`b3f244dd62e034936dd32eaa3ca31c6b001d0b10e8a0759a480a9e985e7567c3`).
The four execution-archive SHA-256 values are, in Direct/Round-0/Round-1/Round-2
order: `d81711b0fca40475667a1d05ff92237e07b1b2924a1fcbae164299154cd6bb84`,
`a4f26935b0692c6649607f16d10bf84f4b21e95c0d65239279d62ba961152c07`,
`5fece80ac52d1aa95e28f7c103663f9f02311d433a0f101d5dcef2a2cbc6585d`,
and `b533056e9b0fa9a80de8c940b3447f9b5a266945062025be8420385d4e6affd1`.

## C. Baselines and Ablations

The paper groups PA-SCoRe into four conceptual modules: Resource-Grounded
Access (metadata grounding plus its internal pre-execution resource guard),
Failure Diagnosis, Structured Feedback, and Iterative Repair. This is a
presentation-level grouping, not a change to the frozen implementation or
experimental conditions. The table below retains the separately executed
`no_metadata_grounding` and `no_resource_guard` interventions; the
single-round condition tests the repair budget rather than removing a fifth
module. Neither grouping nor the combined access-module label implies that
the guard independently improves semantic accuracy.

| Experiment | Model | Tasks | Status | Source hash |
| --- | --- | ---: | --- | --- |
| Prompt-only baseline | Qwen 3.8 Chat | 108 | 76/108 (70.37%) | `6367a59b254747b060c8be37c9856d7a993a1a7960bc38d413333b65784feab1` |
| Feedback baseline | Qwen 3.8 Chat | 108 | 83/108 (76.85%), terminal | Round 0-2 diagnostics below |
| Full PA-SCoRe ablation reference | Qwen 3.8 Chat | 60 | 58/60 (96.67%) | `6367a59b254747b060c8be37c9856d7a993a1a7960bc38d413333b65784feab1` |
| Without Metadata Grounding | same fixed model | 60 | 0/60 (0.00%), terminal | Round 0-2 diagnostics below |
| Without Failure Diagnosis | same fixed model | 60 | 46/60 (76.67%), terminal | Feedback baseline subset below |
| Without Structured Feedback | same fixed model | 60 | 55/60 (91.67%), terminal | Round 0-2 diagnostics below |
| Without Resource-Aware Guard | same fixed model | 60 | 58/60 (96.67%), terminal; no difference from Full | Round 0-2 diagnostics below |
| Single-Round Repair | same fixed model | 60 | 57/60 (95.00%) | `6367a59b254747b060c8be37c9856d7a993a1a7960bc38d413333b65784feab1` |

The fixed model is Qwen 3.8 Chat. Prompt-only is the metadata-grounded
Round-0 generation without execution feedback or repair; it is neither the
Direct Zero-shot condition nor a few-shot prompt. The ablation subset contains
13 L1, 21 L2, and 26 L3 held-out tasks. On this subset, Prompt-only is 44/60
(73.33%); Single-Round Repair is 57/60 (95.00%); and terminal Full PA-SCoRe is
58/60 (96.67%). The deterministic source is
`experiments/audits/baseline_ablation_reuse_v1.json`, bound to
`benchmark/splits.json` SHA-256
`2163fdd4d290693476b923e38e4a71c34fcde81c67b489808f588b316cb7b881`.

### C1. Feedback Baseline and Failure-Diagnosis Ablation

The `no_failure_diagnosis` condition uses the same Qwen Round-0 programs as
Full PA-SCoRe. It replaces multi-grained diagnosis with raw execution feedback
and permits at most two repairs. On all 108 held-out tasks, Round 0 succeeds
on 76/108, Round 1 adds 7/32, and Round 2 adds 0/25. Terminal accuracy is
83/108 (76.85%): L1 24/24, L2 36/38, and L3 23/46. On the frozen 60-task
ablation subset, terminal accuracy is 46/60 (76.67%): L1 13/13, L2 20/21,
and L3 13/26. Full PA-SCoRe on the same subset is 58/60 (96.67%).

Formal sources, in round order:

- `experiments/packages/feedback-baseline-heldout-qwen-v1-diagnostics-round-0.zip` (SHA-256 `5587191e5805e6427547ad89c506c89575a084678c2bef227e294da07fb72e32`).
- `experiments/packages/feedback-baseline-heldout-qwen-v1-diagnostics-round-1.zip` (SHA-256 `5e798f653658983b88061d947f87ad527d1b5cc43b944244a42ce37236495218`).
- `experiments/packages/feedback-baseline-heldout-qwen-v1-diagnostics-round-2.zip` (SHA-256 `90c87ca87d471825c80ed1b76da0b3a47253e49ecfbc02868ea27f1a15cde7a2`).

Each archive contains 108, 32, and 25 distinct Qwen task records respectively;
all referenced evaluation payload hashes passed verification. The Round-2
terminal failures are 25, including 18 `signal_dtype_mismatch` cases. The
strict-dtype sensitivity analysis remains pending.

### C2. Without Metadata Grounding

The Qwen `no_metadata_grounding` condition has completed Round 0 on the frozen
60-task subset: 0/60 semantic successes and 60/60 execution failures. Failure
codes are 47 `api_invocation_error`, 11 `runtime_error`, and 2
`policy_rejected`; all 60 cases are eligible for repair. This is an intermediate
round result, not the terminal ablation accuracy.

Formal source:
`experiments/packages/ablation-no-metadata-grounding-qwen-v1-diagnostics-round-0.zip`
(SHA-256 `d784c8ede9c3c12df0fa8829b9993cd1c677b0b0c87b21cba4b74c3cadbab131`).
Its `source_manifest_sha256` matches the corrected execution package
`experiments/packages/ablation-no-metadata-grounding-qwen-v1-round-0-runtimefix.zip`
(SHA-256 `e06ec0a0df7111243d70dbae19d7957137c484e15a0d2eb2eb645df597798953`).
The original package failed before executing any generated program; the
runtime-only replacement and unchanged program entries are documented in
`experiments/audits/ablation_no_metadata_grounding_runtimefix_v1.json`.

Round 1 repaired 0/60 eligible failures. Its evaluation contains 59 execution
failures and one semantic failure; failure codes are 47 `runtime_error`, 5
`node_lookup_error`, 4 `policy_rejected`, 3 `api_invocation_error`, and one
`signal_dtype_mismatch`. Infrastructure failures are zero. All 60 cases remain
eligible for the final repair round. The error messages include incompatible
MDSplus calls and invalid TDI expressions, so this stage is not a shared
package-verification failure. Formal Round 1 source:
`experiments/packages/ablation-no-metadata-grounding-qwen-v1-diagnostics-round-1.zip`
(SHA-256 `fcc6bd81082a632e94e0a244d2024933e2037a3446e8c401d6d796100babbca8`).
Its `source_manifest_sha256` matches
`experiments/packages/ablation-no-metadata-grounding-qwen-v1-round-1.zip`.

Round 2 repaired 0/60 eligible failures. Its evaluation contains 53 execution
failures and 7 semantic failures, with no generation or infrastructure failure.
The terminal result is 0/60 (0.00%) after two repair rounds: L1 0/13, L2
0/21, and L3 0/26. All three diagnostic rounds contain the same 60 frozen
task IDs, one Qwen record per task, and use `semantic-evaluator-v2`. The
Round-2 failure codes are 36 `runtime_error`, 10 `node_lookup_error`, 4
`policy_rejected`, 3 `api_invocation_error`, 2
`time_axis_dtype_mismatch`, 2 `signal_dtype_mismatch`, and one each of
`target_identity_mismatch`, `signal_hash_mismatch`, and
`time_axis_hash_mismatch`.

Formal Round 2 source:
`experiments/packages/ablation-no-metadata-grounding-qwen-v1-diagnostics-round-2.zip`
(SHA-256 `6b4ecbe5306ce8a16124177014e3e2de9e15ff31cf38eeae6df0ee0aa7bc29c3`).
Its `source_manifest_sha256` matches
`experiments/packages/ablation-no-metadata-grounding-qwen-v1-round-2.zip`
(SHA-256 `edba759c4b2bf71e24ab55cc5b8ae3dd414d95698c5321259c91e1a9ba2eb592`).

### C3. Without Structured Feedback: Rounds 0-2

The Qwen `no_structured_feedback` condition has completed Round 0 on the same
60 frozen tasks: 44/60 semantic successes (73.33%), with 16 eligible for
repair. Per-level success is L1 13/13, L2 18/21, and L3 13/26. The 60 initial
program hashes and per-task Round-0 success outcomes match Full PA-SCoRe on
this subset. All 16 repair feedback records have `structured: false` and carry
raw errors. This is an intermediate result, not the terminal ablation accuracy.

Formal source:
`experiments/packages/ablation-no-structured-feedback-qwen-v1-diagnostics-round-0.zip`
(SHA-256 `c49d4d9f95dc8f249dad33f26011531c59d7be3658b069342823f93a2e36e297`).
Its `source_manifest_sha256` matches
`experiments/packages/ablation-no-structured-feedback-qwen-v1-round-0.zip`
(SHA-256 `e11f87e46175cff34ff3f54f6e44e7bbcde8f5e8f15a71d74db7122baaad9024`).

Round 1 repaired 10/16 eligible failures. Cumulative success is 54/60
(90.00%), leaving six cases eligible for Round 2. The 16 Round-1 cases contain
10 successes, five semantic failures, and one execution failure; no generation,
protocol, or infrastructure failures were reported. Per-level Round-1 repairs
are L2 2/3 and L3 8/13. The imported diagnostic archive contains 16 unique
Qwen task records; all referenced file hashes passed verification, its source
manifest hash matches the Round-1 execution package, and the individual
evaluation categories reproduce the reported counts. Formal Round 1 source:
`experiments/packages/ablation-no-structured-feedback-qwen-v1-diagnostics-round-1.zip`
(SHA-256 `7ed8c5ae72099f76f1327870f773bfbb8dea2cd9be55d918e5b4238333fb740c`).
The six feedback-bearing task IDs match the Round-2 repair plan exactly.

Round 2 repaired 1/6 remaining failures (one L3 adaptive-access task). Its six cases
contain one success, four semantic failures, and one execution failure, with
zero generation, protocol, or infrastructure failures. Cumulative semantic
success is 55/60 (91.67%): L1 13/13, L2 20/21, and L3 22/26. Full PA-SCoRe
on the same frozen subset is 58/60 (96.67%), an absolute difference of 3/60
(5.00 percentage points). The diagnostic archive contains six unique Qwen
task records; all 36 referenced payload hashes passed verification, the
source manifest hash matches the Round-2 execution package, and the six task
IDs match the Round-2 repair plan. Individual evaluation records reproduce the
reported success and failure counts. Formal Round 2 source:
`experiments/packages/ablation-no-structured-feedback-qwen-v1-diagnostics-round-2.zip`
(SHA-256 `add083f42e3f1bcd0a1491f9a0c9096ab94ba1d468f40c136ffc2a6f91900fc1`).

### C4. Without Resource-Aware Guard: Rounds 0-2

The Qwen `no_resource_guard` condition has a Round-0 result of
44/60 semantic successes (73.33%) on the frozen ablation subset, with 16 cases
eligible for repair. Its summary reports 56 execution successes, four execution
failures, 12 semantic failures, and no generation, protocol, or infrastructure
failures. Per-level success is L1 13/13, L2 18/21, and L3 13/26. The initial
60 program hashes in the prepared execution package match Full PA-SCoRe on the
same task IDs. The 60 per-task evaluation categories and codes also match Full
exactly. In Full's Round-0 diagnostics for this subset, 57 programs reached the
guard and all 57 decisions were `allow`; the other three were rejected by the
separate source policy before the guard. Thus disabling the resource guard did
not exercise a blocking decision in Round 0, limiting what this comparison can
establish about that module. The diagnostic archive contains 60 unique Qwen
task records; all 373 referenced payload hashes passed verification, its
source manifest hash matches the Round-0 execution package, and 16 records
carry feedback for repair. Formal Round 0 source:
`experiments/packages/ablation-no-resource-guard-qwen-v1-diagnostics-round-0.zip`
(SHA-256 `abea2eb1b6a9c1445226fbaa1e0684bc6e0afc03a36bcc4c45a03425fb2773a6`).
This is not the terminal ablation result.

Round 1 repaired 13/16 eligible failures. Cumulative success is 57/60
(95.00%), leaving three cases eligible for Round 2. The 16 Round-1 cases
contain 13 successes and three semantic failures, with no execution,
generation, protocol, or infrastructure failures. Per-level Round-1 repairs
are L2 3/3 and L3 10/13. All 16 per-task evaluation categories and codes match
Full PA-SCoRe's Round-1 records for the same tasks. Full's guard decisions were
`allow` for all 16 cases, and the disabled guard reported `allow` for all 16.
The diagnostic archive contains 16 unique Qwen task records; all 99 referenced
payload hashes passed verification, its source manifest hash matches the
Round-1 execution package, and three records carry feedback for Round 2.
Formal Round 1 source:
`experiments/packages/ablation-no-resource-guard-qwen-v1-diagnostics-round-1.zip`
(SHA-256 `35635c522276a75701bbf8d656fcca708aafd82ab6f3cf548d0154e487a887e0`).
No guard intervention occurred through Round 1 in this subset.

Round 2 repaired 1/3 remaining failures. Its three cases contain one success and two
semantic failures, with no execution, generation, protocol, or infrastructure
failures. Cumulative semantic success is 58/60 (96.67%): L1 13/13, L2 21/21,
and L3 24/26, identical to Full PA-SCoRe on the same frozen subset. The three
per-task Round-2 evaluation categories and codes match Full exactly; Full's
guard decisions were `allow` for all three. The diagnostic archive contains
three unique Qwen task records; all 18 referenced payload hashes passed
verification, its source manifest hash matches the Round-2 execution package,
and the three task IDs match the repair plan. Formal Round 2 source:
`experiments/packages/ablation-no-resource-guard-qwen-v1-diagnostics-round-2.zip`
(SHA-256 `aec0f58951542b3ee3f33c895041262c4efb2842dfb5202b7a9163a378c7b352`).

Across all three rounds on this fixed subset, Full never issued a blocking
resource-guard decision and the two conditions had the same per-task outcomes.
The observed paired accuracy difference is zero. This experiment cannot
establish an accuracy benefit or a lack of safety value for the guard; any
claim of demonstrated protective effect would require a separate,
pre-specified safety evaluation. The existing Pilot static check is an
implementation check, not evidence of reduced live resource consumption.

### C5. Paired Ablation Statistics

The fixed-model Qwen ablation uses the same 60 Held-out task IDs for every
condition (13 L1, 21 L2, 26 L3). Final cumulative semantic success and
unweighted level-macro accuracy are:

| Condition | L1 | L2 | L3 | Total | Macro | Delta from Full | Exact McNemar p | Holm p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full PA-SCoRe | 13/13 | 21/21 | 24/26 | 58/60 | 97.44% | - | - | - |
| Without Metadata Grounding | 0/13 | 0/21 | 0/26 | 0/60 | 0.00% | -96.67 pp | 6.94e-18 | 3.47e-17 |
| Without Failure Diagnosis | 13/13 | 20/21 | 13/26 | 46/60 | 81.75% | -20.00 pp | 0.000488 | 0.00195 |
| Without Structured Feedback | 13/13 | 20/21 | 22/26 | 55/60 | 93.28% | -5.00 pp | 0.25 | 0.75 |
| Without Resource-Aware Guard | 13/13 | 21/21 | 24/26 | 58/60 | 97.44% | 0.00 pp | 1.00 | 1.00 |
| Single-Round Repair | 13/13 | 21/21 | 23/26 | 57/60 | 96.15% | -1.67 pp | 1.00 | 1.00 |

The five exact two-sided paired McNemar tests compare each variant with Full;
Holm correction is over these five comparisons. Discordances favoring Full
are 58, 12, 3, 0, and 1 in table order, with none favoring a variant. The
tests and confidence language are exploratory because the Held-out results
have already been inspected during development. Metadata removal also removes
verified access-plan/API guidance. The guard ablation never reached a blocking
decision on this Qwen subset, and no accuracy effect is observed. Do not write
that every module improves accuracy. Four conceptual architecture modules do
not reduce the five recorded comparisons or alter their Holm correction. The
corresponding source archive hashes are recorded in B2 and C1-C4 above.

## D. Artifact Intake Checklist

Before changing any `PENDING` value:

1. Record run ID, split, condition, model aliases, and task count.
2. Verify package and diagnostics manifest hashes.
3. Confirm the frozen prompt, protocol, condition, evaluator, and model hashes.
4. Separate generation, infrastructure, execution, protocol, semantic failure,
   and success counts.
5. Confirm all expected task-model pairs are present exactly once.
6. Record per-level and per-subtype denominators.
7. Audit terminal failures before using model-failure percentages.
8. Store the source paths and SHA-256 values in this registry.
