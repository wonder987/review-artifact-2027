# Anonymous Review Artifact for PA-SCoRe

This repository is an anonymized review artifact for the paper
"Beyond Executability: Failure-Aware Repair of LLM-Generated Scientific Data
Access Programs." It supports inspection of the PA-SCoRe implementation,
Gold-138 task specification, prompts, frozen protocols, evaluation logic, and
archived experiment evidence.

## Scope

The artifact contains:

- `scripts/`: generation, grounding, guard, diagnosis, feedback, repair,
  execution, evaluation, auditing, and Self-Debugging baseline code;
- `benchmark/`: the 138 task specifications, Pilot/Held-out/ablation splits,
  grounded prompts, and direct zero-shot prompts;
- `experiments/config/`: frozen protocols and experimental conditions;
- `tests/`: unit tests and synthetic fixtures;
- `evidence/audits/`: machine-readable result audits;
- `evidence/archives/`: formal diagnostic archives used for the reported
  Held-out, baseline, and ablation results;
- `evidence/self-debugging-prompts/`: the two-stage explanation and repair
  inputs used by the Self-Debugging baseline;
- `evidence/results_registry.md`: the manuscript-facing result registry.

The artifact does not contain API credentials, author information, raw
provider responses, Git history, or the underlying facility arrays. Executing
the live MDSplus tasks requires network access to a compatible deployment and
the separately maintained reference index. The included archives and audits
allow the reported outputs, denominators, and aggregate claims to be inspected
without that access.

## Environment

The orchestration and audit scripts were run with Python 3.9. Generated
programs were executed in a controlled Python 3.6 environment with MDSplus and
NumPy installed. Most offline tests use only the Python standard library; the
semantic reference helpers require NumPy.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

Live model calls are not required to inspect the archived results. If model
generation is repeated, create a provider configuration from
`experiments/config/models.example.json` and supply credentials only through environment
variables. Never store credentials in this artifact.

## Quick Inspection

Validate the task split and benchmark invariants:

```bash
python scripts/audit_gold138.py --help
python -m unittest tests.test_audit_gold138 -q
```

Inspect the principal quantitative claims in:

```text
evidence/results_registry.md
evidence/audits/pascore_heldout_full_v2_final_comparison.json
evidence/audits/baseline_ablation_reuse_v1.json
evidence/audits/table6_cost_latency_v1.json
evidence/self_debugging_summary.json
```

Formal per-round execution and evaluation records are stored in
`evidence/archives/`. Archive hashes are recorded in the result registry and
the corresponding audit files. Four archives contained absolute execution
paths in diagnostic text. Those paths were replaced in the review copies;
`evidence/archive_checksums.json` records both the immutable source hashes and
the hashes of the anonymized review copies.

## Reproduction Boundary

The offline portion of this artifact supports inspection and testing of task
construction, prompt rendering, controller behavior, failure diagnosis,
structured feedback, guarding, evaluation, and aggregation. End-to-end replay
requires three external resources that are not embedded here: model-provider
access, a compatible MDSplus deployment, and the Gold-138 reference index.
Consequently, this review package should be described as an inspection and
partial-reproduction artifact rather than as a self-contained replay of the
live experiment.

## Anonymity

This snapshot intentionally omits author names, affiliations, email addresses,
personal paths, acknowledgments, repository history, and identifying project
URLs. Please report any accidental identity disclosure through the anonymous
submission discussion rather than attempting to identify the authors.

## License During Review

This copy is supplied confidentially for peer review. No redistribution or
public-use license is granted by this review snapshot. A documented public
release is planned after acceptance.
