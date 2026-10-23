# PA-SCoRe Pilot v2 Round 2 Terminal Audit

Audit date: 2026-09-28

## Package integrity

- Run ID: `pascore-pilot-full-v2`
- Round: `2`
- Round package manifest SHA-256: `dc8cc700c490096788f18c278d3b2f0fee43a765ab7b19b740b8b03197fe2174`
- Diagnostics source manifest SHA-256: `dc8cc700c490096788f18c278d3b2f0fee43a765ab7b19b740b8b03197fe2174`
- Integrity result: match
- Diagnostics entries: 12
- Round 2 semantic successes: 4
- Round 2 terminal failures: 8

## Cumulative pilot result

- Round 0: 59/90 semantic successes
- Round 1: 19/31 repaired
- Round 2: 4/12 repaired
- Final: 82/90 semantic successes (91.11%)
- Infrastructure failures: 0
- Protocol failures in Round 2: 0

## Terminal failure audit

| Model | Task | Final code | Direct evidence | Verdict |
| --- | --- | --- | --- | --- |
| DeepSeek | `gold138_l3_038` | `signal_dtype_mismatch` | Raw signals are converted to `float64` before target summaries and hashes. | Model repair failure |
| DeepSeek | `gold138_l3_040` | `runtime_error` | The program stores signal/time summaries, then passes those dictionaries to NumPy alignment as if they were arrays. | Model repair failure |
| DeepSeek | `gold138_l3_058` | `signal_dtype_mismatch` | Both retrieved raw signals are converted to `float64` before target summaries and hashes. | Model repair failure |
| Gemini | `gold138_l3_017` | `resource_decision_mismatch` | The program substitutes the 10,001-point preview and recomputed memory for disclosed full-target metadata (10,680,000 points; 122.222900390625 MiB). | Model repair failure |
| Gemini | `gold138_l3_019` | `resource_decision_mismatch` | The program substitutes 10,001 points and 0.15 MiB for disclosed full-target metadata (21,000,000 points; 240.325927734375 MiB). | Model repair failure |
| Gemini | `gold138_l3_056` | `signal_dtype_mismatch` | Retrieved signals are converted to contiguous `float64` arrays before target summaries and hashes. | Model repair failure |
| Qwen | `gold138_l3_017` | `signal_size_mismatch` | The summary reports `arr.nbytes` as `size`, yielding 40,004 and 80,008 instead of 10,001 points. | Model repair failure |
| Qwen | `gold138_l3_056` | `signal_dtype_mismatch` | Retrieved signals are converted to `float64` before target summaries and hashes. | Model repair failure |

## Audit conclusion

All eight terminal failures are supported by generated-program evidence. None is attributable to network failure, unavailable MDSplus infrastructure, missing execution output, or an evaluator defect. Exact dtype and hash checks are retained because the task contract requires faithful summaries of raw retrieved arrays; aligned arrays may be `float64`, but raw target summaries may not silently change dtype.

The PA-SCoRe Pilot v2 protocol is eligible for freezing. The held-out run must use the same three models, metadata contract, resource budget, strict evaluator, maximum of two repair rounds, and failure-to-feedback mapping. No Pilot-derived task-specific fix may be added after the held-out run begins.
