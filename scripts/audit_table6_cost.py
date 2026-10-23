"""Audit per-task token and stage-time costs for the held-out comparison."""

import argparse
import csv
import hashlib
import json
import statistics
import zipfile
from pathlib import Path


MODELS = (
    ("qwen3.8-chat", "Qwen 3.8 Chat"),
    ("deepseek-v4-flash-ascend1", "DeepSeek V4 Flash Ascend1"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite"),
)
METRICS = (
    "model_calls", "input_tokens", "output_tokens",
    "generation_seconds", "execution_seconds", "repairs_used",
)
LAUNCHED = {"execution_success", "program_failure", "timeout"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("invalid {}: {!r}".format(label, value))
    return value


def case_cost(key, phases):
    """Sum attempted rounds for one model/task; retain missing calls as missing."""
    totals = {name: 0 for name in METRICS}
    totals["generation_seconds"] = 0.0
    totals["execution_seconds"] = 0.0
    missing = []
    statuses = []
    attempts = 0
    for round_index, (generation, execution) in enumerate(phases):
        if key not in execution:
            continue
        record = execution[key]
        status = record["status"]
        statuses.append(status)
        totals["model_calls"] += 1
        generated = generation.get(key)
        if generated is None:
            if status != "generation_failure":
                raise ValueError("missing generation result for {} round {}".format(key, round_index))
            missing.append(round_index)
        elif status == "generation_failure":
            raise ValueError("generation failure has a result for {} round {}".format(key, round_index))
        else:
            try:
                for name in ("input_tokens", "output_tokens", "generation_seconds"):
                    totals[name] += _number(generated[name], name)
            except (KeyError, TypeError, ValueError):
                missing.append(round_index)
            if generated.get("attempts") is not None:
                attempts += int(_number(generated["attempts"], "attempts"))
        if status in LAUNCHED:
            try:
                totals["execution_seconds"] += _number(record["elapsed_seconds"], "execution_seconds")
            except (KeyError, TypeError, ValueError):
                missing.append(round_index)
        elif status not in {"generation_failure", "policy_rejected"}:
            raise ValueError("unexpected execution status: {}".format(status))
    if not statuses:
        raise ValueError("no execution records for {}".format(key))
    totals["repairs_used"] = totals["model_calls"] - 1
    complete = not missing
    return {
        "model": key[0], "task_id": key[1], "complete": complete,
        "missing_rounds": sorted(set(missing)), "statuses": statuses,
        "http_attempts": attempts if complete else None,
        **{name: totals[name] if complete else None for name in METRICS},
    }


def paired_summary(baseline, full):
    if set(baseline) != set(full):
        raise ValueError("baseline and full task-model sets differ")
    retained = sorted(key for key in baseline if baseline[key]["complete"] and full[key]["complete"])
    excluded = sorted(set(baseline) - set(retained))
    if not retained:
        raise ValueError("no complete paired cases")

    def row(cases):
        return {
            "mean_{}".format(metric): statistics.mean(cases[key][metric] for key in retained)
            for metric in METRICS
        }

    return {"n": len(retained), "excluded_cases": excluded,
            "direct": row(baseline), "full": row(full)}


def _load_generation(directory):
    records = {}
    for source in directory.rglob("result.json"):
        parts = source.relative_to(directory).parts
        if len(parts) != 4 or not parts[2].startswith("attempt-"):
            raise ValueError("unexpected generation path: {}".format(source))
        key = (parts[0], parts[1])
        if key in records:
            raise ValueError("duplicate generation result: {}".format(key))
        result = json.loads(source.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            raise ValueError("not a completed result: {}".format(source))
        identity = result["identity"]
        if identity["model_alias"] != key[0] or identity["task_id"] != key[1]:
            raise ValueError("generation identity mismatch: {}".format(source))
        usage = result["normalized"].get("usage") or {}
        transport = result.get("transport") or {}
        records[key] = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "generation_seconds": transport.get("elapsed_seconds"),
            "attempts": transport.get("attempts"),
        }
    return records


def _load_execution(archive):
    records = {}
    with zipfile.ZipFile(archive) as package:
        corrupt = package.testzip()
        if corrupt:
            raise ValueError("corrupt ZIP member: {}".format(corrupt))
        for name in package.namelist():
            if not name.endswith("/execution.json"):
                continue
            parts = name.split("/")
            if len(parts) != 4 or parts[0] != "execution":
                raise ValueError("unexpected execution path: {}".format(name))
            key = (parts[1], parts[2])
            if key in records:
                raise ValueError("duplicate execution: {}".format(key))
            record = json.loads(package.read(name))
            if record["model_alias"] != key[0] or record["task_id"] != key[1]:
                raise ValueError("execution identity mismatch: {}".format(name))
            records[key] = record
    return records


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fmt(value):
    return "{:.2f}".format(value)


def _latex_table(summaries):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Observed per-task model usage and stage time on matched Held-out cases. Values are means within each model. API time includes network and retries; execution time covers launched programs only. Incomplete API requests are excluded from both paired rows for that model.}",
        r"\label{tab:cost}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Method & $n$ & Calls & Input tokens & Output tokens & API time (s) & Exec. time (s) \\",
        r"\midrule",
    ]
    for index, (alias, label) in enumerate(MODELS):
        summary = summaries[alias]
        for method, text in (("direct", "Direct Zero-shot"), ("full", "Full PA-SCoRe")):
            row = summary[method]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    label if method == "direct" else "",
                    text, summary["n"], _fmt(row["mean_model_calls"]),
                    _fmt(row["mean_input_tokens"]), _fmt(row["mean_output_tokens"]),
                    _fmt(row["mean_generation_seconds"]),
                    _fmt(row["mean_execution_seconds"]),
                )
            )
        if index < len(MODELS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def audit(repo):
    results = repo / "experiments/results/held-out"
    packages = repo / "experiments/packages"
    archives = [packages / "direct-zero-shot-heldout-v2-diagnostics.zip"] + [
        packages / "pascore-heldout-full-v2-diagnostics-round-{}.zip".format(i)
        for i in range(3)
    ]
    execution = [_load_execution(path) for path in archives]
    baseline_keys = set(execution[0])
    if len(baseline_keys) != 324 or set(execution[1]) != baseline_keys:
        raise ValueError("baseline and Full Round-0 must contain the same 324 cases")
    if len(execution[2]) != 127 or len(execution[3]) != 27:
        raise ValueError("unexpected repair-round execution counts")
    if not set(execution[2]).issubset(baseline_keys) or not set(execution[3]).issubset(set(execution[2])):
        raise ValueError("repair cohort is not nested")
    generation = [
        _load_generation(results / "primary-zero-shot-v2")
    ] + [
        _load_generation(results / "pascore-v2/full/round-{}/generation".format(i))
        for i in range(3)
    ]
    if any(not set(g).issubset(set(e)) for g, e in zip(generation, execution)):
        raise ValueError("generation record has no execution counterpart")
    phases = list(zip(generation, execution))
    cases = {
        "direct": {key: case_cost(key, phases[:1]) for key in baseline_keys},
        "full": {key: case_cost(key, phases[1:]) for key in baseline_keys},
    }
    summaries = {}
    for alias, _ in MODELS:
        selected = {method: {key: value for key, value in rows.items() if key[0] == alias}
                    for method, rows in cases.items()}
        if len(selected["direct"]) != 108:
            raise ValueError("model does not have 108 task pairs: {}".format(alias))
        summaries[alias] = paired_summary(selected["direct"], selected["full"])
    source_hashes = {str(path.relative_to(repo)).replace("\\", "/"): _sha256(path)
                     for path in archives}
    return cases, summaries, source_hashes, [len(g) for g in generation]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    cases, summaries, hashes, generation_counts = audit(repo)
    out = repo / "experiments/audits"
    out.mkdir(parents=True, exist_ok=True)
    case_file = out / "table6_cost_latency_cases_v1.csv"
    with case_file.open("w", newline="", encoding="utf-8") as target:
        fields = ["condition", "model", "task_id", "complete", "missing_rounds", "statuses",
                  "http_attempts", *METRICS]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for condition, records in cases.items():
            for key in sorted(records):
                record = dict(records[key])
                record["condition"] = condition
                record["missing_rounds"] = ",".join(map(str, record["missing_rounds"]))
                record["statuses"] = ",".join(record["statuses"])
                writer.writerow(record)
    payload = {
        "scope": "Held-out 108 tasks x 3 models, matched complete-case by model",
        "metrics": "Per-task sum over attempted rounds; API transport and sandbox subprocess times are separate",
        "missing_policy": "Missing provider usage or timed-out API request excludes the task from both paired rows for its model",
        "generation_record_counts": generation_counts,
        "execution_record_counts": [324, 324, 127, 27],
        "source_zip_sha256": hashes,
        "paired_by_model": summaries,
        "case_csv_sha256": _sha256(case_file),
    }
    summary_file = out / "table6_cost_latency_v1.json"
    summary_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_file = repo / "paper/tables/table6_cost_latency.tex"
    table_file.parent.mkdir(parents=True, exist_ok=True)
    table_file.write_text(_latex_table(summaries), encoding="ascii")
    standalone = out / "table6_cost_latency_compile_check.tex"
    standalone.write_text(
        "\\documentclass{article}\n\\usepackage{booktabs}\n"
        "\\begin{document}\n" + _latex_table(summaries) + "\\end{document}\n",
        encoding="ascii",
    )
    print(json.dumps({"summary": str(summary_file), "cases": str(case_file),
                      "latex_table": str(table_file), "paired_by_model": summaries}, indent=2))


if __name__ == "__main__":
    main()
