#!/usr/bin/env python3
"""Execute packaged generated programs with policy, timeout, and logs."""

from __future__ import print_function

import argparse
import ast
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

try:
    from scripts.resource_aware_guard import guard_program
except ImportError:
    from resource_aware_guard import guard_program

try:
    import resource
except ImportError:
    resource = None


FORBIDDEN_IMPORTS = frozenset((
    "ctypes",
    "ftplib",
    "http",
    "paramiko",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "urllib",
))
FORBIDDEN_BUILTINS = frozenset(("__import__", "compile", "eval", "exec", "open"))
FORBIDDEN_ATTRIBUTE_PREFIXES = (
    "os.remove",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "os.system",
    "os.unlink",
    "pathlib.Path.unlink",
    "shutil.",
    "socket.",
    "subprocess.",
)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def audit_program_source(source):
    violations = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["syntax_error:{}:{}".format(error.lineno, error.msg)]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORTS:
                    violations.append("forbidden_import:{}".format(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS:
                violations.append("forbidden_import:{}".format(module))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                violations.append("forbidden_call:{}".format(node.func.id))
            elif isinstance(node.func, ast.Attribute):
                name = _attribute_name(node.func)
                if name and any(
                    name == prefix or name.startswith(prefix)
                    for prefix in FORBIDDEN_ATTRIBUTE_PREFIXES
                ):
                    violations.append("forbidden_call:{}".format(name))
    return sorted(set(violations))


def _atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_program_path(package_root, relative_path):
    normalized = os.path.normpath(relative_path)
    if os.path.isabs(normalized) or normalized.startswith(".." + os.sep):
        raise ValueError("unsafe program path: {}".format(relative_path))
    absolute = os.path.abspath(os.path.join(package_root, normalized))
    prefix = os.path.abspath(package_root) + os.sep
    if not absolute.startswith(prefix):
        raise ValueError("program path escapes package: {}".format(relative_path))
    return absolute


def _materialize_generation_failure(package_root, entry, output_root):
    model_alias = entry["model_alias"]
    task_id = entry["task_id"]
    source_path = _safe_program_path(package_root, entry["failure_path"])
    source_bytes = _read_bytes(source_path)
    actual_hash = _sha256_bytes(source_bytes)
    if actual_hash != entry.get("failure_sha256"):
        raise ValueError(
            "failure hash mismatch for {}/{}".format(model_alias, task_id)
        )
    failure = json.loads(source_bytes.decode("utf-8", errors="strict"))
    identity = failure.get("identity", {})
    if (
        failure.get("status") != "failed"
        or identity.get("model_alias") != model_alias
        or identity.get("task_id") != task_id
    ):
        raise ValueError(
            "generation failure binding mismatch for {}/{}".format(
                model_alias, task_id
            )
        )
    result = {
        "error_message": failure.get("error_message"),
        "error_type": failure.get("error_type"),
        "exit_status": None,
        "failure_code": entry["failure_code"],
        "failure_sha256": actual_hash,
        "generation_timed_out": entry["failure_code"] == "generation_timeout",
        "model_alias": model_alias,
        "status": "generation_failure",
        "task_id": task_id,
        "timed_out": False,
    }
    execution_path = os.path.join(
        output_root, model_alias, task_id, "execution.json"
    )
    _atomic_json(execution_path, result)
    return result


def _limit_resources(max_memory_mib, max_output_mib):
    if os.name != "posix" or resource is None:
        return None

    def apply_limits():
        os.setsid()
        memory_bytes = int(max_memory_mib) * 1024 * 1024
        output_bytes = int(max_output_mib) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))

    return apply_limits


def _read_text(path, limit_bytes):
    with open(path, "rb") as handle:
        value = handle.read(int(limit_bytes) + 1)
    truncated = len(value) > int(limit_bytes)
    value = value[:int(limit_bytes)]
    return value.decode("utf-8", errors="replace"), truncated


def _kill_process(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    process.kill()


def _run_entry(
    package_root,
    entry,
    output_root,
    timeout_seconds,
    python_executable,
    max_memory_mib,
    max_output_mib,
    task=None,
    guard_output_root=None,
    resource_guard_enabled=True,
):
    entry_type = entry.get("entry_type", "program")
    if entry_type == "generation_failure":
        return _materialize_generation_failure(
            package_root, entry, output_root
        )
    if entry_type != "program":
        raise ValueError("unknown manifest entry type: {}".format(entry_type))
    model_alias = entry["model_alias"]
    task_id = entry["task_id"]
    output_directory = os.path.join(output_root, model_alias, task_id)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    source_path = _safe_program_path(package_root, entry["program_path"])
    source_bytes = _read_bytes(source_path)
    actual_hash = _sha256_bytes(source_bytes)
    if actual_hash != entry.get("program_sha256"):
        raise ValueError("program hash mismatch for {}/{}".format(model_alias, task_id))
    source = source_bytes.decode("utf-8", errors="strict")
    violations = audit_program_source(source)
    execution_path = os.path.join(output_directory, "execution.json")
    stdout_path = os.path.join(output_directory, "stdout.log")
    stderr_path = os.path.join(output_directory, "stderr.log")
    program_copy = os.path.join(output_directory, "program.py")
    shutil.copyfile(source_path, program_copy)

    if violations:
        open(stdout_path, "wb").close()
        open(stderr_path, "wb").close()
        result = {
            "exit_status": None,
            "model_alias": model_alias,
            "policy_violations": violations,
            "program_sha256": actual_hash,
            "status": "policy_rejected",
            "task_id": task_id,
            "timed_out": False,
        }
        _atomic_json(execution_path, result)
        return result

    guard = None
    if task is not None:
        guard = guard_program(source, task, enabled=resource_guard_enabled)
        if guard_output_root:
            _atomic_json(
                os.path.join(
                    guard_output_root, model_alias, task_id, "guard.json"
                ),
                guard,
            )
        if guard["decision"] in ("reject", "require_bounded_access"):
            open(stdout_path, "wb").close()
            open(stderr_path, "wb").close()
            result = {
                "exit_status": None,
                "guard_decision": guard["decision"],
                "model_alias": model_alias,
                "policy_violations": [
                    "resource_guard:{}".format(item)
                    for item in guard.get("violations", [])
                ],
                "program_sha256": actual_hash,
                "status": "policy_rejected",
                "task_id": task_id,
                "timed_out": False,
            }
            _atomic_json(execution_path, result)
            return result

    started = time.time()
    timed_out = False
    with open(stdout_path, "wb") as stdout_stream, open(stderr_path, "wb") as stderr_stream:
        process = subprocess.Popen(
            [python_executable, program_copy],
            cwd=output_directory,
            stdout=stdout_stream,
            stderr=stderr_stream,
            preexec_fn=_limit_resources(max_memory_mib, max_output_mib),
        )
        try:
            exit_status = process.wait(timeout=float(timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process(process)
            exit_status = process.wait()
    elapsed = time.time() - started
    output_limit = int(max_output_mib) * 1024 * 1024
    stdout_text, stdout_truncated = _read_text(stdout_path, output_limit)
    stderr_text, stderr_truncated = _read_text(stderr_path, output_limit)
    reported_result = None
    stdout_json_valid = False
    try:
        candidate = json.loads(stdout_text.strip())
        if isinstance(candidate, dict):
            reported_result = candidate
            stdout_json_valid = True
    except ValueError:
        pass

    if timed_out:
        status = "timeout"
    elif (
        exit_status == 0
        and stdout_json_valid
        and str(reported_result.get("status", "")).strip().lower() == "success"
    ):
        status = "execution_success"
    else:
        status = "program_failure"
    result = {
        "elapsed_seconds": round(elapsed, 6),
        "exit_status": int(exit_status),
        "model_alias": model_alias,
        "policy_violations": [],
        "program_sha256": actual_hash,
        "reported_result": reported_result,
        "status": status,
        "stderr_bytes": os.path.getsize(stderr_path),
        "stderr_truncated": bool(stderr_truncated),
        "stdout_bytes": os.path.getsize(stdout_path),
        "stdout_json_valid": bool(stdout_json_valid),
        "stdout_truncated": bool(stdout_truncated),
        "task_id": task_id,
        "timed_out": bool(timed_out),
    }
    _atomic_json(execution_path, result)
    return result


def _completed_result(path, entry):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as handle:
            value = json.load(handle)
    except (IOError, OSError, ValueError):
        return None
    hash_field = (
        "failure_sha256"
        if entry.get("entry_type") == "generation_failure"
        else "program_sha256"
    )
    if value.get(hash_field) != entry.get(hash_field):
        return None
    return value


def execute_manifest(
    manifest_path,
    output_root,
    timeout_seconds=90,
    python_executable=None,
    max_memory_mib=8192,
    max_output_mib=2,
    resume=False,
    progress_fn=None,
    task_manifest_path=None,
    guard_output_root=None,
    resource_guard_enabled=True,
):
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest entries must be a list")
    package_root = os.path.dirname(os.path.abspath(manifest_path))
    task_by_id = {}
    if task_manifest_path:
        with open(task_manifest_path, "r") as handle:
            task_manifest = json.load(handle)
        tasks = task_manifest.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("task manifest tasks must be a list")
        task_by_id = dict((task["id"], task) for task in tasks)
        missing_tasks = sorted(set(
            entry["task_id"] for entry in entries
            if entry["task_id"] not in task_by_id
        ))
        if missing_tasks:
            raise ValueError(
                "task manifest is missing: {}".format(", ".join(missing_tasks))
            )
    python_executable = python_executable or sys.executable
    results = []
    for index, entry in enumerate(entries, 1):
        execution_path = os.path.join(
            output_root,
            entry["model_alias"],
            entry["task_id"],
            "execution.json",
        )
        existing = _completed_result(execution_path, entry) if resume else None
        if existing is not None:
            result = existing
            event = "execution_skipped"
        else:
            if progress_fn:
                progress_fn({
                    "event": "execution_start",
                    "index": index,
                    "model_alias": entry["model_alias"],
                    "task_id": entry["task_id"],
                    "total": len(entries),
                })
            result = _run_entry(
                package_root,
                entry,
                output_root,
                timeout_seconds,
                python_executable,
                max_memory_mib,
                max_output_mib,
                task=task_by_id.get(entry["task_id"]),
                guard_output_root=guard_output_root,
                resource_guard_enabled=resource_guard_enabled,
            )
            event = "execution_complete"
        results.append(result)
        if progress_fn:
            progress_fn({
                "event": event,
                "index": index,
                "model_alias": entry["model_alias"],
                "status": result["status"],
                "task_id": entry["task_id"],
                "total": len(entries),
            })
    statuses = (
        "generation_failure",
        "execution_success",
        "program_failure",
        "timeout",
        "policy_rejected",
    )
    summary = {
        "completed": len(results),
        "requested": len(entries),
    }
    for status in statuses:
        summary[status] = sum(result.get("status") == status for result in results)
    _atomic_json(os.path.join(output_root, "batch_summary.json"), summary)
    return summary


def _print_progress(event):
    print(json.dumps(event, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--max-memory-mib", type=int, default=8192)
    parser.add_argument("--max-output-mib", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task-manifest")
    parser.add_argument("--guard-output-dir")
    parser.add_argument("--disable-resource-guard", action="store_true")
    args = parser.parse_args(argv)
    summary = execute_manifest(
        args.manifest,
        args.output_dir,
        timeout_seconds=args.timeout_seconds,
        max_memory_mib=args.max_memory_mib,
        max_output_mib=args.max_output_mib,
        resume=args.resume,
        progress_fn=_print_progress,
        task_manifest_path=args.task_manifest,
        guard_output_root=args.guard_output_dir,
        resource_guard_enabled=not args.disable_resource_guard,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
