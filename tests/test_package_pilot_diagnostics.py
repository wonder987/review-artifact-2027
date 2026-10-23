import hashlib
import json
import os
import tempfile
import unittest
import zipfile

from scripts.package_pilot_diagnostics import build_package


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


class PilotDiagnosticsPackageTests(unittest.TestCase):
    def build_fixture(self, root):
        files = {}
        for name in (
            "semantic_evaluator.py",
            "evaluate_generated_batch.py",
            "failure_taxonomy.py",
            "analyze_pilot_failures.py",
        ):
            path = os.path.join(root, name)
            with open(path, "wb") as handle:
                handle.write(("# " + name + "\n").encode("ascii"))
            files[name] = path
        taxonomy_path = os.path.join(root, "failure_taxonomy_v0.json")
        with open(taxonomy_path, "w") as handle:
            json.dump({"taxonomy_version": "test-v0"}, handle)
        return files, taxonomy_path

    def test_package_is_byte_deterministic_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as root:
            runtime_files, taxonomy_path = self.build_fixture(root)
            first = os.path.join(root, "first.zip")
            second = os.path.join(root, "second.zip")
            first_manifest = build_package(first, runtime_files, taxonomy_path)
            second_manifest = build_package(second, runtime_files, taxonomy_path)
            with open(first, "rb") as handle:
                first_bytes = handle.read()
            with open(second, "rb") as handle:
                second_bytes = handle.read()
            with zipfile.ZipFile(first, "r") as archive:
                names = sorted(archive.namelist())
                embedded_manifest = json.loads(archive.read("manifest.json"))
                embedded = dict(
                    (name, archive.read(name)) for name in names
                    if name != "manifest.json"
                )

        self.assertEqual(first_manifest, second_manifest)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual([
            "analyze_pilot_failures.py",
            "evaluate_generated_batch.py",
            "failure_taxonomy.py",
            "failure_taxonomy_v0.json",
            "manifest.json",
            "semantic_evaluator.py",
        ], names)
        self.assertEqual(first_manifest, embedded_manifest)
        for name, value in embedded.items():
            self.assertEqual(
                sha256_bytes(value), first_manifest["runtime_sha256"][name]
            )

    def test_package_excludes_experiment_evidence_and_credentials(self):
        with tempfile.TemporaryDirectory() as root:
            runtime_files, taxonomy_path = self.build_fixture(root)
            output = os.path.join(root, "tools.zip")
            build_package(output, runtime_files, taxonomy_path)
            with zipfile.ZipFile(output, "r") as archive:
                names = archive.namelist()
                manifest = json.loads(archive.read("manifest.json"))

        lowered_names = "\n".join(names).lower()
        for token in (
            "request", "response", "programs/", "credential", "api_key",
            "gold", "execution.json",
        ):
            self.assertNotIn(token, lowered_names)
        self.assertNotIn("source_paths", manifest)
        self.assertEqual(5, manifest["runtime_file_count"])

    def test_package_preserves_safe_taxonomy_basename(self):
        with tempfile.TemporaryDirectory() as root:
            runtime_files, taxonomy_path = self.build_fixture(root)
            v1_path = os.path.join(root, "failure_taxonomy_v1.json")
            os.replace(taxonomy_path, v1_path)
            output = os.path.join(root, "tools-v1.zip")
            build_package(output, runtime_files, v1_path)
            with zipfile.ZipFile(output, "r") as archive:
                names = archive.namelist()

        self.assertIn("failure_taxonomy_v1.json", names)
        self.assertNotIn("failure_taxonomy_v0.json", names)

    def test_rejects_missing_or_unexpected_runtime_names(self):
        with tempfile.TemporaryDirectory() as root:
            runtime_files, taxonomy_path = self.build_fixture(root)
            runtime_files.pop("failure_taxonomy.py")
            with self.assertRaises(ValueError):
                build_package(
                    os.path.join(root, "missing.zip"),
                    runtime_files,
                    taxonomy_path,
                )


if __name__ == "__main__":
    unittest.main()
