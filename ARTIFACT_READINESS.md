# Artifact Readiness

Status: PASS WITH DOCUMENTED EXTERNAL DEPENDENCIES

## Included Evidence

- PA-SCoRe and baseline implementation
- Gold-138 task manifest, splits, prompts, and access plans
- Frozen experimental protocols and ablation conditions
- Unit tests and synthetic evaluation fixtures
- Machine-readable audits and manuscript result registry
- Text-anonymized diagnostic archives for formal experiments
- Self-Debugging explanation and repair prompts

## Verification Performed

- `python -m unittest discover -s tests -q`: 290 tests passed
- no author name, affiliation email, personal home path, institutional model
  gateway, or literal API credential was detected in text files
- ZIP contents were scanned separately for the same identity and credential
  patterns
- every individual file is below the 8 MB limit of Anonymous GitHub
- no `.git` directory is included

## External Dependencies

End-to-end live replay requires model-provider access, a compatible MDSplus
deployment, and the separately maintained Gold-138 reference index. The
review artifact supports source inspection, offline unit testing, archive
inspection, and verification of aggregate result provenance without those
resources.

## Review-Phase Restriction

This snapshot is prepared only for anonymous peer review. It is not the final
public release and does not grant a redistribution license.
