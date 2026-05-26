# -*- coding: utf-8 -*-
"""
drug_ranking_adapter.py

Adapter layer to plug `06_drug_ranking_widget.ipynb` into the
Pediatric Neuro-Oncology Surgical Planning Agent.

It keeps the drug system optional:
- If artifacts are available, it produces preclinical drug sensitivity rankings.
- If artifacts / GitHub / intermediates are unavailable, it returns a safe fallback.
- Output is explicitly NOT a treatment recommendation.

Expected external repo from the original widget notebook:
    https://github.com/otonifrio2812/pediatric-bt-drug-prediction
Expected API in that repo:
    from src.drug_ranking import load_artifacts, list_cells_by_cancer_type, predict_drug_ranking
"""

from __future__ import annotations

import os
import re
import sys
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DRUG_RANKING_DISCLAIMER = (
    "AI preclinical drug-ranking module for research triage only. "
    "It is not a treatment recommendation, prescription, or substitute for "
    "pediatric neuro-oncology / pharmacy / molecular tumor board review."
)

DEFAULT_GITHUB_USER = "otonifrio2812"
DEFAULT_REPO_NAME = "pediatric-bt-drug-prediction"
DEFAULT_RELEASE_TAG = "v1.0.1"


# ---------------------------------------------------------------------------
# Cell-line preference tables (Cellosaurus-verified 2026-05-27)
# See README.md "Data provenance" section for full audit.
# ---------------------------------------------------------------------------

# Only pediatric CNS cell line in the model. Used as DMG/DIPG/H3K27M/pediatric-HGG surrogate.
# Cellosaurus CVCL_0378: 16 y/o male, anaplastic astrocytoma.
PEDIATRIC_HGG_CELL_LINE = "SIDM00607"  # KNS-42

# Adult GBM preference: T98G first (no identity disputes); U-87-MG deprioritized
# due to 2016 Uppsala/ATCC identity issue but kept as fallback.
ADULT_GBM_PREFERENCE = [
    "SIDM01171",  # T98G       — 61 y M GBM, p53 mutant, no identity dispute
    "SIDM01189",  # U-87-MG    — 44 y F; classic but 2016 ATCC identity flag
    "SIDM00684",  # LN-229     — 60 y F GBM
]

# Low-grade glioma surrogate (no pilocytic/LGG-specific line in model).
LGG_PREFERENCE = [
    "SIDM00666",  # Hs-683     — adult oligodendroglioma origin
    "SIDM00852",  # H4         — adult grade II neuroglioma
]

# Neuroblastoma default (industry reference; MYCN-amplified).
NEUROBLASTOMA_DEFAULT = "SIDM01009"  # KELLY

# Cancer types NOT covered by the model. Drug ranking is SKIPPED for these
# (no fallback to unrelated cancer type — clinical safety red line).
SKIP_CANCER_TYPES_KEYWORDS = {
    "medulloblastoma": "Medulloblastoma is not covered by the model (no MB cell line in GDSC2 panel used).",
    "ependymoma": "Ependymoma is not covered by the model (no EP cell line in GDSC2 panel used).",
}

# Cell lines excluded from prediction pool (data quality issues).
EXCLUDED_CELL_LINE_IDS = {
    # D-263MG: Cellosaurus CVCL_1154 flagged "Possibly misidentified" (sex chromosome
    # discrepancy). 34 other GBM lines remain — exclusion does not reduce coverage.
    "SIDM00732",
}


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    """Run a shell command safely from Python."""
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def _default_workdir() -> str:
    """Default workdir = <repo_root>/external (cross-platform; not /content)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "external"))


def setup_drug_ranking_repo(
    workdir: Optional[str] = None,
    github_user: str = DEFAULT_GITHUB_USER,
    repo_name: str = DEFAULT_REPO_NAME,
    release_tag: str = DEFAULT_RELEASE_TAG,
    install_requirements: bool = False,
    download_timeout: float = 120.0,
) -> str:
    """
    Cross-platform setup:
    - clone pediatric-bt-drug-prediction into <repo_root>/external/ (or `workdir`)
    - download intermediates.zip from GitHub Release using requests (no wget)
    - extract using stdlib zipfile (no system unzip)
    - optionally install requirements (default OFF: caller controls their env)

    Returns repo_dir.

    NOTE on numpy compatibility:
    The pickle artifacts were generated with numpy<2. Loading them under numpy 2.x
    may fail. Pin numpy<2 (project requirements.txt already does this).
    """
    if workdir is None:
        workdir = _default_workdir()

    workdir_path = Path(workdir)
    repo_dir = workdir_path / repo_name
    workdir_path.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        _run([
            "git", "clone", "--quiet",
            f"https://github.com/{github_user}/{repo_name}.git",
            str(repo_dir),
        ])

    if install_requirements and (repo_dir / "requirements.txt").exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=str(repo_dir))

    intermediates_dir = repo_dir / "intermediates"
    pkl_path = intermediates_dir / "stage6_ensemble_models.pkl"
    if not pkl_path.exists():
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        release_url = (
            f"https://github.com/{github_user}/{repo_name}/"
            f"releases/download/{release_tag}/intermediates.zip"
        )
        zip_path = repo_dir / "intermediates.zip"

        # Download via requests (cross-platform; replaces wget).
        try:
            import requests  # local import: only needed for setup
        except ImportError as exc:
            raise RuntimeError(
                "Setup requires `requests` (pip install requests). "
                f"Or pre-download {release_url} to {zip_path}."
            ) from exc

        print(f"Downloading {release_url} ...")
        try:
            with requests.get(release_url, stream=True, timeout=download_timeout) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {release_url}: {exc}. "
                f"Manual fix: download the zip and place at {zip_path}, then re-run."
            ) from exc

        # Extract via stdlib zipfile (cross-platform; replaces system unzip).
        import zipfile
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(intermediates_dir))

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    return str(repo_dir)


def _default_repo_dir() -> str:
    """Where this adapter expects the cloned repo to live by default."""
    return os.path.join(_default_workdir(), DEFAULT_REPO_NAME)


def ensure_numpy_pickle_compatibility() -> Dict[str, Any]:
    """
    Helper for Colab before loading the external pickle artifacts.
    Returns a dict; if needs_restart=True, restart runtime then run cells again.
    """
    import numpy as np

    version = np.__version__
    out = {
        "numpy_version": version,
        "needs_restart": False,
        "message": f"numpy={version} is compatible.",
    }

    if version.startswith("2"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0", "scipy<1.13"])
        out["needs_restart"] = True
        out["message"] = (
            f"Current numpy={version}. Downgraded to numpy<2.0. "
            "Please restart runtime/session, then rerun the setup cell."
        )
    return out


def _import_drug_api(repo_dir: Optional[str] = None):
    """Import the external repo's drug-ranking API.

    Bypasses src/__init__.py: the external package init eagerly imports `pdf_report`
    which requires `reportlab` — a heavy dep unrelated to drug-ranking inference.
    We load `src/drug_ranking.py` directly via importlib so the agent does not need
    PDF generation deps just to call predict_drug_ranking.
    """
    if repo_dir is None:
        repo_dir = _default_repo_dir()
    if repo_dir and os.path.isdir(repo_dir) and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    drug_ranking_path = os.path.join(repo_dir, "src", "drug_ranking.py")
    if not os.path.isfile(drug_ranking_path):
        raise ImportError(
            f"Could not find drug_ranking.py at {drug_ranking_path}. "
            "Run `python tools/drug_ranking_adapter.py --setup` to clone the repo "
            "and download model artifacts."
        )

    try:
        import importlib.util
        module_name = "_drug_ranking_external"
        # Reuse if already loaded (avoid re-executing module body across calls).
        if module_name in sys.modules:
            mod = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, drug_ranking_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"importlib could not build spec for {drug_ranking_path}")
            mod = importlib.util.module_from_spec(spec)
            # Register BEFORE exec_module: Python 3.11+ @dataclass looks up
            # cls.__module__ in sys.modules and crashes if absent.
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
        return mod.load_artifacts, mod.list_cells_by_cancer_type, mod.predict_drug_ranking
    except Exception as exc:
        raise ImportError(
            f"Could not load drug_ranking from {drug_ranking_path}: {exc}. "
            "Verify deps: pandas, numpy<2, joblib, scikit-learn, xgboost, pyarrow."
        ) from exc


def load_drug_ranking_artifacts(
    repo_dir: Optional[str] = None,
    intermediates_dir: Optional[str] = None,
):
    """Load external drug-ranking artifacts. Defaults to <repo_root>/external/.../intermediates/."""
    if repo_dir is None:
        repo_dir = _default_repo_dir()
    if intermediates_dir is None:
        intermediates_dir = os.path.join(repo_dir, "intermediates")
    load_artifacts, _, _ = _import_drug_api(repo_dir)
    return load_artifacts(intermediates_dir=intermediates_dir)


def list_available_cancer_types(artifacts, repo_dir: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
    """Return {cancer_type: [(cell_id, cell_name), ...]}."""
    _, list_cells_by_cancer_type, _ = _import_drug_api(repo_dir)
    return list_cells_by_cancer_type(artifacts)


def _case_freetext(structured_case: Dict[str, Any]) -> str:
    return " ".join([
        str(structured_case.get("tumor_type", "")),
        str(structured_case.get("pathology", "")),
        str(structured_case.get("diagnosis", "")),
        str(structured_case.get("tumor_location", "")),
        str(structured_case.get("imaging_description", "")),
    ]).lower()


# Sentinel cancer-type strings returned when the model does not cover the tumor.
# rank_drugs_for_case() detects these and returns a "cancer_type_not_in_model" result.
SKIP_SENTINEL_PREFIX = "__SKIP__"


def infer_cancer_type_from_case(
    structured_case: Dict[str, Any],
    available_types: List[str],
) -> str:
    """
    Heuristic mapping from clinical case text to the drug-ranking repo's cancer_type.

    Clinical safety contract:
    - Tumor types NOT covered by the model (medulloblastoma, ependymoma) return a
      "__SKIP__:<tumor_type>" sentinel. Caller MUST treat as skip — DO NOT fall back
      to an unrelated cancer type. Mapping MB or EP to Glioma is a clinical safety
      red line (decision committed 2026-05-27).
    - Only the 3 cancer types actually present in the model (Glioblastoma, Glioma,
      Neuroblastoma) are returned for live prediction.
    """
    text = _case_freetext(structured_case)
    available_lc = {x.lower(): x for x in available_types}

    def choose(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c.lower() in available_lc:
                return available_lc[c.lower()]
        return None

    # Hard-skip clauses FIRST (no Glioma fallback for these).
    for keyword, _ in SKIP_CANCER_TYPES_KEYWORDS.items():
        if keyword in text:
            return f"{SKIP_SENTINEL_PREFIX}:{keyword}"

    if "neuroblastoma" in text:
        picked = choose(["Neuroblastoma"])
        if picked:
            return picked

    if any(k in text for k in ["glioblastoma", "gbm"]):
        picked = choose(["Glioblastoma", "Glioma"])
        if picked:
            return picked

    if any(k in text for k in ["diffuse midline", "dmg", "dipg", "h3 k27", "h3k27"]):
        # DMG → must use Glioblastoma category (KNS-42 lives there); never fall back to Glioma.
        picked = choose(["Glioblastoma"])
        if picked:
            return picked

    # HGG identifiers → Glioblastoma category (where KNS-42 lives).
    # Must come BEFORE generic "glioma"/"astrocytoma" check so HGG cases aren't
    # routed to the LGG/Glioma surrogate path. Lets choose_surrogate_cell_line
    # trigger the pediatric branch (age<18 → KNS-42 PEDIATRIC-SURROGATE).
    if any(k in text for k in [
        "high-grade glioma", "high grade glioma", "hgg",
        "anaplastic astrocytoma", "anaplastic glioma",
        "who grade 3", "who grade 4", "grade 4",
    ]):
        picked = choose(["Glioblastoma"])
        if picked:
            return picked

    if any(k in text for k in ["glioma", "astrocytoma", "oligodendroglioma"]):
        # Default glioma → Glioma category first; choose_surrogate_cell_line picks Hs-683.
        picked = choose(["Glioma", "Glioblastoma"])
        if picked:
            return picked

    # Last-resort default for CNS demo (only if text gave us nothing).
    picked = choose(["Glioblastoma", "Glioma", "Neuroblastoma"])
    if picked:
        return picked

    return sorted(available_types)[0] if available_types else f"{SKIP_SENTINEL_PREFIX}:unknown"


def _case_markers_text(structured_case: Dict[str, Any]) -> str:
    markers = structured_case.get("molecular_markers") or {}
    if isinstance(markers, dict):
        return " ".join(f"{k} {v}" for k, v in markers.items()).lower()
    return str(markers).lower()


def _case_age(structured_case: Dict[str, Any]) -> int:
    try:
        age = int(structured_case.get("age"))
        return age if age >= 0 else -1
    except (TypeError, ValueError):
        return -1


def _pick_first_available(preferred_ids: List[str], cells_by_type_for_ct: List[Tuple[str, str]],
                          lookup: Dict[str, Dict[str, str]]) -> Tuple[Optional[str], Optional[str]]:
    """Walk preferred IDs in order; return first one that exists in the model and is not excluded."""
    for cid in preferred_ids:
        if cid in lookup and cid not in EXCLUDED_CELL_LINE_IDS:
            return cid, lookup[cid].get("CELL_LINE_NAME", cid)
    # Fallback to first non-excluded cell of that cancer type
    for cid, cname in cells_by_type_for_ct:
        if cid not in EXCLUDED_CELL_LINE_IDS:
            return cid, cname
    return None, None


def choose_surrogate_cell_line(
    structured_case: Dict[str, Any],
    artifacts,
    cells_by_type: Dict[str, List[Tuple[str, str]]],
    cancer_type: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Weighted cell-line selection using markers + age + tumor_location.

    Selection rules (Cellosaurus-verified 2026-05-27, audit-logged via `warnings`):
    - H3K27M / DMG / DIPG / pontine glioma  →  hard-prefer KNS-42 (SIDM00607) + OFF-DIST warning
    - pediatric age (<18) + Glioblastoma     →  prefer KNS-42 + PEDIATRIC-SURROGATE warning
    - adult Glioblastoma                     →  T98G > U-87-MG > LN-229
    - Glioma category (assume LGG-like)      →  Hs-683 > H4
    - Neuroblastoma                          →  KELLY (MYCN-amplified ref)
    - D-263MG (SIDM00732) excluded everywhere (Cellosaurus possibly_misidentified flag)
    - Explicit `cell_line_id` in case overrides ALL above (but still rejected if excluded)

    Returns (cell_line_id, cell_line_name, warnings-as-audit-log).
    """
    warnings: List[str] = []
    lookup = getattr(artifacts, "cell_metadata_lookup", {})

    # 1) Honor explicit cell_line_id if valid (and not excluded).
    requested = structured_case.get("cell_line_id") or structured_case.get("drug_cell_line_id")
    if requested:
        if requested in EXCLUDED_CELL_LINE_IDS:
            warnings.append(
                f"Requested cell_line_id={requested} is on exclusion list "
                "(Cellosaurus data-quality flag); ignoring and choosing surrogate."
            )
        elif requested in lookup:
            name = lookup[requested].get("CELL_LINE_NAME", requested)
            warnings.append(f"Selection rule: user-specified cell_line_id={requested} ({name}) accepted.")
            return requested, name, warnings
        else:
            warnings.append(f"Requested cell_line_id={requested} not found in model; choosing surrogate.")

    text = _case_freetext(structured_case)
    markers = _case_markers_text(structured_case)
    age = _case_age(structured_case)
    location = str(structured_case.get("tumor_location", "")).lower()
    cells_for_ct = cells_by_type.get(cancer_type, [])

    # 2) Weighted decision tree per cancer type.
    if cancer_type == "Glioblastoma":
        # H3K27M polarity check: must distinguish "h3k27m positive" from "h3k27m negative".
        # Look for h3k27[m] followed by a value token; treat negative qualifiers as NOT-DMG.
        _NEG_VALS = {"negative", "neg", "wild-type", "wildtype", "wt", "absent", "no", "not", "not_detected"}
        h3k27_match = re.search(r"h3\s*k27m?[\s:=,;]+(\S+)", markers)
        is_h3k27_positive_in_markers = False
        if h3k27_match:
            val = h3k27_match.group(1).strip(" -_/,.;").lower()
            is_h3k27_positive_in_markers = val not in _NEG_VALS
        # Text-based DMG indicators: clinical narrative phrasing implies positive context.
        is_dmg_in_text = any(k in text for k in [
            "diffuse midline", "dipg", "dmg",
            "h3 k27-altered", "h3k27-altered", "h3 k27 altered",
        ])
        is_dmg_marker = is_h3k27_positive_in_markers or is_dmg_in_text
        is_pontine = ("pons" in location or "pontine" in location or "brainstem" in location
                      or "pons" in text)
        is_pediatric = (0 <= age < 18)

        if is_dmg_marker or (is_pediatric and is_pontine):
            cid, name = _pick_first_available([PEDIATRIC_HGG_CELL_LINE], cells_for_ct, lookup)
            warnings.append(
                "OFF-DISTRIBUTION: no DIPG/DMG cell line exists in the model. "
                f"Using KNS-42 ({PEDIATRIC_HGG_CELL_LINE}, pediatric anaplastic astrocytoma) "
                "as the closest pediatric high-grade-glioma surrogate. "
                "Treat all rankings as HYPOTHESIS GENERATION ONLY."
            )
            warnings.append(
                "Selection rule: DMG/DIPG/H3K27M/pontine-glioma → hard-prefer KNS-42 "
                f"(age={age if age >= 0 else 'unknown'}, location='{location}', "
                f"markers='{markers or 'none'}')."
            )
            return cid, name, warnings

        if is_pediatric:
            cid, name = _pick_first_available([PEDIATRIC_HGG_CELL_LINE], cells_for_ct, lookup)
            warnings.append(
                f"PEDIATRIC-SURROGATE: case age={age} is pediatric but the model's only "
                f"pediatric CNS line is KNS-42 ({PEDIATRIC_HGG_CELL_LINE}, anaplastic astrocytoma). "
                "Selected as surrogate; other Glioblastoma lines are adult-derived."
            )
            warnings.append("Selection rule: pediatric age → KNS-42 (only pediatric CNS line in model).")
            return cid, name, warnings

        # Adult GBM
        cid, name = _pick_first_available(ADULT_GBM_PREFERENCE, cells_for_ct, lookup)
        warnings.append(
            f"Selection rule: adult glioblastoma → preferred {cid} ({name}). "
            "T98G selected first; U-87-MG deprioritized due to 2016 ATCC identity dispute. "
            "D-263MG excluded (Cellosaurus possibly-misidentified flag)."
        )
        return cid, name, warnings

    if cancer_type == "Glioma":
        cid, name = _pick_first_available(LGG_PREFERENCE, cells_for_ct, lookup)
        warnings.append(
            f"Selection rule: Glioma category → preferred {cid} ({name}) as LGG/low-grade proxy. "
            "Model has no pilocytic-astrocytoma or BRAF-specific line; all Glioma-category "
            "lines in the model are adult-derived (Cellosaurus-verified). "
            "Pediatric LGG rankings should be treated as surrogate only."
        )
        return cid, name, warnings

    if cancer_type == "Neuroblastoma":
        is_mycn = "mycn" in markers or "n-myc" in markers or "n_myc" in markers
        cid, name = _pick_first_available([NEUROBLASTOMA_DEFAULT], cells_for_ct, lookup)
        if is_mycn:
            warnings.append(
                f"Selection rule: Neuroblastoma + MYCN amplification → {cid} ({name}); "
                "KELLY is itself MYCN-amplified, used as reference for MYCN-amp NB."
            )
        else:
            warnings.append(
                f"Selection rule: Neuroblastoma default → {cid} ({name}); "
                "all 31 NB lines in the model are pediatric by tumor biology."
            )
        return cid, name, warnings

    # Unknown cancer_type — return first non-excluded available
    if cells_for_ct:
        cid, name = _pick_first_available([], cells_for_ct, lookup)
        warnings.append(
            f"No preference rule for cancer_type='{cancer_type}'; falling back to first "
            f"available cell line {cid} ({name}). Treat as low-confidence surrogate."
        )
        return cid, name, warnings

    warnings.append(f"No cell lines available for cancer_type='{cancer_type}'.")
    return None, None, warnings


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _records_from_ranking_df(ranking_df, top_k: int) -> List[Dict[str, Any]]:
    rows = []
    for _, r in ranking_df.head(top_k).iterrows():
        rows.append({
            "drug_name": str(r.get("drug_name", "")),
            "P_sens": _safe_float(r.get("P_sens")),
            "CI_lo": _safe_float(r.get("CI_lo")),
            "CI_hi": _safe_float(r.get("CI_hi")),
            "target": str(r.get("target", "")),
            "pathway": str(r.get("pathway", "")),
        })
    return rows


def _skip_result(reason_keyword: str, message: str) -> Dict[str, Any]:
    """Build a clean skip result for cancer types NOT covered by the model."""
    return {
        "status": "cancer_type_not_in_model",
        "skip_reason": reason_keyword,
        "error": message,
        "selected_cancer_type": None,
        "selected_cell_line_id": None,
        "selected_cell_line_name": None,
        "top_drugs": [],
        "warnings": [message],
        "disclaimer": DRUG_RANKING_DISCLAIMER,
    }


def rank_drugs_for_case(
    structured_case: Dict[str, Any],
    artifacts=None,
    repo_dir: Optional[str] = None,
    top_k: int = 10,
    with_ci: bool = True,
) -> Dict[str, Any]:
    """
    Main integration entry point.

    Returns a dict for case["drug_ranking_result"].

    Status semantics:
      - "ok"                      → top_drugs populated; ready for report
      - "cancer_type_not_in_model" → MB/EP/etc. skipped on purpose (clinical safety)
      - "unavailable"             → artifacts not installed / could not load (graceful)
      - "error"                   → unexpected exception in prediction

    The main agent flow MUST NOT raise from this function (graceful degradation contract).
    """
    structured_case = structured_case or {}

    # Pre-check: if the case obviously matches a skip-cancer-type, return EARLY
    # before attempting to load artifacts. This avoids unnecessary file IO and
    # makes the "MB/EP skipped" decision visible even when artifacts are missing.
    text_lc = _case_freetext(structured_case)
    for keyword, message in SKIP_CANCER_TYPES_KEYWORDS.items():
        if keyword in text_lc:
            return _skip_result(keyword, message)

    if artifacts is None:
        try:
            artifacts = load_drug_ranking_artifacts(repo_dir=repo_dir)
        except Exception as exc:
            return {
                "status": "unavailable",
                "error": str(exc),
                "selected_cancer_type": None,
                "selected_cell_line_id": None,
                "selected_cell_line_name": None,
                "top_drugs": [],
                "warnings": [
                    "Drug-ranking artifacts are unavailable; skipped optional drug-ranking module.",
                    "Run `python tools/drug_ranking_adapter.py --setup` to install.",
                ],
                "disclaimer": DRUG_RANKING_DISCLAIMER,
            }

    try:
        _, list_cells_by_cancer_type, predict_drug_ranking = _import_drug_api(repo_dir)
        cells_by_type = list_cells_by_cancer_type(artifacts)
        cancer_type = structured_case.get("drug_cancer_type") or infer_cancer_type_from_case(
            structured_case, list(cells_by_type.keys())
        )

        # Defensive: catch sentinel returned by infer_cancer_type_from_case.
        if isinstance(cancer_type, str) and cancer_type.startswith(SKIP_SENTINEL_PREFIX):
            keyword = cancer_type.split(":", 1)[1] if ":" in cancer_type else "unknown"
            return _skip_result(
                keyword,
                SKIP_CANCER_TYPES_KEYWORDS.get(
                    keyword,
                    f"Cancer type '{keyword}' is not covered by the model; drug-ranking skipped.",
                ),
            )

        cell_id, cell_name, warnings = choose_surrogate_cell_line(
            structured_case, artifacts, cells_by_type, cancer_type
        )

        if not cell_id:
            return {
                "status": "unavailable",
                "error": "No valid cell line could be selected.",
                "selected_cancer_type": cancer_type,
                "selected_cell_line_id": None,
                "selected_cell_line_name": None,
                "top_drugs": [],
                "warnings": warnings,
                "disclaimer": DRUG_RANKING_DISCLAIMER,
            }

        ranking = predict_drug_ranking(cell_id, artifacts, top_k=top_k, with_ci=with_ci)
        top_drugs = _records_from_ranking_df(ranking, top_k=top_k)

        return {
            "status": "ok",
            "selected_cancer_type": cancer_type,
            "selected_cell_line_id": cell_id,
            "selected_cell_line_name": cell_name,
            "top_drugs": top_drugs,
            "warnings": warnings,
            "disclaimer": DRUG_RANKING_DISCLAIMER,
            "source": "pediatric-bt-drug-prediction / 06_drug_ranking_widget adapter",
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "selected_cancer_type": None,
            "selected_cell_line_id": None,
            "selected_cell_line_name": None,
            "top_drugs": [],
            "warnings": [
                f"Drug-ranking module failed: {exc}. Continuing without drug-ranking output.",
            ],
            "disclaimer": DRUG_RANKING_DISCLAIMER,
        }


def drug_ranking_to_markdown(result: Dict[str, Any], title: str = "Preclinical Drug Ranking") -> str:
    """Convert drug-ranking result into a safe MDT-report markdown section.

    Always emits the four guardrail-required strings:
    `cell line`, `preclinical`, `hypothesis generation`, `not a treatment recommendation`.
    """
    result = result or {}
    status = result.get("status", "unknown")
    lines: List[str] = [f"## {title}", ""]

    # ---- Skip path: cancer type not covered by model (clinical-safety skip) ----
    if status == "cancer_type_not_in_model":
        skip_keyword = result.get("skip_reason", "unknown tumor type")
        lines.append(f"**Status:** ⛔ **Skipped — preclinical drug-ranking model does not cover `{skip_keyword}`.**")
        lines.append("")
        lines.append(f"**Reason:** {result.get('error', '')}")
        lines.append("")
        lines.append(
            "**Clinical-safety rationale:** Mapping this tumor type to an unrelated "
            "cell line (e.g. Glioma) would generate misleading drug rankings. The "
            "preclinical model is intentionally skipped — drug selection for this tumor "
            "requires MTB / pharmacy review with condition-specific evidence."
        )
        lines.append("")
        lines.append(f"**Safety note:** {DRUG_RANKING_DISCLAIMER}")
        lines.append("> This decision is hypothesis generation and triage support only — "
                     "not a treatment recommendation. Cell line predictions are preclinical surrogates.")
        return "\n".join(lines)

    # ---- Unavailable / error path ----
    if status != "ok":
        lines.append(f"**Status:** `{status}`")
        lines.append(f"**Reason:** {result.get('error', 'not available')}")
        for w in result.get("warnings", []):
            lines.append(f"- {w}")
        lines.append("")
        lines.append(f"**Safety note:** {DRUG_RANKING_DISCLAIMER}")
        lines.append("> Preclinical cell line drug-ranking module unavailable; "
                     "this is not a treatment recommendation. Hypothesis generation only.")
        return "\n".join(lines)

    # ---- Success path ----
    warnings_list = result.get("warnings", []) or []
    off_dist = [w for w in warnings_list if "OFF-DISTRIBUTION" in w.upper()]
    ped_surr = [w for w in warnings_list if "PEDIATRIC-SURROGATE" in w.upper()]
    rule_lines = [w for w in warnings_list if w.startswith("Selection rule")]
    other_w = [w for w in warnings_list if w not in off_dist + ped_surr + rule_lines]

    # Prominent banner for OFF-DISTRIBUTION (KNS-42 used for DMG/DIPG).
    if off_dist:
        lines.append("> ## ⚠️ OFF-DISTRIBUTION WARNING ⚠️")
        for w in off_dist:
            lines.append(f"> {w}")
        lines.append("")
    elif ped_surr:
        lines.append("> ## ⚠️ PEDIATRIC-SURROGATE WARNING")
        for w in ped_surr:
            lines.append(f"> {w}")
        lines.append("")

    lines.append(f"**Safety note:** {DRUG_RANKING_DISCLAIMER}")
    lines.append("")

    lines.append(f"- Selected cancer type: `{result.get('selected_cancer_type')}`")
    lines.append(
        f"- Surrogate cell line: `{result.get('selected_cell_line_id')}` "
        f"({result.get('selected_cell_line_name')})"
    )

    if rule_lines:
        lines.append("- Selection audit trail:")
        for w in rule_lines:
            lines.append(f"  - {w}")
    for w in other_w:
        lines.append(f"- ℹ️ {w}")

    lines.append("")
    lines.append("| Rank | Drug | P_sens | 95% CI | Target | Pathway |")
    lines.append("|---:|---|---:|---|---|---|")
    for i, d in enumerate(result.get("top_drugs", []), start=1):
        ps = d.get("P_sens")
        lo = d.get("CI_lo")
        hi = d.get("CI_hi")
        ps_s = "" if ps is None else f"{ps:.3f}"
        ci_s = "" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"
        lines.append(
            f"| {i} | {d.get('drug_name','')} | {ps_s} | {ci_s} | "
            f"{d.get('target','')} | {d.get('pathway','')} |"
        )

    lines.append("")
    lines.append(
        "> This table is preclinical cell line output, intended for **hypothesis generation** "
        "and trial discussion only. It is **not a treatment recommendation**. "
        "It must pass medical guardrails and MDT review before being shown as a clinical-facing appendix."
    )
    return "\n".join(lines)


def launch_drug_ranking_widget(artifacts, repo_dir: Optional[str] = None):
    """
    Interactive Colab/Jupyter widget copied from 06_drug_ranking_widget.ipynb,
    wrapped as a reusable function.
    """
    _, list_cells_by_cancer_type, predict_drug_ranking = _import_drug_api(repo_dir)
    cells_by_type = list_cells_by_cancer_type(artifacts)

    import ipywidgets as widgets
    from IPython.display import clear_output, display

    cancer_dropdown = widgets.Dropdown(
        options=sorted(cells_by_type.keys()),
        value="Glioblastoma" if "Glioblastoma" in cells_by_type else sorted(cells_by_type.keys())[0],
        description="Cancer:",
    )

    initial_cells = cells_by_type[cancer_dropdown.value]
    cell_dropdown = widgets.Dropdown(
        options=[f"{cid} ({name})" for cid, name in initial_cells],
        description="Cell:",
    )

    top_k_slider = widgets.IntSlider(value=10, min=5, max=30, step=5, description="Top K:")
    output = widgets.Output()

    def on_cancer_change(change):
        ctype = change["new"]
        cell_dropdown.options = [f"{cid} ({name})" for cid, name in cells_by_type[ctype]]

    def update_ranking(*args):
        cell_id = cell_dropdown.value.split(" ")[0]
        top_k = top_k_slider.value
        with output:
            clear_output()
            try:
                ranking = predict_drug_ranking(cell_id, artifacts, top_k=top_k, with_ci=True)
                meta = artifacts.cell_metadata_lookup[cell_id]
                print("=" * 70)
                print(f"Cell: {cell_id} ({meta['CELL_LINE_NAME']})")
                print(f"Cancer type: {meta['CANCER_TYPE']}")
                print("=" * 70)
                display_df = ranking.copy()
                display_df["P_sens"] = display_df["P_sens"].apply(lambda x: f"{x:.3f}")
                display_df["CI"] = display_df.apply(
                    lambda r: f"[{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]", axis=1
                )
                display_df["target"] = display_df["target"].apply(
                    lambda t: str(t)[:25] + ".." if len(str(t)) > 27 else str(t)
                )
                display_df["pathway"] = display_df["pathway"].apply(
                    lambda p: str(p)[:25] + ".." if len(str(p)) > 27 else str(p)
                )
                display(display_df[["drug_name", "P_sens", "CI", "target", "pathway"]])
                print("\nSafety note:", DRUG_RANKING_DISCLAIMER)
            except Exception as exc:
                print(f"Error: {exc}")

    cancer_dropdown.observe(on_cancer_change, names="value")
    cancer_dropdown.observe(update_ranking, names="value")
    cell_dropdown.observe(update_ranking, names="value")
    top_k_slider.observe(update_ranking, names="value")

    ui = widgets.VBox([cancer_dropdown, cell_dropdown, top_k_slider, output])
    display(ui)
    update_ranking()
    return ui


def attach_drug_ranking_to_case(
    structured_case: Dict[str, Any],
    artifacts=None,
    repo_dir: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Minimal helper for agent.py integration."""
    case = dict(structured_case or {})
    result = rank_drugs_for_case(case, artifacts=artifacts, repo_dir=repo_dir, top_k=top_k)
    case["drug_ranking_result"] = result
    case["drug_ranking_report_section"] = drug_ranking_to_markdown(result)
    return case


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Pediatric brain-tumor drug-ranking adapter (Nemotron agent)."
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Clone the external model repo + download artifacts to <repo_root>/external/",
    )
    parser.add_argument(
        "--list-cells", action="store_true",
        help="List all cell lines in the model, grouped by cancer type. "
             "Marks D-263MG as [EXCLUDED] and KNS-42 as [PEDIATRIC].",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a demo prediction on a DMG sample case.",
    )
    args = parser.parse_args()

    if args.setup:
        repo_dir = setup_drug_ranking_repo()
        intermediates = Path(repo_dir) / "intermediates"
        n_files = sum(1 for _ in intermediates.glob("*")) if intermediates.exists() else 0
        print(f"Setup complete: {repo_dir}")
        print(f"  intermediates: {intermediates}  ({n_files} files)")
        sys.exit(0)

    if args.list_cells:
        try:
            artifacts = load_drug_ranking_artifacts()
            cells_by_type = list_available_cancer_types(artifacts)
        except Exception as exc:
            print(f"ERROR: Could not load artifacts: {exc}")
            print("Run `python tools/drug_ranking_adapter.py --setup` first.")
            sys.exit(1)

        total = 0
        print(f"Cancer types in model: {len(cells_by_type)}\n")
        for ct in sorted(cells_by_type):
            cells = sorted(cells_by_type[ct], key=lambda x: x[1])
            usable = sum(1 for cid, _ in cells if cid not in EXCLUDED_CELL_LINE_IDS)
            print(f"=== {ct}  (total {len(cells)}, usable {usable}) ===")
            for cid, cname in cells:
                tags = []
                if cid in EXCLUDED_CELL_LINE_IDS:
                    tags.append("EXCLUDED")
                if cid == PEDIATRIC_HGG_CELL_LINE:
                    tags.append("PEDIATRIC")
                if cid in ADULT_GBM_PREFERENCE:
                    tags.append(f"GBM-PREF#{ADULT_GBM_PREFERENCE.index(cid)+1}")
                if cid in LGG_PREFERENCE:
                    tags.append(f"LGG-PREF#{LGG_PREFERENCE.index(cid)+1}")
                if cid == NEUROBLASTOMA_DEFAULT:
                    tags.append("NB-DEFAULT")
                tag_s = "  [" + ", ".join(tags) + "]" if tags else ""
                print(f"  {cid:<12} {cname}{tag_s}")
            print()
            total += len(cells)
        print(f"Total cell lines: {total}")
        print(f"Excluded from prediction pool: {sorted(EXCLUDED_CELL_LINE_IDS)}")
        print(f"Cancer types SKIPPED (not in model): {sorted(SKIP_CANCER_TYPES_KEYWORDS)}")
        sys.exit(0)

    if args.demo:
        case = {
            "age": 8,
            "tumor_type": "diffuse midline glioma",
            "pathology": "H3 K27-altered diffuse midline glioma",
            "tumor_location": "pons",
            "molecular_markers": {"H3K27M": "positive/suspected"},
        }
        print(json.dumps(rank_drugs_for_case(case, top_k=5), ensure_ascii=False, indent=2))
        sys.exit(0)

    parser.print_help()
