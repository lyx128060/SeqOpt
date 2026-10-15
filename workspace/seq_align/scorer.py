"""Pure scorer module (no Harness, no SLURM).

Parses a jackhmmer/hmmsearch ``--tblout`` file and scores remote homology
against SCOP ground truth. The scoring logic is NOT re-implemented here: it
reuses the exact lineage loading + scoring used by the existing pipeline
(run_jack._load_scop_lineages / _score_scop_names / _lookup_scop) so the
numbers stay identical to the shell/SLURM path. This module only adds the
thin, testable glue: tblout parsing and a flat metric dict.

Field semantics (same-superfamily positive / different-fold negative):
    TP  : target in same superfamily (first 3 SCCS parts) as query
    FP  : target in a different fold (first 2 SCCS parts differ)
    ignored : same fold but different superfamily
    unclassified : target (or query) has no SCOP ground truth
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .run_jack import _load_scop_lineages, _lookup_scop, _score_scop_names

__all__ = [
    "parse_tblout",
    "load_scop",
    "lookup_scop",
    "score_hits",
    "evaluate_remote_homology",
]


def parse_tblout(tblout_path: str) -> List[str]:
    """Parse an hmmsearch/jackhmmer --tblout file into target name strings.

    Skips blank lines and ``#`` comment lines (including the trailing
    ``# Program: ...`` footer). Returns the first whitespace token of each
    data line (the target identifier).
    """
    targets: List[str] = []
    if not os.path.exists(tblout_path):
        return targets
    with open(tblout_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if parts:
                targets.append(parts[0])
    return targets


def load_scop(scop_annot_path: str) -> Dict[str, str]:
    """Return domain_id -> complete SCOP SCCS lineage map (reuses run_jack)."""
    return _load_scop_lineages(scop_annot_path)


def lookup_scop(identifier: str, scop: Dict[str, str]) -> Optional[str]:
    """Look up a target/query id's SCOP lineage (reuses run_jack)."""
    return _lookup_scop(identifier, scop)


def score_hits(
    hit_names: List[str],
    scop: Dict[str, str],
    query_lineage: Optional[str],
) -> Dict[str, Any]:
    """Score parsed hit names against a query's SCOP lineage.

    Args:
        hit_names: target names from parse_tblout()
        scop: lineage map from load_scop()
        query_lineage: the query's SCCS lineage (None if unknown)

    Returns:
        {"TP","FP","Ignored","Unclassified","Hits_total","TP_FP_Ratio"}
    """
    score = _score_scop_names(hit_names, scop, query_lineage)
    hits_total = len(hit_names)
    tp_fp_ratio = (score.tp / (score.tp + score.fp)) if (score.tp + score.fp) > 0 else 0.0
    return {
        "TP": score.tp,
        "FP": score.fp,
        "Ignored": score.ignored,
        "Unclassified": score.unclassified,
        "Hits_total": hits_total,
        "TP_FP_Ratio": round(tp_fp_ratio, 4),
    }


def evaluate_remote_homology(
    tblout_path: str,
    scop_annot_path: str,
    query_id: str,
    scop_cache: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convenience: parse a tblout file and score it against a query.

    Args:
        tblout_path: path to the hmmsearch/jackhmmer .tbl file
        scop_annot_path: path to dir.cla.scope (used only if scop_cache is None)
        query_id: the query's SCOP domain id (e.g. "d1xmka1")
        scop_cache: pre-loaded lineage map (avoids re-reading the big txt file
            across many queries; recommended for batch evaluation)

    Returns:
        {"Qid","TP","FP","Ignored","Unclassified","Hits_total","TP_FP_Ratio"}
    """
    scop = scop_cache if scop_cache is not None else load_scop(scop_annot_path)
    hits = parse_tblout(tblout_path)
    query_lineage = lookup_scop(query_id, scop)
    metrics = score_hits(hits, scop, query_lineage)
    metrics["Qid"] = query_id
    return metrics
