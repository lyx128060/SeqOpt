"""Compact, model-facing evaluation feedback with per-query evidence."""

from __future__ import annotations

from typing import Any

from .eval_schema import EvalResult


def query_results(result: EvalResult | dict[str, Any]) -> list[dict[str, Any]]:
    payload = result.model_dump(mode="json") if isinstance(result, EvalResult) else result
    meta = payload.get("meta") or {}
    rows = meta.get("query_results") or []
    return sorted(
        (dict(row) for row in rows if isinstance(row, dict)),
        key=lambda row: int(row.get("query_index", 0)),
    )


def compact_eval(result: EvalResult | dict[str, Any]) -> dict[str, Any]:
    payload = result.model_dump(mode="json") if isinstance(result, EvalResult) else result
    meta = payload.get("meta") or {}
    return {
        "stage": payload.get("stage"),
        "status": payload.get("status"),
        "tp": payload.get("tp", 0),
        "fp": payload.get("fp", 0),
        "tp_fp_ratio": payload.get("tp_fp_ratio", 0.0),
        "cost_s": payload.get("cost_s", 0.0),
        "n_seqs": payload.get("n_seqs", 0),
        "iterations": payload.get("iterations", 0),
        "acceptance": meta.get("acceptance"),
        "slurm_job_id": meta.get("slurm_job_id"),
        "query_results": query_results(payload),
        "errors": list(payload.get("errors") or []),
    }


def format_eval_observation(result: EvalResult, *, label: str) -> str:
    """Render aggregate and query rows into the actual ToolMessage content."""
    meta = result.meta if isinstance(result.meta, dict) else {}
    acceptance = meta.get("acceptance")
    headline = (
        f"[{label}] status={result.status} tp={result.tp} fp={result.fp} "
        f"ratio={result.tp_fp_ratio:.6f} cost_s={result.cost_s:.4f} "
        f"n_queries={result.n_seqs} max_iters={result.iterations}"
    )
    if isinstance(acceptance, dict):
        headline += (
            f" decision={acceptance.get('decision')} "
            f"accepted={acceptance.get('accepted')}"
        )
        if acceptance.get("quality_non_regression") is not None:
            headline += f" quality_non_regression={acceptance.get('quality_non_regression')}"
        if acceptance.get("speedup_fraction") is not None:
            headline += (
                f" speedup={float(acceptance['speedup_fraction']):.4%}"
                f" required={float(acceptance.get('minimum_speedup_fraction', 0.0)):.4%}"
            )

    rows = query_results(result)
    lines = [headline]
    if rows:
        lines.extend([
            "Per-query results (cost ratio is candidate/reference):",
            "idx query_id status tp fp ratio cost_s ref_s x_ref limit_s iter hits",
        ])
        for row in rows:
            reference = row.get("reference") or {}
            lines.append(" ".join((
                str(row.get("query_index", "?")),
                str(row.get("query_id") or "?"),
                str(row.get("status") or "?"),
                str(row.get("tp", 0)),
                str(row.get("fp", 0)),
                _number(row.get("tp_fp_ratio"), 4),
                _number(row.get("cost_s"), 2),
                _number(reference.get("cost_s"), 2),
                _number(row.get("cost_ratio"), 3),
                _number(row.get("time_limit_s"), 2),
                str(row.get("iterations", 0)),
                str(row.get("nhits") if row.get("nhits") is not None else "?"),
            )))
            if row.get("error"):
                lines.append(f"  query_error[{row.get('query_id') or row.get('query_index')}]: {row['error']}")
    else:
        lines.append("Per-query results unavailable in this legacy result; inspect meta.job_dir shards.")

    if result.errors:
        lines.append("Errors: " + "; ".join(result.errors[:10]))
    return "\n".join(lines)


def _number(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
