from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from treeindex.utils.config import project_root
from treeindex.utils.table import render_results_table


def default_results_dir() -> Path:
    return project_root() / "results"


def timestamped_report_path(
    *,
    filename_prefix: str,
    extension: str,
    output_dir: str | Path | None = None,
    timestamp: str | None = None,
) -> Path:
    report_dir = Path(output_dir) if output_dir is not None else default_results_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    return report_dir / f"{filename_prefix}-{timestamp}.{extension}"


def write_text_report(
    *,
    text: str,
    filename_prefix: str,
    output_dir: str | Path | None = None,
) -> Path:
    txt_path = timestamped_report_path(
        filename_prefix=filename_prefix,
        extension="txt",
        output_dir=output_dir,
    )
    txt_path.write_text(f"{text.rstrip()}\n", encoding="utf-8")
    return txt_path


def write_experiment_report(
    *,
    title: str,
    rows: Sequence[Any],
    filename_prefix: str,
    output_dir: str | Path | None = None,
    write_json: bool = True,
) -> tuple[Path, Path | None]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    txt_path = timestamped_report_path(
        filename_prefix=filename_prefix,
        extension="txt",
        output_dir=output_dir,
        timestamp=timestamp,
    )
    json_path = (
        timestamped_report_path(
            filename_prefix=filename_prefix,
            extension="json",
            output_dir=output_dir,
            timestamp=timestamp,
        )
        if write_json
        else None
    )

    table_text = render_results_table(rows)
    txt_path.write_text(f"{title}\n{table_text}\n", encoding="utf-8")
    if json_path is not None:
        json_path.write_text(json.dumps([_row_to_dict(row) for row in rows], indent=2), encoding="utf-8")
    return txt_path, json_path


def _row_to_dict(row: Any) -> dict[str, Any]:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if is_dataclass(row):
        return asdict(row)
    raise TypeError(f"Unsupported result row type for reporting: {type(row)!r}")
