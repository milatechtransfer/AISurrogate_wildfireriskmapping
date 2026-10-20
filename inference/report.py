"""
Self-contained HTML summary reports for input checks, predictions, and BurnP3+ evaluations.

The generated HTML files use inline CSS and relative links to output rasters and PNG plots so they work
offline on any workstation without external web dependencies.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from inference.bundle import ModelBundle
from inference.check import CheckReport, Finding
from inference.evaluate import HEADLINE_METRICS, EvaluateRun
from inference.predict import HAZARD_CLASS_NODATA, OUTPUT_NODATA, PredictRun, available_rasters, raster_descriptions

SUMMARY_REPORT_FILENAME = "summary_report.html"

_CSS = """
:root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --accent: #b91c1c;
    --ok-bg: #dcfce7;
    --ok-fg: #166534;
    --warn-bg: #fef9c3;
    --warn-fg: #854d0e;
    --err-bg: #fee2e2;
    --err-fg: #991b1b;
    --note-bg: #e0f2fe;
    --note-fg: #075985;
}
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
    line-height: 1.5;
}
.container {
    max-width: 1180px;
    margin: 0 auto;
}
header {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 6px solid var(--accent);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
header h1 {
    margin: 0 0 6px 0;
    font-size: 1.45rem;
}
header .subtitle {
    color: var(--muted);
    font-size: 0.92rem;
}
.meta-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-top: 16px;
}
.meta-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
}
.meta-item .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
}
.meta-item .value {
    font-weight: 600;
    font-size: 0.95rem;
    word-break: break-all;
}
section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 18px;
}
section h2 {
    margin: 0 0 12px 0;
    font-size: 1.15rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
}
section h3 {
    margin: 16px 0 8px 0;
    font-size: 1rem;
}
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-ok { background: var(--ok-bg); color: var(--ok-fg); }
.badge-warning { background: var(--warn-bg); color: var(--warn-fg); }
.badge-error { background: var(--err-bg); color: var(--err-fg); }
.badge-note { background: var(--note-bg); color: var(--note-fg); }
.table-scroll {
    overflow-x: auto;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 8px;
}
th, td {
    text-align: left;
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
}
th {
    background: var(--bg);
    font-weight: 600;
    color: var(--muted);
}
td.num, th.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
}
ul.findings {
    list-style: none;
    padding: 0;
    margin: 0;
}
ul.findings li {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 10px;
    align-items: baseline;
}
ul.findings li:last-child {
    border-bottom: none;
}
.finding-path {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.84rem;
    color: var(--muted);
}
.plot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 14px;
    margin-top: 10px;
}
.plot-card {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    background: var(--bg);
}
.plot-card img {
    width: 100%;
    height: auto;
    border-radius: 4px;
    display: block;
}
.plot-card .caption {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 6px;
    text-align: center;
}
code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.86rem;
    background: var(--bg);
    padding: 1px 5px;
    border-radius: 4px;
}
a {
    color: #1d4ed8;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _page_shell(title: str, subtitle: str, meta_items: list[tuple[str, str]], body_html: str) -> str:
    meta_cards = "\n".join(
        f'<div class="meta-item"><div class="label">{_esc(label)}</div><div class="value">{val_html}</div></div>'
        for label, val_html in meta_items
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
<header>
  <h1>{_esc(title)}</h1>
  <div class="subtitle">{_esc(subtitle)}</div>
  <div class="meta-grid">{meta_cards}</div>
</header>
{body_html}
</div>
</body>
</html>
"""


def _status_badge_for_check(report: CheckReport) -> str:
    if report.errors:
        return f'<span class="badge badge-error">{len(report.errors)} Error(s)</span>'
    if report.warnings:
        return f'<span class="badge badge-warning">Ready ({len(report.warnings)} Warning(s))</span>'
    return '<span class="badge badge-ok">Ready</span>'


def _render_findings_list(findings: list[Finding]) -> str:
    if not findings:
        return "<p>No issues or notes reported.</p>"
    items = []
    for finding in findings:
        badge_cls = {
            "error": "badge-error",
            "warning": "badge-warning",
            "note": "badge-note",
        }.get(finding.level, "badge-note")
        loc = ""
        if finding.path:
            loc = f'<span class="finding-path">{_esc(finding.path)}:</span> '
        elif finding.hexel:
            loc = f'<span class="finding-path">{_esc(finding.hexel)}:</span> '
        items.append(f'<li><span class="badge {badge_cls}">{_esc(finding.level)}</span><div>{loc}{_esc(finding.message)}</div></li>')
    return f'<ul class="findings">{"".join(items)}</ul>'


def _run_warnings_section(warnings: list[str]) -> str:
    """Problems hit while running (e.g. a picture that could not be drawn) that did not stop the run."""
    items = "".join(f'<li><span class="badge badge-warning">warning</span><div>{_esc(w)}</div></li>' for w in warnings)
    return f'<section><h2>Run Warnings</h2><ul class="findings">{items}</ul></section>'


def _map_gallery_html(output_dir: Path, bundle: ModelBundle, heading: str, intro: str) -> str:
    """A section of the PNG pictures in each map-type folder (bp/, fi/, ...), grouped by map type; "" when there are none."""
    descriptions = raster_descriptions(bundle)
    groups = []
    for name in available_rasters(bundle):
        kinds = ("", "_maps", "_scatter", "_histogram")

        def order(png: Path, name: str = name, kinds: tuple[str, ...] = kinds) -> tuple[str, int, str]:
            hexel, _, rest = png.stem.partition(f"_{name}")
            return (hexel, kinds.index(rest) if rest in kinds else len(kinds), png.name)

        pngs = sorted((output_dir / name).glob("*.png"), key=order)
        if not pngs:
            continue
        cards = []
        for png in pngs:
            rel = png.relative_to(output_dir).as_posix()
            cards.append(
                f'<div class="plot-card">'
                f'<a href="{_esc(rel)}" target="_blank"><img src="{_esc(rel)}" alt="{_esc(png.stem)}"></a>'
                f'<div class="caption"><code>{_esc(rel)}</code></div>'
                f"</div>"
            )
        label, units = descriptions[name]
        title = f"{label}, {units}" if units and name != "hazard_class" else label
        groups.append(f'<h3>{_esc(title)} &mdash; <code>{_esc(name)}/</code></h3><div class="plot-grid">{"".join(cards)}</div>')
    if not groups:
        return ""
    return f"<section><h2>{_esc(heading)}</h2><p>{intro}</p>{''.join(groups)}</section>"


def render_check_report_html(report: CheckReport) -> str:
    """Render an input validation :class:`CheckReport` as a standalone HTML page."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    meta = [
        ("Status", _status_badge_for_check(report)),
        ("Model Bundle", _esc(report.bundle)),
        ("Project Folder", f"<code>{_esc(report.project_dir)}</code>"),
        ("Mask Scope", f"<code>{_esc(report.mask_scope)}</code>"),
        ("Regions (Hexels)", _esc(", ".join(report.hexels) if report.hexels else "None")),
        ("Checked At", _esc(timestamp)),
    ]

    sections = []
    project_findings = [f for f in report.findings if f.hexel is None]
    if project_findings:
        sections.append(f"<section><h2>Project-Level Notes &amp; Checks</h2>{_render_findings_list(project_findings)}</section>")

    rows = []
    for hexel in report.hexels:
        hex_findings = [f for f in report.findings if f.hexel == hexel]
        n_err = sum(f.level == "error" for f in hex_findings)
        n_warn = sum(f.level == "warning" for f in hex_findings)
        if n_err:
            badge = f'<span class="badge badge-error">{n_err} error(s)</span>'
        elif n_warn:
            badge = f'<span class="badge badge-warning">{n_warn} warning(s)</span>'
        else:
            badge = '<span class="badge badge-ok">OK</span>'
        detail = _render_findings_list(hex_findings) if hex_findings else "All required spatial &amp; tabular inputs verified."
        rows.append(f"<tr><td><code>{_esc(hexel)}</code></td><td>{badge}</td><td>{detail}</td></tr>")

    if rows:
        sections.append(
            "<section><h2>Region (Hexel) Input Checks</h2>"
            "<table><thead><tr><th>Region</th><th>Status</th><th>Details</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    elif not project_findings:
        sections.append(f"<section><h2>Findings</h2>{_render_findings_list(report.findings)}</section>")

    return _page_shell(
        title="Wildfire AI Surrogate — Input Check Report",
        subtitle=f"Validation summary for preparing to {report.action}",
        meta_items=meta,
        body_html="\n".join(sections),
    )


def write_check_report_html(report: CheckReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_check_report_html(report), encoding="utf-8")
    return path


def render_predict_report_html(run: PredictRun, bundle: ModelBundle, project_dir: str | Path) -> str:
    """Render a completed :class:`PredictRun` as a standalone HTML summary page."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    total_seconds = sum(h.seconds for h in run.hexels)
    meta = [
        ("Status", '<span class="badge badge-ok">Completed</span>'),
        ("Model Bundle", _esc(f"{bundle.manifest.name} v{bundle.manifest.version}")),
        ("Project Folder", f"<code>{_esc(project_dir)}</code>"),
        ("Output Folder", f"<code>{_esc(run.output_dir)}</code>"),
        ("Regions Predicted", _esc(str(len(run.hexels)))),
        ("Total Runtime", _esc(f"{total_seconds:.1f} s")),
        ("Generated At", _esc(timestamp)),
    ]

    sections: list[str] = []

    hex_rows = []
    for hex_res in run.hexels:
        hex_name = f"hex{hex_res.hex_id}"
        links = []
        for out_key, rel_path in hex_res.outputs.items():
            links.append(f'<a href="{_esc(rel_path)}"><code>{_esc(out_key)}</code></a>')
        fuel_note = "Standard bundle curves"
        if hex_res.fuel_curves:
            derived = hex_res.fuel_curves.get("derived") or {}
            if derived:
                summary_pairs = [f"{code} ({ft})" for code, ft in list(derived.items())[:6]]
                if len(derived) > 6:
                    summary_pairs.append(f"+{len(derived) - 6} more")
                fuel_note = "Derived: " + ", ".join(summary_pairs)
        hex_rows.append(
            f"<tr>"
            f"<td><code>{_esc(hex_name)}</code></td>"
            f'<td class="num">{hex_res.num_patches:,}</td>'
            f'<td class="num">{hex_res.seconds:.1f} s</td>'
            f"<td>{' &middot; '.join(links)}</td>"
            f"<td>{_esc(fuel_note)}</td>"
            f"</tr>"
        )

    sections.append(
        "<section><h2>Predicted Regions &amp; Output GeoTIFFs</h2>"
        '<table><thead><tr><th>Region</th><th class="num">Patches</th><th class="num">Time</th>'
        "<th>Output Rasters (click to open/download)</th><th>Fuel Curves</th></tr></thead>"
        f"<tbody>{''.join(hex_rows)}</tbody></table></section>"
    )

    written = {name for hex_res in run.hexels for name in hex_res.outputs}
    target_rows = [
        f"<tr><td><code>{_esc(name)}</code></td><td>{_esc(label)}</td><td><code>{_esc(units or '')}</code></td>"
        f"<td><code>{HAZARD_CLASS_NODATA if name == 'hazard_class' else f'{OUTPUT_NODATA:g}'}</code></td></tr>"
        for name, (label, units) in raster_descriptions(bundle).items()
        if name in written
    ]

    gallery = _map_gallery_html(
        run.output_dir,
        bundle,
        "Maps",
        "Each map type has its own folder in the Output Folder, holding the GeoTIFF and this picture for every region. "
        "Click a picture to enlarge it. Colours stop at the 99th percentile so a few extreme cells do not hide the pattern.",
    )
    if gallery:
        sections.insert(0, gallery)

    sections.append(
        "<section><h2>Output Layer Reference &amp; GIS Info</h2>"
        f"<p>All output GeoTIFFs are written in <code>{_esc(bundle.manifest.inputs.crs)}</code> "
        f"(~{bundle.manifest.inputs.resolution_m:g} m resolution) and can be dragged directly into QGIS, ArcGIS Pro, or SyncroSim.</p>"
        "<table><thead><tr><th>Layer Key</th><th>Description</th><th>Units</th><th>NoData Value</th></tr></thead>"
        f"<tbody>{''.join(target_rows)}</tbody></table></section>"
    )

    if run.input_check is not None:
        sections.append(
            f"<section><h2>Input Validation Notes &amp; Warnings</h2>{_render_findings_list(run.input_check.findings)}</section>"
        )
    if run.warnings:
        sections.append(_run_warnings_section(run.warnings))

    return _page_shell(
        title="Wildfire AI Surrogate — Prediction Summary",
        subtitle="Operational surrogate map generation report",
        meta_items=meta,
        body_html="\n".join(sections),
    )


def write_predict_report_html(
    run: PredictRun,
    bundle: ModelBundle,
    project_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    path = Path(output_path) if output_path is not None else (run.output_dir / SUMMARY_REPORT_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_predict_report_html(run, bundle, project_dir), encoding="utf-8")
    return path


def _dataframe_to_html_table(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    if df.empty:
        return "<p>No metrics available.</p>"
    headers = []
    for col in df.columns:
        is_num = pd.api.types.is_numeric_dtype(df[col])
        cls = ' class="num"' if is_num else ""
        headers.append(f"<th{cls}>{_esc(col)}</th>")
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                cells.append('<td class="num">&mdash;</td>')
            elif pd.api.types.is_float_dtype(df[col]):
                cells.append(f'<td class="num">{_esc(float_fmt.format(float(val)))}</td>')
            elif pd.api.types.is_integer_dtype(df[col]):
                cells.append(f'<td class="num">{int(val):,}</td>')
            else:
                cells.append(f"<td><code>{_esc(val)}</code></td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="table-scroll"><table><thead><tr>{"".join(headers)}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render_evaluate_report_html(run: EvaluateRun, bundle: ModelBundle, project_dir: str | Path) -> str:
    """Render a completed :class:`EvaluateRun` as a standalone HTML report with tables and embedded plots."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    meta = [
        ("Status", '<span class="badge badge-ok">Evaluated</span>'),
        ("Model Bundle", _esc(f"{bundle.manifest.name} v{bundle.manifest.version}")),
        ("Project Folder", f"<code>{_esc(project_dir)}</code>"),
        ("Evaluation Folder", f"<code>{_esc(run.output_dir)}</code>"),
        ("Mask Scope", f"<code>{_esc(run.mask_scope)}</code>"),
        ("Regions Evaluated", _esc(", ".join(run.hexels))),
        ("Generated At", _esc(timestamp)),
    ]

    sections: list[str] = []

    if run.warnings:
        sections.append(_run_warnings_section(run.warnings))

    if not run.summary.empty:
        # The full metric set (top-k MAE/IoU, AUC, ...) is in metrics_summary.csv; keep the page table readable.
        preferred = ["target", "area", "n_hexels", *HEADLINE_METRICS, "iou_top10"]
        cols = [c for c in preferred if c in run.summary.columns]
        sections.append(
            "<section><h2>Summary Metrics (AI Surrogate vs. BurnP3+)</h2>"
            "<p>Concordance correlation (<code>ccc</code>) and <code>spearman</code> rank correlation measure spatial agreement "
            "(1.0 is best); <code>mae</code> and <code>bias</code> are in physical units (0.0 is best); "
            "<code>iou_top10</code> is the overlap of the top-10% highest-value areas (1.0 is best). "
            "All other metrics are in the CSV files below.</p>"
            f"{_dataframe_to_html_table(run.summary[cols])}"
            '<p style="margin-top:10px;font-size:0.86rem;">'
            'CSV downloads: <a href="metrics_summary.csv"><code>metrics_summary.csv</code></a> &middot; '
            '<a href="metrics_per_hexel.csv"><code>metrics_per_hexel.csv</code></a></p>'
            "</section>"
        )

    if run.hazard_summary:
        haz_df = pd.DataFrame([run.hazard_summary])
        headline_cols = [
            c
            for c in (
                "n_hexels",
                "exact_accuracy",
                "within_1_accuracy",
                "within_2_accuracy",
                "mean_absolute_class_error",
                "macro_f1",
                "macro_iou",
            )
            if c in haz_df.columns
        ]
        cm_img_html = ""
        cm_png = run.output_dir / "hazard_confusion_matrix.png"
        if cm_png.is_file():
            cm_img_html = (
                '<div class="plot-grid" style="max-width:520px;margin-top:12px;">'
                '<div class="plot-card"><img src="hazard_confusion_matrix.png" alt="Hazard confusion matrix">'
                '<div class="caption">Hazard Class Confusion Matrix (BurnP3+ vs AI Surrogate)</div></div></div>'
            )
        sections.append(
            "<section><h2>Wildfire Hazard Class Agreement</h2>"
            f"{_dataframe_to_html_table(haz_df[headline_cols] if headline_cols else haz_df)}"
            f"{cm_img_html}</section>"
        )

    if not run.metrics.empty and len(run.hexels) > 1:
        headline_hex_cols = [c for c in ("hexel", "target", "area", "n_pixels", *HEADLINE_METRICS) if c in run.metrics.columns]
        sections.append(f"<section><h2>Per-Region Metrics</h2>{_dataframe_to_html_table(run.metrics[headline_hex_cols])}</section>")

    gallery = _map_gallery_html(
        run.output_dir,
        bundle,
        "Maps & Comparison Plots",
        "Per region: <code>_maps</code> shows the surrogate, BurnP3+ and their difference side by side; "
        "<code>_scatter</code> and <code>_histogram</code> compare their values. Click a picture to enlarge it.",
    )
    if gallery:
        sections.append(gallery)

    if run.input_check is not None:
        sections.append(
            f"<section><h2>Input Validation Notes &amp; Warnings</h2>{_render_findings_list(run.input_check.findings)}</section>"
        )

    return _page_shell(
        title="Wildfire AI Surrogate — BurnP3+ Evaluation Report",
        subtitle="Comparison of AI surrogate predictions against BurnP3+ simulations",
        meta_items=meta,
        body_html="\n".join(sections),
    )


def write_evaluate_report_html(
    run: EvaluateRun,
    bundle: ModelBundle,
    project_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    path = Path(output_path) if output_path is not None else (run.output_dir / SUMMARY_REPORT_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_evaluate_report_html(run, bundle, project_dir), encoding="utf-8")
    return path
