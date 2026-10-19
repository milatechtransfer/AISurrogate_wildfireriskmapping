"""
One-click desktop application for checking, predicting, and evaluating BurnP3+ projects with an AI surrogate bundle.

Launch with:
    python -m inference.app [--bundle PATH] [--project PATH] [--output PATH]

Or double-click ``run_surrogate_app.bat`` (Windows) / ``run_surrogate_app.command`` (macOS / Linux).
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from data_preparation.paths import Paths
from data_preparation.utils import find_hex_ids
from inference.bundle import (
    MANIFEST_FILENAME,
    MASK_SCOPES,
    NO_MASK_SCOPE,
    WEIGHTS_FILENAME,
    BundleError,
    ModelBundle,
    load_bundle,
    resolve_mask_scope,
    resolve_scenario_name,
)
from inference.check import check_project
from inference.evaluate import EvaluateError, format_summary, run_evaluate
from inference.predict import PredictError, available_rasters, resolve_fire_size_table, run_predict
from inference.report import (
    SUMMARY_REPORT_FILENAME,
    write_check_report_html,
    write_evaluate_report_html,
    write_predict_report_html,
)

ActionType = Literal["check", "predict", "evaluate"]
_HEX_FOLDER_RE = re.compile(r"^hex(\d+)$")
_SCENARIO_FBP_RE = re.compile(r"^hex\d+_fbp_(.+)\.tif$")


@dataclass
class ProjectInspection:
    """Summary of a BurnP3+ project folder discovered on disk."""

    project_dir: Path
    selected_single_hexel: str | None = None
    hex_ids: list[str] = field(default_factory=list)
    has_actual_mask: bool = False
    has_buffer_mask: bool = False
    has_default_fuel_grid: bool = False
    scenarios: list[str] = field(default_factory=list)
    has_burnp3_results: bool = False
    recommended_mask_scope: str = "actual"
    recommended_scenario: str | None = None
    default_predict_output: Path = field(default_factory=Path)
    default_evaluate_output: Path = field(default_factory=Path)

    @property
    def hexel_names(self) -> list[str]:
        return [f"hex{hid}" for hid in self.hex_ids]

    def summary_line(self) -> str:
        if not self.hex_ids:
            return "No hexNN region folders found in this directory."
        regions = (
            f"1 region (hex{self.hex_ids[0]})"
            if len(self.hex_ids) == 1
            else f"{len(self.hex_ids)} regions ({', '.join(self.hexel_names[:4])}{'...' if len(self.hex_ids) > 4 else ''})"
        )
        mask_desc = {
            "actual": "actual mask (.shp)",
            "buffer": "buffer mask (.shp)",
            NO_MASK_SCOPE: "none (full raster extent)",
        }.get(self.recommended_mask_scope, self.recommended_mask_scope)
        scen_desc = self.recommended_scenario or (
            f"{len(self.scenarios)} scenario(s) available" if self.scenarios else "national/default fuel grid"
        )
        results_desc = "BurnP3+ results found (Evaluate ready)" if self.has_burnp3_results else "Inputs only (Predict ready)"
        return f"{regions}  |  Mask: {mask_desc}  |  Fuel scenario: {scen_desc}  |  {results_desc}"


@dataclass
class WorkflowRequest:
    action: ActionType
    bundle_dir: Path
    project_dir: Path
    output_dir: Path
    hex_ids: list[str] | None = None
    mask_scope: str | None = None
    scenario_name: str | None = None
    fire_size_table: Path | None = None
    device: str = "auto"
    batch_size: int = 8
    overwrite: bool = True
    plots: bool = True
    by_firezone: bool = False
    rasters: list[str] | None = None  # predict only: GeoTIFFs to write (None = all)


@dataclass
class WorkflowOutcome:
    action: ActionType
    ok: bool
    summary_text: str
    output_dir: Path
    report_html: Path | None = None
    error: str | None = None


# Predict output checkboxes (raster name, label) and the ones ticked by default.
PREDICT_RASTER_OPTIONS = (
    ("bp", "Burn probability"),
    ("fi", "Fire intensity"),
    ("ros", "Rate of spread"),
    ("hazard_raw", "Hazard (raw)"),
    ("hazard_scaled", "Hazard (scaled 0-100)"),
    ("hazard_class", "Hazard class"),
)
DEFAULT_PREDICT_RASTERS = ("bp", "fi", "ros", "hazard_class")

# Shown in the fuel-scenario dropdown in place of an empty value (maps to scenario_name=None).
DEFAULT_SCENARIO_LABEL = "(default fuel grid)"


def scenario_name_from_ui(raw: str) -> str | None:
    """Map the fuel-scenario dropdown value to a scenario name (blank or the default label -> None)."""
    cleaned = raw.strip()
    return None if not cleaned or cleaned == DEFAULT_SCENARIO_LABEL else cleaned


def is_bundle_dir(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / MANIFEST_FILENAME).is_file() and (candidate / WEIGHTS_FILENAME).is_file()


def _default_bundle_search_roots() -> list[Path]:
    """Directories searched automatically for a model bundle (supports source checkouts, Windows/Linux builds, and macOS .app)."""
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        roots.extend(exe_path.parents[:4])
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass).resolve())
    roots.extend([Path.cwd().resolve(), Path(__file__).resolve().parents[1]])
    return roots


def discover_bundles(search_roots: Iterable[str | Path] | None = None) -> list[Path]:
    """Find model bundle directories in the app folder, working directory, repository root, or common subfolders."""
    roots = [Path(r).expanduser().resolve() for r in search_roots] if search_roots is not None else _default_bundle_search_roots()
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _check(directory: Path) -> None:
        resolved = directory.resolve()
        if resolved not in seen and is_bundle_dir(resolved):
            seen.add(resolved)
            candidates.append(resolved)

    for root in roots:
        if not root.is_dir():
            continue
        _check(root)
        for sub_name in ("bundles", "model_bundles", "releases"):
            sub = root / sub_name
            if sub.is_dir():
                for child in sorted(sub.iterdir()):
                    _check(child)
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                _check(child)
    return candidates


def discover_sample_projects(search_roots: Iterable[str | Path] | None = None) -> list[Path]:
    """Find bundled sample BurnP3+ projects (e.g. ``sample_project/`` or ``examples/sample_project/``)."""
    roots = [Path(r).expanduser().resolve() for r in search_roots] if search_roots is not None else _default_bundle_search_roots()
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for rel in ("sample_project", "examples/sample_project"):
            candidate = (root / rel).resolve()
            if candidate not in seen and candidate.is_dir() and find_hex_ids(str(candidate)):
                seen.add(candidate)
                found.append(candidate)
    return found


def inspect_project_folder(project_dir: str | Path) -> ProjectInspection:
    """Inspect a user-selected project folder (or a single ``hexNN`` folder) and recommend settings."""
    raw_path = Path(project_dir).expanduser().resolve()
    selected_single: str | None = None

    # If the user clicked directly inside a hexNN folder (or its spatial/tabular subfolder), step up to the project root.
    if raw_path.name in {"spatial", "tabular", "results"} and _HEX_FOLDER_RE.match(raw_path.parent.name):
        raw_path = raw_path.parent
    match = _HEX_FOLDER_RE.match(raw_path.name)
    if match and ((raw_path / "spatial").is_dir() or (raw_path / "tabular").is_dir()):
        selected_single = match.group(1)
        raw_path = raw_path.parent

    output_parent = raw_path.parent
    if output_parent.exists() and not os.access(output_parent, os.W_OK):
        desktop = Path.home() / "Desktop"
        output_parent = desktop if (desktop.is_dir() and os.access(desktop, os.W_OK)) else Path.home()
    default_pred = output_parent / f"{raw_path.name}_predictions"
    default_eval = output_parent / f"{raw_path.name}_evaluation"

    if not raw_path.is_dir():
        return ProjectInspection(
            project_dir=raw_path,
            default_predict_output=default_pred,
            default_evaluate_output=default_eval,
        )

    all_hex_ids = sorted(find_hex_ids(str(raw_path)))
    hex_ids = [selected_single] if (selected_single and selected_single in all_hex_ids) else all_hex_ids
    if not hex_ids:
        return ProjectInspection(
            project_dir=raw_path,
            default_predict_output=default_pred,
            default_evaluate_output=default_eval,
        )

    has_actual = True
    has_buffer = True
    has_default_fbp = False
    scenarios_set: set[str] = set()
    has_results = False

    for hex_id in hex_ids:
        paths = Paths(hex_id=hex_id, root_dir=raw_path)
        if not paths.mask_grid_actual(hex_id).is_file():
            has_actual = False
        if not paths.mask_grid_buffer(hex_id).is_file():
            has_buffer = False
        if paths.fuel_grid(hex_id, scenario_name=None).is_file():
            has_default_fbp = True
        if paths.spatial_dir.is_dir():
            for tif in paths.spatial_dir.glob(f"hex{hex_id}_fbp_*.tif"):
                m = _SCENARIO_FBP_RE.match(tif.name)
                if m:
                    scenarios_set.add(m.group(1))
        if paths.output_burn_prob(scenario_name=None).is_file():
            has_results = True
        results_dir = paths.base_dir / "results"
        if results_dir.is_dir():
            for child in results_dir.iterdir():
                if (
                    child.is_dir()
                    and not child.name.startswith("burnP3Plus_Output")
                    and paths.output_burn_prob(scenario_name=child.name).is_file()
                ):
                    has_results = True
                    scenarios_set.add(child.name)

    scenarios = sorted(scenarios_set)
    if has_actual:
        rec_mask = "actual"
    elif has_buffer:
        rec_mask = "buffer"
    else:
        rec_mask = NO_MASK_SCOPE

    rec_scenario: str | None = None
    if not has_default_fbp and len(scenarios) == 1:
        rec_scenario = scenarios[0]

    return ProjectInspection(
        project_dir=raw_path,
        selected_single_hexel=selected_single,
        hex_ids=hex_ids,
        has_actual_mask=has_actual,
        has_buffer_mask=has_buffer,
        has_default_fuel_grid=has_default_fbp,
        scenarios=scenarios,
        has_burnp3_results=has_results,
        recommended_mask_scope=rec_mask,
        recommended_scenario=rec_scenario,
        default_predict_output=default_pred,
        default_evaluate_output=default_eval,
    )


def parse_hex_ids_input(raw: str) -> list[str] | None:
    """Parse a comma- or space-separated hexel list from the UI (empty or 'all' -> None)."""
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in {"all", "*"}:
        return None
    parts = [tok.strip() for tok in re.split(r"[\s,]+", cleaned) if tok.strip()]
    return parts or None


def execute_workflow_action(request: WorkflowRequest) -> WorkflowOutcome:
    """Run ``check``, ``predict``, or ``evaluate`` and generate an HTML summary report."""
    output_dir = request.output_dir.expanduser().resolve()
    project_dir = request.project_dir.expanduser().resolve()
    bundle_dir = request.bundle_dir.expanduser().resolve()
    scenario = request.scenario_name.strip() if request.scenario_name and request.scenario_name.strip() else None

    try:
        bundle: ModelBundle = load_bundle(bundle_dir)
        scope = resolve_mask_scope(request.mask_scope, bundle)
        resolved_scenario = resolve_scenario_name(scenario, bundle)

        if request.action == "check":
            fire_size_path = resolve_fire_size_table(bundle, request.fire_size_table)
            inspection = inspect_project_folder(project_dir)
            report = check_project(
                bundle=bundle,
                project_dir=project_dir,
                hex_ids=request.hex_ids,
                fire_size_table=fire_size_path,
                mask_scope=scope,
                scenario_name=resolved_scenario,
                inputs=True,
                outputs=inspection.has_burnp3_results,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = write_check_report_html(report, output_dir / SUMMARY_REPORT_FILENAME)
            return WorkflowOutcome(
                action="check",
                ok=report.ok,
                summary_text=report.format(),
                output_dir=output_dir,
                report_html=report_path,
                error=None if report.ok else f"Found {len(report.errors)} input error(s).",
            )

        if request.action == "predict":
            pred_run = run_predict(
                bundle_dir=bundle_dir,
                project_dir=project_dir,
                output_dir=output_dir,
                fire_size_table=request.fire_size_table,
                hex_ids=request.hex_ids,
                device=request.device,
                batch_size=request.batch_size,
                num_workers=0,
                mask_scope=scope,
                hazard=True,
                overwrite=request.overwrite,
                scenario_name=resolved_scenario,
                rasters=request.rasters,
            )
            report_path = write_predict_report_html(pred_run, bundle, project_dir, output_dir / SUMMARY_REPORT_FILENAME)
            lines = [
                f"Predicted {len(pred_run.hexels)} region(s) into {output_dir}:",
                *[f"  hex{h.hex_id}: {', '.join(h.outputs.keys())} ({h.num_patches} patches, {h.seconds:.1f}s)" for h in pred_run.hexels],
                f"Summary report: {report_path}",
            ]
            return WorkflowOutcome(
                action="predict",
                ok=True,
                summary_text="\n".join(lines),
                output_dir=output_dir,
                report_html=report_path,
            )

        if request.action == "evaluate":
            eval_run = run_evaluate(
                bundle_dir=bundle_dir,
                project_dir=project_dir,
                output_dir=output_dir,
                hex_ids=request.hex_ids,
                mask_scope=scope,
                by_firezone=request.by_firezone,
                plots=request.plots,
                fire_size_table=request.fire_size_table,
                device=request.device,
                batch_size=request.batch_size,
                num_workers=0,
                scenario_name=resolved_scenario,
                overwrite=request.overwrite,
            )
            report_path = write_evaluate_report_html(eval_run, bundle, project_dir, output_dir / SUMMARY_REPORT_FILENAME)
            summary = format_summary(eval_run) + f"\nSummary report: {report_path}"
            return WorkflowOutcome(
                action="evaluate",
                ok=True,
                summary_text=summary,
                output_dir=output_dir,
                report_html=report_path,
            )

        raise ValueError(f"Unsupported action: {request.action}")
    except (BundleError, PredictError, EvaluateError, FileNotFoundError, ValueError, KeyError) as exc:
        return WorkflowOutcome(
            action=request.action,
            ok=False,
            summary_text=f"Error: {exc}",
            output_dir=output_dir,
            report_html=None,
            error=str(exc),
        )


def open_in_os(path: str | Path) -> bool:
    """Open a file or folder in the user's desktop environment."""
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return False
    try:
        if target.is_file() and target.suffix.lower() in {".html", ".htm"}:
            return webbrowser.open(target.as_uri())
        system = platform.system()
        if system == "Windows":
            os.startfile(str(target))  # type: ignore[attr-defined]
            return True
        if system == "Darwin":
            subprocess.Popen(["open", str(target)])
            return True
        subprocess.Popen(["xdg-open", str(target)])
        return True
    except OSError:
        return False


class _QueueLogHandler(logging.Handler):
    """Forwards Python log records into a thread-safe queue for the Tkinter log viewer."""

    def __init__(self, msg_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__(level=logging.INFO)
        self._queue = msg_queue
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(("log", self.format(record)))


def launch_gui(
    initial_bundle: str | Path | None = None,
    initial_project: str | Path | None = None,
    initial_output: str | Path | None = None,
) -> int:
    """Start the Tkinter desktop interface."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError as exc:
        print(
            f"Error: Tkinter GUI support is not installed in this Python build ({exc}).\n"
            "Use the command line instead: python -m inference.predict --help",
            file=sys.stderr,
        )
        return 2

    root = tk.Tk()
    root.title("Wildfire AI Surrogate — BurnP3+ Project Launcher")
    root.geometry("920x740")
    root.minsize(780, 620)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
    style.configure("Subheader.TLabel", font=("Segoe UI", 9), foreground="#475569")
    style.configure("Action.TButton", font=("Segoe UI", 10), padding=(10, 7))

    # State variables
    bundle_var = tk.StringVar()
    project_var = tk.StringVar()
    output_var = tk.StringVar()
    mask_var = tk.StringVar(value="actual")
    scenario_var = tk.StringVar(value=DEFAULT_SCENARIO_LABEL)
    hex_ids_var = tk.StringVar(value="all")
    device_var = tk.StringVar(value="auto")
    batch_size_var = tk.IntVar(value=8)
    fire_size_var = tk.StringVar(value="")
    overwrite_var = tk.BooleanVar(value=True)
    open_report_var = tk.BooleanVar(value=True)
    plots_var = tk.BooleanVar(value=True)
    by_firezone_var = tk.BooleanVar(value=False)
    raster_vars = {name: tk.BooleanVar(value=name in DEFAULT_PREDICT_RASTERS) for name, _ in PREDICT_RASTER_OPTIONS}

    bundle_status_var = tk.StringVar(value="Select a model bundle folder (contains manifest.yaml & model.pt).")
    project_status_var = tk.StringVar(value="Select a BurnP3+ project folder (containing hexNN/ subfolders).")
    run_status_var = tk.StringVar(value="Ready.")

    output_customized = False
    last_report_path: Path | None = None
    msg_queue: queue.Queue[tuple[str, object]] = queue.Queue()

    # Main layout
    outer = ttk.Frame(root, padding=14)
    outer.pack(fill=tk.BOTH, expand=True)

    # Header
    hdr = ttk.Frame(outer)
    hdr.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(hdr, text="NRCan Wildfire AI Surrogate — BurnP3+ Project Launcher", style="Header.TLabel").pack(anchor=tk.W)
    ttk.Label(
        hdr,
        text="Check project inputs, generate BP / FI / ROS / Hazard maps, or evaluate against BurnP3+ simulations.",
        style="Subheader.TLabel",
    ).pack(anchor=tk.W)

    # 1. Folders Frame
    folders_box = ttk.LabelFrame(outer, text=" 1. Select Model Bundle & BurnP3+ Project Folder ", padding=10)
    folders_box.pack(fill=tk.X, pady=(0, 8))
    folders_box.columnconfigure(1, weight=1)

    def _browse_bundle() -> None:
        chosen = filedialog.askdirectory(title="Select AI Surrogate Model Bundle Folder", initialdir=bundle_var.get() or None)
        if chosen:
            bundle_var.set(chosen)
            _on_bundle_change()

    def _browse_project() -> None:
        chosen = filedialog.askdirectory(title="Select BurnP3+ Project Folder", initialdir=project_var.get() or None)
        if chosen:
            project_var.set(chosen)
            _on_project_change()

    def _browse_output() -> None:
        nonlocal output_customized
        chosen = filedialog.askdirectory(title="Select Output Folder", initialdir=output_var.get() or None)
        if chosen:
            output_customized = True
            output_var.set(chosen)

    ttk.Label(folders_box, text="Model Bundle:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8), pady=3)
    bundle_entry = ttk.Entry(folders_box, textvariable=bundle_var)
    bundle_entry.grid(row=0, column=1, sticky=tk.EW, pady=3)
    ttk.Button(folders_box, text="Browse...", command=_browse_bundle).grid(row=0, column=2, padx=(8, 0), pady=3)
    ttk.Label(folders_box, textvariable=bundle_status_var, style="Subheader.TLabel").grid(
        row=1, column=1, columnspan=2, sticky=tk.W, pady=(0, 6)
    )

    ttk.Label(folders_box, text="BurnP3+ Project:").grid(row=2, column=0, sticky=tk.W, padx=(0, 8), pady=3)
    project_entry = ttk.Entry(folders_box, textvariable=project_var)
    project_entry.grid(row=2, column=1, sticky=tk.EW, pady=3)
    proj_btn_frame = ttk.Frame(folders_box)
    proj_btn_frame.grid(row=2, column=2, padx=(8, 0), pady=3, sticky=tk.E)
    ttk.Button(proj_btn_frame, text="Browse...", command=_browse_project).pack(side=tk.LEFT)

    def _load_or_create_sample_project() -> None:
        samples = discover_sample_projects()
        if samples:
            project_var.set(str(samples[0]))
            _on_project_change()
            _append_log(f"Loaded bundled sample project: {samples[0]}")
            return
        from inference.build_standalone import SAMPLE_PROJECT_DIRNAME, create_sample_project

        b_str = bundle_var.get().strip()
        loaded: ModelBundle | None = None
        if b_str:
            try:
                loaded = load_bundle(b_str, verify_checksums=False)
            except (BundleError, FileNotFoundError, ValueError):
                loaded = None
        target_dir = Path.cwd().resolve() / SAMPLE_PROJECT_DIRNAME
        created = create_sample_project(target_dir, bundle=loaded)
        project_var.set(str(created))
        _on_project_change()
        _append_log(f"Created and loaded sample project at: {created}")

    ttk.Button(proj_btn_frame, text="Sample", command=_load_or_create_sample_project).pack(side=tk.LEFT, padx=(4, 0))
    ttk.Label(folders_box, textvariable=project_status_var, style="Subheader.TLabel").grid(
        row=3, column=1, columnspan=2, sticky=tk.W, pady=(0, 6)
    )

    ttk.Label(folders_box, text="Output Folder:").grid(row=4, column=0, sticky=tk.W, padx=(0, 8), pady=3)
    output_entry = ttk.Entry(folders_box, textvariable=output_var)
    output_entry.grid(row=4, column=1, sticky=tk.EW, pady=3)
    ttk.Button(folders_box, text="Browse...", command=_browse_output).grid(row=4, column=2, padx=(8, 0), pady=3)

    # 2. Settings Frame
    opts_box = ttk.LabelFrame(outer, text=" 2. Project & Regional Options (Auto-filled from project) ", padding=10)
    opts_box.pack(fill=tk.X, pady=(0, 8))
    for col_idx in (1, 3, 5):
        opts_box.columnconfigure(col_idx, weight=1)

    ttk.Label(opts_box, text="Mask Scope:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=3)
    mask_combo = ttk.Combobox(opts_box, textvariable=mask_var, values=list(MASK_SCOPES), state="readonly", width=12)
    mask_combo.grid(row=0, column=1, sticky=tk.EW, padx=(0, 12), pady=3)

    ttk.Label(opts_box, text="Fuel Scenario:").grid(row=0, column=2, sticky=tk.W, padx=(0, 6), pady=3)
    scenario_combo = ttk.Combobox(opts_box, textvariable=scenario_var, values=[DEFAULT_SCENARIO_LABEL], width=18)
    scenario_combo.grid(row=0, column=3, sticky=tk.EW, padx=(0, 12), pady=3)

    ttk.Label(opts_box, text="Regions (hexNN):").grid(row=0, column=4, sticky=tk.W, padx=(0, 6), pady=3)
    ttk.Entry(opts_box, textvariable=hex_ids_var, width=14).grid(row=0, column=5, sticky=tk.EW, pady=3)

    ttk.Label(opts_box, text="Device:").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=3)
    device_combo = ttk.Combobox(opts_box, textvariable=device_var, values=["auto", "cpu", "cuda", "mps"], state="readonly", width=12)
    device_combo.grid(row=1, column=1, sticky=tk.EW, padx=(0, 12), pady=3)

    ttk.Label(opts_box, text="Fire-Size CSV:").grid(row=1, column=2, sticky=tk.W, padx=(0, 6), pady=3)
    fs_frame = ttk.Frame(opts_box)
    fs_frame.grid(row=1, column=3, columnspan=3, sticky=tk.EW, pady=3)
    fs_frame.columnconfigure(0, weight=1)
    ttk.Entry(fs_frame, textvariable=fire_size_var).grid(row=0, column=0, sticky=tk.EW)

    def _browse_fire_size() -> None:
        chosen = filedialog.askopenfilename(
            title="Select Custom Fire-Size CSV (Optional)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if chosen:
            fire_size_var.set(chosen)

    ttk.Button(fs_frame, text="Browse CSV...", command=_browse_fire_size).grid(row=0, column=1, padx=(6, 0))
    ttk.Button(fs_frame, text="Default", command=lambda: fire_size_var.set("")).grid(row=0, column=2, padx=(4, 0))

    ttk.Label(opts_box, text="Predict Outputs:").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
    rasters_frame = ttk.Frame(opts_box)
    rasters_frame.grid(row=2, column=1, columnspan=5, sticky=tk.W, pady=(6, 0))
    raster_checks: dict[str, ttk.Checkbutton] = {}
    for name, label in PREDICT_RASTER_OPTIONS:
        raster_checks[name] = ttk.Checkbutton(rasters_frame, text=label, variable=raster_vars[name])
        raster_checks[name].pack(side=tk.LEFT, padx=(0, 12))

    chk_frame = ttk.Frame(opts_box)
    chk_frame.grid(row=3, column=0, columnspan=6, sticky=tk.W, pady=(6, 0))
    ttk.Checkbutton(chk_frame, text="Overwrite existing output folder", variable=overwrite_var).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Checkbutton(chk_frame, text="Open HTML summary report when finished", variable=open_report_var).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Checkbutton(chk_frame, text="Save evaluation comparison plots", variable=plots_var).pack(side=tk.LEFT, padx=(0, 16))
    ttk.Checkbutton(chk_frame, text="Per-firezone evaluation metrics", variable=by_firezone_var).pack(side=tk.LEFT)

    # 3. Action Buttons
    actions_box = ttk.Frame(outer)
    actions_box.pack(fill=tk.X, pady=(0, 8))

    btn_check = ttk.Button(actions_box, text="1. Check Inputs", style="Action.TButton", command=lambda: _start_action("check"))
    btn_check.pack(side=tk.LEFT, padx=(0, 8))

    btn_predict = ttk.Button(
        actions_box, text="2. Predict Surrogate Maps", style="Action.TButton", command=lambda: _start_action("predict")
    )
    btn_predict.pack(side=tk.LEFT, padx=(0, 8))

    btn_evaluate = ttk.Button(
        actions_box, text="3. Evaluate vs. BurnP3+", style="Action.TButton", command=lambda: _start_action("evaluate")
    )
    btn_evaluate.pack(side=tk.LEFT, padx=(0, 16))

    btn_open_report = ttk.Button(
        actions_box,
        text="Open HTML Report",
        style="Action.TButton",
        state=tk.DISABLED,
        command=lambda: open_in_os(last_report_path) if last_report_path else None,
    )
    btn_open_report.pack(side=tk.RIGHT)

    btn_open_folder = ttk.Button(
        actions_box,
        text="Open Output Folder",
        style="Action.TButton",
        command=lambda: open_in_os(output_var.get()) if output_var.get().strip() else None,
    )
    btn_open_folder.pack(side=tk.RIGHT, padx=(0, 8))

    # Progress & Status
    prog_frame = ttk.Frame(outer)
    prog_frame.pack(fill=tk.X, pady=(0, 6))
    progress_bar = ttk.Progressbar(prog_frame, mode="indeterminate", length=200)
    progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
    ttk.Label(prog_frame, textvariable=run_status_var, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

    # 4. Log / Output Console
    log_box = ttk.LabelFrame(outer, text=" Activity & Results Log ", padding=8)
    log_box.pack(fill=tk.BOTH, expand=True)
    log_text = tk.Text(
        log_box,
        wrap=tk.WORD,
        font=("Consolas", 9) if platform.system() == "Windows" else ("Monospace", 9),
        bg="#0f172a",
        fg="#f8fafc",
        insertbackground="#f8fafc",
        padx=10,
        pady=8,
    )
    scrollbar = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=log_text.yview)
    log_text.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _append_log(line: str) -> None:
        log_text.insert(tk.END, line + "\n")
        log_text.see(tk.END)

    def _on_bundle_change(*_: object) -> None:
        raw = bundle_var.get().strip()
        if not raw:
            bundle_status_var.set("Select a model bundle folder (contains manifest.yaml & model.pt).")
            return
        try:
            b = load_bundle(raw, verify_checksums=False)
            targets = ", ".join(t.name.upper() for t in b.manifest.targets)
            bundle_status_var.set(f"Loaded bundle: {b.manifest.name} v{b.manifest.version}  |  Targets: {targets}")
            supported = set(available_rasters(b))
            for name, check in raster_checks.items():
                check.configure(state=tk.NORMAL if name in supported else tk.DISABLED)
        except (BundleError, FileNotFoundError, ValueError) as exc:
            bundle_status_var.set(f"Invalid bundle folder: {exc}")

    def _on_project_change(*_: object) -> None:
        raw = project_var.get().strip()
        if not raw:
            project_status_var.set("Select a BurnP3+ project folder (containing hexNN/ subfolders).")
            return
        info = inspect_project_folder(raw)
        if str(info.project_dir) != str(Path(raw).expanduser().resolve()):
            project_var.set(str(info.project_dir))
        project_status_var.set(info.summary_line())
        if info.hex_ids:
            mask_var.set(info.recommended_mask_scope)
            scenario_combo["values"] = [DEFAULT_SCENARIO_LABEL, *info.scenarios]
            scenario_var.set(info.recommended_scenario or DEFAULT_SCENARIO_LABEL)
            if info.selected_single_hexel:
                hex_ids_var.set(f"hex{info.selected_single_hexel}")
            else:
                hex_ids_var.set("all")
            if not output_customized:
                output_var.set(str(info.default_evaluate_output if info.has_burnp3_results else info.default_predict_output))

    bundle_entry.bind("<FocusOut>", _on_bundle_change)
    project_entry.bind("<FocusOut>", _on_project_change)

    def _set_running_state(running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        for btn in (btn_check, btn_predict, btn_evaluate):
            btn.configure(state=state)
        if running:
            progress_bar.start(12)
        else:
            progress_bar.stop()

    def _start_action(action: ActionType) -> None:
        nonlocal output_customized
        b_dir = bundle_var.get().strip()
        p_dir = project_var.get().strip()
        if not b_dir:
            messagebox.showwarning("Missing Model Bundle", "Please select a Model Bundle folder first.")
            return
        if not p_dir:
            messagebox.showwarning("Missing Project Folder", "Please select a BurnP3+ Project folder first.")
            return

        rasters = [name for name, check in raster_checks.items() if raster_vars[name].get() and check.instate(["!disabled"])]
        if action == "predict" and not rasters:
            messagebox.showwarning("No Outputs Selected", "Tick at least one map under 'Predict Outputs'.")
            return

        info = inspect_project_folder(p_dir)
        out_str = output_var.get().strip()
        if not out_str or not output_customized:
            out_path = info.default_evaluate_output if action == "evaluate" else info.default_predict_output
            output_var.set(str(out_path))
        else:
            out_path = Path(out_str)

        fs_str = fire_size_var.get().strip()
        req = WorkflowRequest(
            action=action,
            bundle_dir=Path(b_dir),
            project_dir=info.project_dir,
            output_dir=out_path,
            hex_ids=parse_hex_ids_input(hex_ids_var.get()),
            mask_scope=mask_var.get().strip() or None,
            scenario_name=scenario_name_from_ui(scenario_var.get()),
            fire_size_table=Path(fs_str) if fs_str else None,
            device=device_var.get().strip() or "auto",
            batch_size=max(1, int(batch_size_var.get() or 8)),
            overwrite=bool(overwrite_var.get()),
            plots=bool(plots_var.get()),
            by_firezone=bool(by_firezone_var.get()),
            rasters=rasters,
        )

        _set_running_state(True)
        run_status_var.set(f"Running {action}...")
        _append_log("=" * 76)
        _append_log(f"Starting '{action.upper()}' on project: {req.project_dir}")

        def _worker() -> None:
            handler = _QueueLogHandler(msg_queue)  # type: ignore[arg-type]
            root_logger = logging.getLogger()
            pkg_logger = logging.getLogger("inference")
            prev_level = pkg_logger.level
            root_logger.addHandler(handler)
            pkg_logger.setLevel(logging.INFO)
            try:
                outcome = execute_workflow_action(req)
                msg_queue.put(("done", outcome))
            finally:
                root_logger.removeHandler(handler)
                pkg_logger.setLevel(prev_level)

        threading.Thread(target=_worker, daemon=True).start()

    def _poll_queue() -> None:
        nonlocal last_report_path
        while True:
            try:
                kind, payload = msg_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                _append_log(str(payload))
            elif kind == "done":
                outcome: WorkflowOutcome = payload  # type: ignore[assignment]
                _set_running_state(False)
                _append_log("-" * 76)
                _append_log(outcome.summary_text)
                if outcome.report_html and outcome.report_html.is_file():
                    last_report_path = outcome.report_html
                    btn_open_report.configure(state=tk.NORMAL)
                    if open_report_var.get():
                        open_in_os(outcome.report_html)
                if outcome.ok:
                    run_status_var.set(f"{outcome.action.capitalize()} finished successfully.")
                else:
                    run_status_var.set(f"{outcome.action.capitalize()} stopped with errors.")
                    messagebox.showerror(
                        f"{outcome.action.capitalize()} Error",
                        outcome.error or outcome.summary_text,
                    )
        root.after(100, _poll_queue)

    # Pre-populate bundle / project if discovered or passed on CLI
    if initial_bundle:
        bundle_var.set(str(Path(initial_bundle).expanduser().resolve()))
        _on_bundle_change()
    else:
        found_bundles = discover_bundles()
        if found_bundles:
            bundle_var.set(str(found_bundles[0]))
            _on_bundle_change()
            _append_log(f"Auto-detected model bundle: {found_bundles[0]}")

    if initial_project:
        project_var.set(str(Path(initial_project).expanduser().resolve()))
        _on_project_change()
    else:
        found_samples = discover_sample_projects()
        if found_samples:
            project_var.set(str(found_samples[0]))
            _on_project_change()
            _append_log(f"Auto-loaded bundled sample project: {found_samples[0]}")

    if initial_output:
        output_customized = True
        output_var.set(str(Path(initial_output).expanduser().resolve()))

    _append_log("Welcome! Select your BurnP3+ project folder above and click Check, Predict, or Evaluate.")
    root.after(100, _poll_queue)
    root.mainloop()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference.app",
        description="Launch the Wildfire AI Surrogate desktop app for BurnP3+ projects.",
    )
    parser.add_argument("--bundle", default=None, help="Optional initial model bundle folder.")
    parser.add_argument("--project", default=None, help="Optional initial BurnP3+ project folder.")
    parser.add_argument("--output", default=None, help="Optional initial output folder.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,  # CI-only: run check/predict/evaluate on the bundled sample project without opening the GUI.
    )
    return parser


def run_smoke_test() -> int:
    """Headless self-test of a packaged app: auto-discover the bundle and sample project, then run every workflow."""
    import tempfile

    bundles = discover_bundles()
    if not bundles:
        print("SMOKE TEST FAILED: no model bundle found next to the application.")
        return 1
    samples = discover_sample_projects()
    if not samples:
        print("SMOKE TEST FAILED: no sample_project/ found next to the application.")
        return 1
    print(f"Bundle: {bundles[0]}\nSample project: {samples[0]}")

    info = inspect_project_folder(samples[0])
    actions: list[ActionType] = ["check", "predict"]
    if info.has_burnp3_results:
        actions.append("evaluate")
    out_root = Path(tempfile.mkdtemp(prefix="wildfire_smoke_"))
    for action in actions:
        outcome = execute_workflow_action(
            WorkflowRequest(
                action=action,
                bundle_dir=bundles[0],
                project_dir=info.project_dir,
                output_dir=out_root / action,
                mask_scope=info.recommended_mask_scope,
                scenario_name=info.recommended_scenario,
                device="cpu",
            )
        )
        print(f"[{action}] ok={outcome.ok} report={outcome.report_html}")
        if not outcome.ok:
            print(f"SMOKE TEST FAILED during '{action}': {outcome.error or outcome.summary_text}")
            return 1
    print("SMOKE TEST PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    import multiprocessing

    multiprocessing.freeze_support()
    args = build_arg_parser().parse_args(argv)
    if args.smoke_test:
        if sys.stdout is None:
            # Windowed (no-console) builds have no stdout; write results to a log CI can print.
            import tempfile

            log_stream = open(Path(tempfile.gettempdir()) / "wildfire_smoke_test.log", "w", encoding="utf-8")  # noqa: SIM115
            sys.stdout = sys.stderr = log_stream
        return run_smoke_test()
    return launch_gui(
        initial_bundle=args.bundle,
        initial_project=args.project,
        initial_output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
