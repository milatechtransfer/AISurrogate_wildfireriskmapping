"""Tests for the desktop app logic, HTML report generator, and standalone PyInstaller builder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import inference.app as app_module
from inference.app import (
    APP_TITLE,
    DEFAULT_PREDICT_RASTERS,
    DEFAULT_SCENARIO_LABEL,
    LOGO_FILENAME,
    WorkflowRequest,
    discover_bundles,
    discover_sample_projects,
    execute_workflow_action,
    find_logo_path,
    inspect_project_folder,
    parse_hex_ids_input,
    run_smoke_test,
    scenario_name_from_ui,
)
from inference.app_help import HELP_SECTIONS
from inference.build_standalone import (
    APP_NAME,
    EXCLUDED_MODULES,
    HIDDEN_IMPORTS,
    SAMPLE_PROJECT_DIRNAME,
    build_pyinstaller_args,
    create_sample_project,
    stage_bundle_into_app,
    write_quickstart_txt,
)
from inference.bundle import load_bundle
from inference.report import SUMMARY_REPORT_FILENAME
from tests.test_evaluate import _write_burnp3_results
from tests.test_predict import HEIGHT, WIDTH, _write_hexel, _write_raster, make_test_bundle


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_test_bundle(tmp_path_factory.mktemp("app_bundle"))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_hexel(project_dir)
    return project_dir


def test_discover_bundles_finds_nested_bundles(bundle_dir: Path, tmp_path: Path):
    app_root = tmp_path / "app_root"
    staged = stage_bundle_into_app(bundle_dir, app_root)
    (app_root / "not_a_bundle").mkdir(parents=True)

    found = discover_bundles([app_root])
    assert found == [staged.resolve()]


def test_inspect_project_folder_handles_standard_and_regional_layouts(project: Path):
    info = inspect_project_folder(project)
    assert info.project_dir == project.resolve()
    assert info.hex_ids == ["01"]
    assert info.has_actual_mask is True
    assert info.recommended_mask_scope == "actual"
    assert info.has_burnp3_results is False
    assert info.recommended_scenario is None
    assert "Inputs only (Predict ready)" in info.summary_line()

    # Selecting hex01/ or hex01/spatial/ directly steps up to the project root and preselects hex01
    sub_info = inspect_project_folder(project / "hex01" / "spatial")
    assert sub_info.project_dir == project.resolve()
    assert sub_info.selected_single_hexel == "01"

    # Regional layout: no mask shapefiles, scenario fuel raster, and BurnP3+ scenario outputs
    for shp in (project / "hex01" / "spatial" / "mask_grids").iterdir():
        shp.unlink()
    default_fbp = project / "hex01" / "spatial" / "hex01_fbp.tif"
    scenario_fbp = project / "hex01" / "spatial" / "hex01_fbp_FireExcludeSpotting.tif"
    default_fbp.rename(scenario_fbp)
    _write_burnp3_results(project, scenario_name="FireExcludeSpotting")

    reg_info = inspect_project_folder(project)
    assert reg_info.recommended_mask_scope == "none"
    assert reg_info.scenarios == ["FireExcludeSpotting"]
    assert reg_info.recommended_scenario == "FireExcludeSpotting"
    assert reg_info.has_burnp3_results is True
    assert "BurnP3+ results found" in reg_info.summary_line()


def test_parse_hex_ids_input():
    assert parse_hex_ids_input("") is None
    assert parse_hex_ids_input("  all ") is None
    assert parse_hex_ids_input("hex01, 12  hex14") == ["hex01", "12", "hex14"]


def test_execute_workflow_check_predict_and_evaluate_write_html_reports(bundle_dir: Path, project: Path, tmp_path: Path):
    _write_burnp3_results(project)

    # 1. Check
    check_out = tmp_path / "check_out"
    check_res = execute_workflow_action(
        WorkflowRequest(
            action="check",
            bundle_dir=bundle_dir,
            project_dir=project,
            output_dir=check_out,
        )
    )
    assert check_res.ok is True
    assert check_res.report_html == check_out / SUMMARY_REPORT_FILENAME
    check_html = check_res.report_html.read_text(encoding="utf-8")
    assert "Wildfire AI Surrogate — Input Check Report" in check_html
    assert "hex01" in check_html

    # 2. Predict
    pred_out = tmp_path / "pred_out"
    pred_res = execute_workflow_action(
        WorkflowRequest(
            action="predict",
            bundle_dir=bundle_dir,
            project_dir=project,
            output_dir=pred_out,
            device="cpu",
            rasters=list(DEFAULT_PREDICT_RASTERS),
        )
    )
    assert pred_res.ok is True
    assert sorted(p.relative_to(pred_out).as_posix() for p in pred_out.glob("*/*.tif")) == sorted(
        f"{n}/hex01_{n}.tif" for n in DEFAULT_PREDICT_RASTERS
    )
    assert sorted(p.relative_to(pred_out).as_posix() for p in pred_out.glob("*/*.png")) == sorted(
        f"{n}/hex01_{n}.png" for n in DEFAULT_PREDICT_RASTERS
    )
    assert pred_res.report_html == pred_out / SUMMARY_REPORT_FILENAME
    pred_html = pred_res.report_html.read_text(encoding="utf-8")
    assert "Wildfire AI Surrogate — Prediction Summary" in pred_html
    assert "bp/hex01_bp.tif" in pred_html
    assert '<img src="hazard_class/hex01_hazard_class.png"' in pred_html
    assert "hazard_class" in pred_html
    assert "hazard_scaled" not in pred_html

    # 3. Evaluate (and verify overwrite cleans up previous summary_report.html cleanly)
    eval_out = tmp_path / "eval_out"
    eval_res = execute_workflow_action(
        WorkflowRequest(
            action="evaluate",
            bundle_dir=bundle_dir,
            project_dir=project,
            output_dir=eval_out,
            device="cpu",
            plots=False,
        )
    )
    assert eval_res.ok is True
    assert eval_res.report_html == eval_out / SUMMARY_REPORT_FILENAME
    eval_html = eval_res.report_html.read_text(encoding="utf-8")
    assert "Wildfire AI Surrogate — BurnP3+ Evaluation Report" in eval_html
    assert "Summary Metrics" in eval_html
    assert "Wildfire Hazard Class Agreement" in eval_html
    assert 'class="table-scroll"' in eval_html
    assert ">iou_top10</th>" in eval_html
    assert ">mae_top10</th>" not in eval_html  # full metric set stays in metrics_summary.csv


def test_execute_workflow_reports_input_errors_cleanly(bundle_dir: Path, project: Path, tmp_path: Path):
    _write_raster(project / "hex01" / "spatial" / "hex01_fbp.tif", np.full((HEIGHT, WIDTH), 999, dtype=np.int16), -9999)

    check_res = execute_workflow_action(
        WorkflowRequest(
            action="check",
            bundle_dir=bundle_dir,
            project_dir=project,
            output_dir=tmp_path / "check_err",
        )
    )
    assert check_res.ok is False
    assert check_res.report_html is not None
    assert "badge-error" in check_res.report_html.read_text(encoding="utf-8")


def test_build_standalone_helpers(bundle_dir: Path, tmp_path: Path):
    args = build_pyinstaller_args(
        entry_script=Path("inference/app.py"),
        dist_dir=tmp_path / "dist",
        work_dir=tmp_path / "work",
    )
    assert "--windowed" in args
    assert APP_NAME in args
    for mod in HIDDEN_IMPORTS:
        assert mod in args

    app_dir = tmp_path / "dist" / APP_NAME
    staged = stage_bundle_into_app(bundle_dir, app_dir)
    assert (staged / "manifest.yaml").is_file()
    sample_dir = create_sample_project(app_dir / SAMPLE_PROJECT_DIRNAME, bundle=load_bundle(bundle_dir))
    assert discover_sample_projects([app_dir]) == [sample_dir.resolve()]
    check_res = execute_workflow_action(
        WorkflowRequest(
            action="check",
            bundle_dir=bundle_dir,
            project_dir=sample_dir,
            output_dir=tmp_path / "sample_check_out",
        )
    )
    assert check_res.ok is True
    readme = write_quickstart_txt(app_dir)
    assert readme.is_file()
    assert "BurnP3+ Project" in readme.read_text(encoding="utf-8")


def test_app_imports_without_modules_excluded_from_standalone_build():
    """The packaged app omits EXCLUDED_MODULES; importing the app must not require any of them."""
    blocked = ", ".join(repr(m) for m in EXCLUDED_MODULES)
    code = f"import sys\nfor m in ({blocked},):\n    sys.modules[m] = None\nimport inference.app\n"
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_smoke_test_runs_all_workflows_on_staged_app(bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    app_dir = tmp_path / APP_NAME
    stage_bundle_into_app(bundle_dir, app_dir)
    create_sample_project(app_dir / SAMPLE_PROJECT_DIRNAME, bundle=load_bundle(bundle_dir))
    monkeypatch.chdir(app_dir)
    assert run_smoke_test() == 0


def test_scenario_name_from_ui_maps_default_label_to_none():
    assert scenario_name_from_ui(DEFAULT_SCENARIO_LABEL) is None
    assert scenario_name_from_ui("  ") is None
    assert scenario_name_from_ui(" FireExcludeSpotting ") == "FireExcludeSpotting"


def test_help_sections_document_project_layout_and_troubleshooting():
    headings = [heading for heading, _kind, _text in HELP_SECTIONS]
    assert "Troubleshooting" in headings
    all_text = "\n".join(text for _h, _k, text in HELP_SECTIONS)
    for expected in ("hexNN_fbp", "DailyWeather", "results", "summary_report.html"):
        assert expected in all_text
    assert all(kind in {"text", "code"} for _h, kind, _t in HELP_SECTIONS)


def test_find_logo_path_returns_none_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module, "ASSETS_DIR", tmp_path / "missing")
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert find_logo_path() is None

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / LOGO_FILENAME).write_bytes(b"png")
    monkeypatch.setattr(app_module, "ASSETS_DIR", assets)
    assert find_logo_path() == assets / LOGO_FILENAME


def test_pyinstaller_args_bundle_logo_only_when_present(tmp_path: Path):
    args = build_pyinstaller_args(Path("inference/app.py"), tmp_path / "dist", tmp_path / "build")
    logo = Path(__file__).resolve().parents[1] / "inference" / "assets" / LOGO_FILENAME
    assert ("--add-data" in args) == logo.is_file()
    assert "inference.app_help" in HIDDEN_IMPORTS
    assert APP_TITLE == "Wildfire AI Surrogate"
