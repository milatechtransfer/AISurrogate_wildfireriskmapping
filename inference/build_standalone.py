"""
Build a standalone, zero-install desktop application bundle using PyInstaller.

The resulting folder (and optional ``.zip`` archive) contains a native executable
(``WildfireSurrogateApp.exe`` on Windows, ``WildfireSurrogateApp.app`` on macOS, or
``WildfireSurrogateApp`` on Linux), all required Python/PyTorch/GDAL libraries, and (when
``--bundle`` is given) the pre-loaded model bundle under ``bundles/``.

Scientists simply unzip the archive and double-click ``WildfireSurrogateApp`` — no Python,
``uv``, or internet connection is required on their computer.

Example (run by maintainers on the target OS or in GitHub Actions):
    uv run --with pyinstaller python -m inference.build_standalone \\
        --bundle nrcan-surrogate-bp-fi-ros-v1.0 \\
        --out_dir dist \\
        --zip
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box

from data_preparation.paths import Paths
from inference.bundle import (
    RESOURCE_FIRE_SIZE_TABLE,
    RESOURCE_FUEL_CURVES,
    BundleError,
    ModelBundle,
    load_bundle,
)
from inference.export_bundle import export_bundle
from src.config import Config
from src.models.factory import build_model

APP_NAME = "WildfireSurrogateApp"
SAMPLE_PROJECT_DIRNAME = "sample_project"
DEMO_BUNDLE_DIRNAME = "demo-surrogate-bp-fi-ros"

# Modules imported dynamically at runtime (e.g. via src.datasets.utils.get_data_source_class
# or Cython/GDAL extensions inside rasterio and pyogrio).
HIDDEN_IMPORTS: tuple[str, ...] = (
    "src.datasets.sources",
    "src.datasets.sources.grids",
    "src.datasets.sources.tabular",
    "src.datasets.sources.spatialized_tabular",
    "rasterio._shim",
    "rasterio.control",
    "rasterio.crs",
    "rasterio.sample",
    "rasterio.vrt",
    "rasterio._features",
    "pyogrio._geometry",
    "pyogrio._io",
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
)

# Geospatial packages whose PROJ/GDAL data files must be bundled for offline CRS reprojection.
COLLECT_DATA_PACKAGES: tuple[str, ...] = (
    "rasterio",
    "pyproj",
    "pyogrio",
)

# Training- or test-only packages excluded to keep the standalone archive smaller.
EXCLUDED_MODULES: tuple[str, ...] = (
    "comet_ml",
    "pytest",
    "xgboost",
    "IPython",
    "notebook",
)


def build_pyinstaller_args(
    entry_script: Path,
    dist_dir: Path,
    work_dir: Path,
    app_name: str = APP_NAME,
    console: bool = False,
) -> list[str]:
    """Construct the PyInstaller CLI arguments for freezing ``inference/app.py``."""
    repo_root = Path(__file__).resolve().parents[1]
    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--console" if console else "--windowed",
        "--name",
        app_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(work_dir),
        "--paths",
        str(repo_root),
    ]
    for mod in HIDDEN_IMPORTS:
        args.extend(["--hidden-import", mod])
    for pkg in COLLECT_DATA_PACKAGES:
        args.extend(["--collect-data", pkg, "--collect-submodules", pkg])
    for excl in EXCLUDED_MODULES:
        args.extend(["--exclude-module", excl])
    args.append(str(entry_script))
    return args


def _write_sample_raster(path: Path, data: np.ndarray, origin_x: float, origin_y: float, cell_m: float, crs: str, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": from_origin(origin_x, origin_y, cell_m, cell_m),
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def create_sample_project(
    project_dir: str | Path,
    bundle: ModelBundle | None = None,
    hex_id: str = "01",
    include_burnp3_results: bool = True,
) -> Path:
    """Create a small, ready-to-run synthetic BurnP3+ project (``hex01/``) compatible with ``bundle``."""
    out_root = Path(project_dir).expanduser().resolve()
    if out_root.exists():
        shutil.rmtree(out_root)

    crs = bundle.manifest.inputs.crs if bundle is not None else "ESRI:102002"
    cell_m = float(bundle.manifest.inputs.resolution_m) if bundle is not None else 100.0

    pad = 4
    height, width = 40, 48
    origin_x, origin_y = 1_000_000.0, 1_000_000.0

    fuel_codes = (1, 13)
    if bundle is not None:
        curves_df = pd.read_csv(bundle.resource_path(RESOURCE_FUEL_CURVES))
        available_codes = sorted({int(c) for c in curves_df["fbp_code"].dropna().unique() if int(c) > 0})
        if len(available_codes) >= 2:
            fuel_codes = (available_codes[0], available_codes[1])
        elif len(available_codes) == 1:
            fuel_codes = (available_codes[0], available_codes[0])

    zone_ids = (21, 22)
    if bundle is not None:
        fs_path = bundle.optional_resource_path(RESOURCE_FIRE_SIZE_TABLE)
        if fs_path is not None and fs_path.is_file():
            fs_df = pd.read_csv(fs_path)
            col = "GRIDCODE" if "GRIDCODE" in fs_df.columns else ("FRU" if "FRU" in fs_df.columns else None)
            if col is not None:
                avail_zones = sorted({int(z) for z in fs_df[col].dropna().unique() if int(z) > 0})
                if len(avail_zones) >= 2:
                    zone_ids = (avail_zones[0], avail_zones[1])
                elif len(avail_zones) == 1:
                    zone_ids = (avail_zones[0], avail_zones[0])

    hexel_dir = out_root / f"hex{hex_id}"
    spatial = hexel_dir / "spatial"
    tabular = hexel_dir / "tabular"
    rows, cols = np.mgrid[0:height, 0:width]

    dem = (200.0 + 3.0 * (rows / max(1, height)) * 50.0 + 2.0 * (cols / max(1, width)) * 50.0).astype(np.float32)
    fbp = np.where(cols < width // 2, fuel_codes[0], fuel_codes[1]).astype(np.int16)
    fzones = np.where(rows < height // 2, zone_ids[0], zone_ids[1]).astype(np.int16)

    _write_sample_raster(spatial / f"hex{hex_id}_dem.tif", dem, origin_x, origin_y, cell_m, crs, -9999.0)
    _write_sample_raster(spatial / f"hex{hex_id}_fbp.tif", fbp, origin_x, origin_y, cell_m, crs, -9999)
    _write_sample_raster(spatial / f"hex{hex_id}_firezones.tif", fzones, origin_x, origin_y, cell_m, crs, -9999)

    for cause in ("H", "N"):
        for season_idx, season in enumerate(("s1", "s2")):
            ign = (0.01 * (1 + season_idx) + 0.0005 * (cols % 20)).astype(np.float32)
            _write_sample_raster(
                spatial / "ignition_grids" / f"hex{hex_id}_ignGrid_{cause}_{season}.tif",
                ign,
                origin_x,
                origin_y,
                cell_m,
                crs,
                -9999.0,
            )

    mask_poly = box(
        origin_x + pad * cell_m,
        origin_y - (height - pad) * cell_m,
        origin_x + (width - pad) * cell_m,
        origin_y - pad * cell_m,
    )
    (spatial / "mask_grids").mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"id": [1]}, geometry=[mask_poly], crs=crs).to_file(spatial / "mask_grids" / f"hex{hex_id}_actual.shp")

    tabular.mkdir(parents=True, exist_ok=True)
    weather_zones = {zid: f"fru{zid}" for zid in dict.fromkeys(zone_ids)}
    weather_rows = [
        {
            "Order": day + 1,
            "Season": season,
            "WeatherZone": zname,
            "Temperature": 18.0 + day,
            "RelativeHumidity": 30 + day,
            "WindSpeed": 15 + day,
            "WindDirection": 180 + 20 * day,
            "Precipitation": 0.0,
            "FineFuelMoistureCode": 90.0,
            "DuffMoistureCode": 40.0,
            "DroughtCode": 300.0,
            "InitialSpreadIndex": 10.0 + day,
            "BuildupIndex": 50.0,
            "FireWeatherIndex": 20.0,
        }
        for zname in weather_zones.values()
        for season in ("s1", "s2")
        for day in range(3)
    ]
    pd.DataFrame(weather_rows).to_csv(tabular / f"hex{hex_id}_DailyWeather.csv", index=False)
    pd.DataFrame({"Name": list(weather_zones.values()), "ID": list(weather_zones.keys())}).to_csv(
        tabular / f"hex{hex_id}_FireZones.csv", index=False
    )
    dist_rows = [
        {"Iteration": "", "Timestep": "", "Season": season, "Cause": cause, "FireZone": zname, "RelativeLikelihood": weight}
        for season, weight in (("s1", 1.0), ("s2", 3.0))
        for cause in ("Human", "Lightning")
        for zname in weather_zones.values()
    ]
    pd.DataFrame(dist_rows).to_csv(tabular / f"hex{hex_id}_IgnitionDistribution.csv", index=False)
    pd.DataFrame({"Season": ["s1", "s2"], "GreenUp": ["No", "Yes"]}).to_csv(tabular / f"hex{hex_id}_GreenUp.csv", index=False)

    if include_burnp3_results:
        paths = Paths(hex_id=hex_id, root_dir=out_root)
        bp = (0.002 + 0.01 * (rows / max(1, height)) + 0.005 * (cols / max(1, width))).astype(np.float32)
        fi = (500.0 + 1200.0 * (cols / max(1, width)) + 300.0 * (rows / max(1, height))).astype(np.float32)
        ros = (1.0 + 4.0 * (rows / max(1, height))).astype(np.float32)
        for grid, out_path in ((bp, paths.output_burn_prob()), (fi, paths.output_fire_intensity()), (ros, paths.output_ros())):
            _write_sample_raster(out_path, grid, origin_x, origin_y, cell_m, crs, -9999.0)

    return out_root


def create_demo_bundle(out_dir: str | Path) -> Path:
    """Create a lightweight, self-contained demo model bundle so standalone builds work out-of-the-box."""
    repo_root = Path(__file__).resolve().parents[1]
    q3_config_path = repo_root / "configs" / "multi_output_spatial_weather_firesize_q3.yaml"
    num_isi_bins = 5
    fuel_log_mean, fuel_log_std = 2.5, 1.25
    feature_channel_map = {
        "fuel_grid": [0],
        "elevation_grid": [1],
        "ignition_grid_human": [2],
        "ignition_grid_lightning": [3],
        "firezones_grid": [4],
        "bp_out_grid": [5],
        "fi_out_grid": [6],
        "ros_out_grid": [7],
    }
    z_cols = [
        "Temperature",
        "WindSpeed",
        "DuffMoistureCode",
        "DroughtCode",
        "InitialSpreadIndex",
        "BuildupIndex",
        "FireWeatherIndex",
        "wind_x",
        "wind_y",
    ]

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        data_root = tmp / "data_samples"
        data_root.mkdir(parents=True)
        norm_stats = {
            "elevation": {"min": -10.0, "max": 3000.0},
            "fuel_curve_iROS": {"log_mean": fuel_log_mean, "log_std": fuel_log_std},
            "fire_intensity": {"log_mean": 7.9, "log_std": 1.1},
            "fire_ros": {"log_mean": 1.8, "log_std": 0.55},
            "fire_burn_probability": {"min": 1e-5, "max": 0.12},
        }
        (data_root / "dataset_norm_stats.json").write_text(json.dumps(norm_stats))
        (data_root / "weather_norm_params.json").write_text(
            json.dumps(
                {
                    "min_max": {"cols": ["RelativeHumidity", "FineFuelMoistureCode"], "min": [0.0, 77.7], "max": [100.0, 99.0]},
                    "z_score": {"cols": z_cols, "mean": [20.0] * len(z_cols), "std": [5.0] * len(z_cols)},
                }
            )
        )
        (data_root / "fire_size_norm_params.json").write_text(json.dumps({"log_size_min": 0.0, "log_size_max": 6.0}))
        (data_root / "feature_channel_map_1.json").write_text(json.dumps(feature_channel_map))
        curve_rows = [
            {"fbp_code": code, "SeasonState": state, "ISI": isi, "ROS": float(code + isi), "HFI": float(code * isi)}
            for code, states in ((1, ["direct"]), (13, ["green", "leafless"]))
            for state in states
            for isi in range(num_isi_bins)
        ]
        pd.DataFrame(curve_rows).to_csv(data_root / "fbp_curves_national_fuel.csv", index=False)

        cfg_dict = yaml.safe_load(q3_config_path.read_text())
        cfg_dict["model"]["hidden_features"] = [4, 8]
        cfg_dict["data"]["root_dir"] = str(data_root)
        cfg_dict["data"]["raw_data_dir"] = str(tmp)
        config = Config(**cfg_dict)

        torch.manual_seed(0)
        model = build_model(
            model_config=config.model,
            spatial_input_channels=20,
            auxiliary_input_dims={"fuel_curve": num_isi_bins},
            fuel_curve_mean=torch.tensor([fuel_log_mean]),
            fuel_curve_std=torch.tensor([fuel_log_std]),
            target_names=["bp", "fi", "ros"],
        ).eval()

        ckpt_path = tmp / "run" / "best.pth"
        ckpt_path.parent.mkdir(parents=True)
        torch.save({"model_state": model.state_dict(), "epoch": 1, "config": config.model_dump()}, ckpt_path)

        denom_json = tmp / "hazard_scale_denominator.json"
        denom_json.write_text(json.dumps({"scale_denominator": 50.0, "scale_denominator_source": "demo"}))
        fs_table = tmp / "df_fire_fru_demo.csv"
        pd.DataFrame({"GRIDCODE": [21, 21, 21, 22, 22, 22], "SIZE_HA": [30.0, 120.0, 900.0, 50.0, 400.0, 5000.0]}).to_csv(
            fs_table, index=False
        )

        return export_bundle(
            checkpoint_path=ckpt_path,
            out_dir=Path(out_dir),
            name="demo-surrogate-bp-fi-ros",
            version="0.1.0-demo",
            description="Lightweight demo bundle included with the standalone desktop application for immediate testing.",
            hazard_denominator_json=denom_json,
            fire_size_table=fs_table,
            fire_size_table_note="Synthetic demo fire-size table (fire zones 21 and 22).",
            selection_note="Demo weights for UI verification; replace with nrcan-surrogate-bp-fi-ros-v1.0 for production.",
            overwrite=True,
        )


def stage_bundle_into_app(bundle_dir: Path, app_dist_dir: Path) -> Path:
    """Validate and copy a model bundle into ``<app_dist_dir>/bundles/<bundle_name>``."""
    bundle = load_bundle(bundle_dir, verify_checksums=True)
    target_dir = app_dist_dir / "bundles" / bundle.root.name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle.root, target_dir)
    return target_dir


def write_quickstart_txt(app_dist_dir: Path, app_name: str = APP_NAME) -> Path:
    """Write a short plain-text README inside the standalone distribution folder."""
    if platform.system() == "Windows":
        exe_label = f"{app_name}.exe"
    elif platform.system() == "Darwin":
        exe_label = f"{app_name}.app"
    else:
        exe_label = app_name
    text = (
        "NRCan Wildfire AI Surrogate — Standalone Desktop Application\n"
        "=============================================================\n\n"
        "No Python or package installation is needed.\n\n"
        "How to run:\n"
        f"  1. Double-click '{exe_label}' in this folder.\n"
        "     (On macOS, if prompted about an unverified developer on first launch,\n"
        f"      right-click '{exe_label}' -> Open, or run: xattr -cr {exe_label})\n"
        "  2. Click 'Sample' to test immediately with the bundled sample BurnP3+ project,\n"
        "     or click 'Browse...' next to 'BurnP3+ Project' to select your own project folder.\n"
        "  3. Click:\n"
        "     - '1. Check Inputs' to verify all rasters and tables,\n"
        "     - '2. Predict Surrogate Maps' to generate BP, FI, ROS, and Hazard GeoTIFFs, or\n"
        "     - '3. Evaluate vs. BurnP3+' if your project also has BurnP3+ results/ rasters.\n"
        "  4. When complete, the visual HTML summary report opens automatically.\n"
    )
    readme_path = app_dist_dir / "README_QUICKSTART.txt"
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def create_platform_zip_archive(dist_dir: Path, app_name: str, archive_base: Path) -> Path:
    """Create a distributable platform archive (``.dmg`` on macOS, ``.zip`` on Windows/Linux)."""
    app_dist_dir = dist_dir / app_name

    # On macOS, Finder's GUI Archive Utility rejects .zip files containing PyInstaller's relative `../..`
    # framework symlinks. Packaging as a native compressed .dmg disk image avoids Archive Utility symlink
    # restrictions and lets users open the app with a double-click in Finder.
    if platform.system() == "Darwin" and shutil.which("hdiutil") is not None:
        dmg_path = archive_base.with_suffix(".dmg")
        if dmg_path.exists():
            dmg_path.unlink()
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                app_name,
                "-srcfolder",
                str(app_dist_dir),
                "-ov",
                "-format",
                "UDZO",
                str(dmg_path),
            ],
            check=True,
        )
        return dmg_path

    zip_path = archive_base.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    created = shutil.make_archive(str(archive_base), "zip", root_dir=dist_dir, base_dir=app_name)
    return Path(created)


def build_standalone_app(
    out_dir: str | Path = "dist",
    bundle_dir: str | Path | None = None,
    app_name: str = APP_NAME,
    make_zip: bool = False,
    console: bool = False,
    include_sample_project: bool = True,
) -> Path:
    """Run PyInstaller, embed the optional model bundle and sample project, and optionally create a release ``.zip``."""
    repo_root = Path(__file__).resolve().parents[1]
    entry_script = repo_root / "inference" / "app.py"
    dist_dir = Path(out_dir).expanduser().resolve()
    work_dir = dist_dir / "_pyinstaller_build"

    loaded_bundle: ModelBundle | None = None
    if bundle_dir is not None:
        loaded_bundle = load_bundle(bundle_dir, verify_checksums=True)

    try:
        import PyInstaller.__main__ as pyinstaller_main  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        pyinstaller_main = None

    pyi_args = build_pyinstaller_args(entry_script, dist_dir=dist_dir, work_dir=work_dir, app_name=app_name, console=console)
    if pyinstaller_main is not None:
        pyinstaller_main.run(pyi_args)
    else:
        pyi_bin = shutil.which("pyinstaller")
        if pyi_bin is None:
            raise RuntimeError(
                "PyInstaller is not installed in the current environment. Run with:\n"
                "  uv run --with pyinstaller python -m inference.build_standalone ..."
            )
        subprocess.run([pyi_bin, *pyi_args], check=True)

    shutil.rmtree(work_dir, ignore_errors=True)
    app_dist_dir = dist_dir / app_name

    # On macOS, PyInstaller --windowed also creates <dist_dir>/<app_name>.app.
    # Place the .app bundle inside <app_dist_dir>/ so the archive contains the native macOS app alongside bundles/ and sample_project/.
    mac_app = dist_dir / f"{app_name}.app"
    if platform.system() == "Darwin" and mac_app.is_dir():
        shutil.rmtree(app_dist_dir, ignore_errors=True)
        app_dist_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mac_app), str(app_dist_dir / f"{app_name}.app"))

    if loaded_bundle is not None:
        stage_bundle_into_app(loaded_bundle.root, app_dist_dir)
    else:
        demo_dir = create_demo_bundle(app_dist_dir / "bundles" / DEMO_BUNDLE_DIRNAME)
        loaded_bundle = load_bundle(demo_dir, verify_checksums=False)
    if include_sample_project:
        create_sample_project(app_dist_dir / SAMPLE_PROJECT_DIRNAME, bundle=loaded_bundle)
    write_quickstart_txt(app_dist_dir, app_name=app_name)

    if make_zip:
        sys_tag = f"{platform.system().lower()}-{platform.machine().lower()}"
        archive_base = dist_dir / f"{app_name}-{sys_tag}"
        return create_platform_zip_archive(dist_dir=dist_dir, app_name=app_name, archive_base=archive_base)
    return app_dist_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference.build_standalone",
        description="Package the Wildfire AI Surrogate desktop app into a standalone zero-install executable.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Optional model bundle folder to embed inside the standalone app (under bundles/).",
    )
    parser.add_argument("--out_dir", default="dist", help="Output directory for the built application (default: dist/).")
    parser.add_argument("--name", default=APP_NAME, help=f"Application executable/folder name (default: {APP_NAME}).")
    parser.add_argument("--zip", action="store_true", help="Also package the built folder into a platform-tagged .zip archive.")
    parser.add_argument("--no_sample_project", action="store_true", help="Skip generating the embedded sample_project/ folder.")
    parser.add_argument("--console", action="store_true", help="Keep a terminal console attached (useful for debugging builds).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        artifact = build_standalone_app(
            out_dir=args.out_dir,
            bundle_dir=args.bundle,
            app_name=args.name,
            make_zip=args.zip,
            console=args.console,
            include_sample_project=not args.no_sample_project,
        )
    except (BundleError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Built standalone desktop artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
