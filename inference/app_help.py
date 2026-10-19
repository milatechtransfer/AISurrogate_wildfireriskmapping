"""Offline help text shown by the desktop app's Help window (``inference.app``)."""

from __future__ import annotations

DOCS_URL = "https://github.com/milatechtransfer/AISurrogate_wildfireriskmapping/blob/main/inference/README.md"

PROJECT_TREE = """\
my_project/                    <- select THIS folder as the BurnP3+ Project
└── hex01/                     <- one folder per region (hex01, hex02, hex100, ...)
    ├── spatial/
    │   ├── hex01_dem.tif                 elevation (~100 m cells)
    │   ├── hex01_fbp.tif                 FBP fuel codes (or hex01_fbp_<scenario>.tif)
    │   ├── hex01_firezones.tif           fire-zone IDs
    │   ├── ignition_grids/
    │   │   └── hex01_ignGrid_{H|N}_<season>.tif   human / lightning ignitions
    │   └── mask_grids/                   optional (see Mask Scope)
    │       ├── hex01_actual.shp
    │       └── hex01_buffer.shp
    ├── tabular/
    │   ├── hex01_DailyWeather.csv
    │   ├── hex01_FireZones.csv
    │   ├── hex01_IgnitionDistribution.csv
    │   ├── hex01_GreenUp.csv
    │   ├── hex01_FuelTypes.csv           optional (new fuel codes)
    │   └── hex01_FuelCodeCrosswalk.csv   optional (new fuel codes)
    └── results/                          BurnP3+ outputs, only needed for Evaluate
"""

OUTPUT_TREE = """\
<output folder>/
├── summary_report.html        opens automatically when a step finishes
├── run_manifest.json          record of the run (model, inputs, options)
├── predict.log                full log (send this when asking for help)
└── hex01/
    ├── hex01_bp.tif           burn probability (0-1)
    ├── hex01_fi.tif           fire intensity (kW/m)
    ├── hex01_ros.tif          rate of spread (m/min)
    └── hex01_hazard_class.tif hazard class (1 = lowest, 0 = no data)
"""

# (heading, kind, text): kind is "text" (wrapped paragraphs) or "code" (monospace, not wrapped).
HELP_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "Quick start",
        "text",
        "1. Model Bundle: already filled in with the model that comes with the app. Leave it as is.\n"
        "2. BurnP3+ Project: click Browse... and select your project folder (the folder that CONTAINS hex01/, "
        "hex02/, ...). New to the app? Click Sample to load a small built-in example.\n"
        "3. The options in section 2 are filled in automatically from your project; usually nothing needs changing.\n"
        "4. Click '1. Check Inputs' first. Fix any errors it reports, then click '2. Predict Surrogate Maps' "
        "or '3. Evaluate vs. BurnP3+'.\n"
        "5. A report opens in your web browser. The maps are GeoTIFFs in the Output Folder: drag them into "
        "QGIS or ArcGIS Pro.",
    ),
    ("Project folder layout", "code", PROJECT_TREE),
    (
        "What the buttons do",
        "text",
        "1. Check Inputs: verifies that every raster and table is present and readable. Writes only a report.\n"
        "2. Predict Surrogate Maps: runs the AI model and writes the maps ticked under Predict Outputs. "
        "Takes seconds to minutes per region on a laptop.\n"
        "3. Evaluate vs. BurnP3+: for projects where BurnP3+ has already been run (a results/ folder). Predicts, "
        "then compares the AI maps with the BurnP3+ maps and reports accuracy scores and plots.",
    ),
    (
        "Options",
        "text",
        "Mask Scope: the area to predict. 'actual' = inside mask_grids/hexNN_actual.shp; 'buffer' = inside "
        "hexNN_buffer.shp; 'none' = the whole raster (use this for regional studies without a mask).\n"
        "Fuel Scenario: which fuel map to use. '(default fuel grid)' = hexNN_fbp.tif. Other names come from "
        "files called hexNN_fbp_<name>.tif (and results/<name>/) found in your project.\n"
        "Regions: 'all', or a list such as 'hex01, hex05'.\n"
        "Device: 'auto' uses a GPU when one is available, otherwise the CPU.\n"
        "Fire-Size CSV: leave empty to use the national fire-size table shipped with the model (Canadian "
        "National Fire Database, fires of 25 ha or more, 1970-2023). To use your own, choose a CSV with "
        "columns GRIDCODE (fire zone) and SIZE_HA.\n"
        "Predict Outputs: the maps to write. Hazard (raw) = BP x capped FI; Hazard (scaled 0-100) is "
        "comparable across studies; Hazard class bins the scaled hazard into classes (1 = lowest).",
    ),
    ("Where the results go", "code", OUTPUT_TREE),
    (
        "Troubleshooting",
        "text",
        "Check Inputs reports errors: each message names the file and what is wrong. Fix it and click Check "
        "Inputs again.\n"
        "Missing mask shapefile: set Mask Scope to 'none'.\n"
        "Unknown fuel codes: add hexNN_FuelTypes.csv and hexNN_FuelCodeCrosswalk.csv to tabular/, as exported "
        "by BurnP3+.\n"
        "Evaluate fails with missing BurnP3+ outputs: Evaluate needs the BurnP3+ results/ folder; use Predict "
        "instead.\n"
        "Output folder not empty: tick 'Overwrite existing output folder' or choose another Output Folder.\n"
        "Out of memory or very slow: predict fewer regions at a time (Regions box) and close other programs.\n"
        "Still stuck: send the summary_report.html and log file from the Output Folder to the developers.",
    ),
    (
        "About the model",
        "text",
        "An AI surrogate (a U-Net) trained to reproduce national BurnP3+ simulations of burn probability, "
        "fire intensity and rate of spread. In a region whose fire regime differs from the national runs it "
        "usually gets the spatial pattern right, but absolute burn probability can be biased: run Evaluate "
        "against a BurnP3+ run of your region before relying on absolute values. Developed by Mila in "
        "collaboration with Natural Resources Canada.",
    ),
)
