from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def run_r_script(
    r_script_path: str | Path,
    output_dir: str | Path,
    fuel_types_path: str | Path | None = None,
    output_filename: str = "fbp_curves_national_fuel.csv",
) -> None:
    """
    Run the R script and pass the output directory as its first argument,
    the curve specs CSV as the second argument, and the output CSV filename
    as the third argument.

    Equivalent command:
        Rscript compute_vector_values_national.R <output_dir> <fuel_types_path> <output_filename>
    """
    r_script_path = Path(r_script_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not r_script_path.is_file():
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    if fuel_types_path is not None:
        fuel_types_path = Path(fuel_types_path).resolve()
        if not fuel_types_path.is_file():
            raise FileNotFoundError(f"Curve specs CSV not found: {fuel_types_path}")
    elif output_filename != "fbp_curves_national_fuel.csv":
        # The R script reads positional args in order (out_dir, fuel_types_path,
        # output_filename), so a custom filename can't be passed without also
        # passing fuel_types_path explicitly.
        raise ValueError("fuel_types_path must be provided when overriding output_filename.")

    rscript_executable = shutil.which("Rscript")

    if rscript_executable is None:
        raise RuntimeError("Rscript was not found on PATH. " "Make sure R is installed and Rscript is accessible.")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        rscript_executable,
        str(r_script_path),
        str(output_dir),
    ]
    if fuel_types_path is not None:
        command.append(str(fuel_types_path))
        command.append(output_filename)

    print("Running command:")
    print(" ".join(command))

    try:
        subprocess.run(
            command,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"R script failed with exit code {exc.returncode}") from exc

    print(f"R script completed successfully.")
    print(f"Outputs saved under: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FBP curve generation R script.")

    parser.add_argument(
        "--r-script",
        type=Path,
        default=Path("data_preparation/tabular/fuel_features/" "compute_vector_values_national.R"),
        help="Path to the R script.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where the R script should save its outputs.",
    )

    parser.add_argument(
        "--fuel_types",
        type=Path,
        default=None,
        help=("Path to the fuel types CSV. " "Defaults to fuel_types_national.csv in the same directory as the R script."),
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        default="fbp_curves_national_fuel.csv",
        help="Filename for the output fuel curve CSV (saved under --output-dir).",
    )

    args = parser.parse_args()

    run_r_script(
        r_script_path=args.r_script,
        output_dir=args.output_dir,
        fuel_types_path=args.fuel_types if args.fuel_types is not None else args.r_script.parent / "fuel_types_national.csv",
        output_filename=args.output_filename,
    )


if __name__ == "__main__":
    main()
