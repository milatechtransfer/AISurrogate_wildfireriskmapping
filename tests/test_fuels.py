"""Fuel curves for fuel codes the model's fuel table does not list, derived from the project's BurnP3+ fuel tables."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.paths import Paths
from inference.fuels import (
    FuelComponent,
    FuelDefinitionError,
    fbp_ros_curve,
    parse_crosswalk_code,
    read_project_fuel_codes,
    resolve_fuel_curves,
)

ISI = np.array([5.0, 20.0, 50.0, 85.0])
# ROS (m/min) from compute_vector_values_national.R (cffdrs::fbp, ISI given, BUIEff 0, BUI 60, curing 90%).
R_CURVES = {
    ("C-1", None): (0.279374341158146, 21.4259099174262, 75.2595611560611, 88.383358110062),
    ("C-2", None): (5.24607217572611, 31.132658742346, 72.2856183222423, 95.3334493412853),
    ("C-3", None): (0.867972407278204, 22.4223456864849, 77.9107276985396, 102.595819578309),
    ("C-4", None): (5.53367272579844, 32.4833710119634, 74.167923943615, 96.6140048037256),
    ("C-5", None): (0.224911290401104, 9.58971070938714, 26.4872565327914, 29.6805066643034),
    ("C-7", None): (0.900251391547729, 9.38377929487566, 27.5452564797304, 38.5170519672142),
    ("D-1", None): (0.871647837396237, 6.14557076694567, 16.4346367228985, 23.603828390574),
    ("D-2", None): (1e-06, 1e-06, 1e-06, 1e-06),
    ("O-1a", None): (10.0414293715535, 51.5721778741259, 108.840099064752, 136.960348996428),
    ("O-1b", None): (8.923500170624, 62.273605094578, 144.57663213442, 182.955577312237),
    ("NF", None): (0.0, 0.0, 0.0, 0.0),
    ("M-1", 25.0): (1.9652539219787, 12.3923427607957, 30.3973821227345, 41.5362336282519),
    ("M-2", 5.0): (0.42791669789159, 2.72429138283697, 6.73686189346284, 9.25139986127333),
    ("M-2", 80.0): (4.23172365407674, 25.1519498245546, 58.4858801267098, 77.2109126086512),
    ("M-1", 80.0): (4.37118730806014, 26.1352411472659, 61.1154220023736, 80.9875251511431),
}


@pytest.mark.parametrize(("fuel_type", "percent_conifer"), list(R_CURVES))
def test_fbp_curves_reproduce_the_r_fuel_table(fuel_type: str, percent_conifer: float | None):
    curve = fbp_ros_curve(FuelComponent("any", fuel_type, percent_conifer), ISI)
    np.testing.assert_allclose(curve, R_CURVES[(fuel_type, percent_conifer)], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("C-2", [FuelComponent("direct", "C-2")]),
        ("D-1/D-2", [FuelComponent("leafless", "D-1"), FuelComponent("green", "D-2")]),
        ("M-2 (05 PC)", [FuelComponent("green", "M-2", 5.0)]),
        ("M-1/M-2 (80 PC)", [FuelComponent("leafless", "M-1", 80.0), FuelComponent("green", "M-2", 80.0)]),
        ("Non-fuel", [FuelComponent("nonfuel", "NF")]),
    ],
)
def test_crosswalk_codes_are_parsed_into_season_components(code: str, expected: list[FuelComponent]):
    assert list(parse_crosswalk_code(code)) == expected


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("C-6", "fuel type C-6 needs curves from the R script"),
        ("M-3/M-4 (35 PDF)", "fuel type M-3/M-4 needs curves from the R script"),
        ("S-1", "fuel type S-1 needs curves from the R script"),
        ("M-1/M-2", "mixedwood needs a percent conifer, e.g. M-1 (25 PC)"),
        ("M-2 (120 PC)", "percent conifer must be between 0 and 100"),
        ("C-2 (50 PC)", "PC only applies to mixedwood fuel types"),
        ("Z-9", "unknown FBP fuel type Z-9"),
        ("grass", "unrecognised FBP code 'grass'"),
    ],
)
def test_crosswalk_codes_without_a_derivable_curve_say_why(code: str, reason: str):
    with pytest.raises(FuelDefinitionError, match=reason.replace("(", r"\(").replace(")", r"\)")):
        parse_crosswalk_code(code)


def _model_curves(isi: np.ndarray = ISI) -> pd.DataFrame:
    """A model table like the national one: C-2 (code 2) and M-1/M-2 at 50% conifer (code 650)."""
    rows = [(2, "C-2", "C-2", "direct", FuelComponent("direct", "C-2"))]
    rows += [
        (650, f"M-1/M-2 50%C {state}", fuel, state, FuelComponent(state, fuel, 50.0))
        for state, fuel in (("leafless", "M-1"), ("green", "M-2"))
    ]
    return pd.DataFrame(
        [
            {"fbp_code": code, "CurveLabel": label, "FuelType": fuel, "SeasonState": state, "ISI": value, "ROS": ros, "HFI": 1.0}
            for code, label, fuel, state, component in rows
            for value, ros in zip(isi, fbp_ros_curve(component, isi), strict=True)
        ]
    )


def test_codes_missing_from_the_model_table_get_curves_from_the_project_definition():
    project = {
        2: ("Boreal Spruce", "C-2"),
        650: ("Boreal Mixedwood (50% Conifer)", "M-1/M-2 (50 PC)"),
        425: ("Boreal Mixedwood - Leafless (25% Conifer)", "M-1 (25 PC)"),
        680: ("Boreal Mixedwood (80% Conifer)", "M-1/M-2 (80 PC)"),
        6: ("Conifer Plantation", "C-6"),
        31: ("Matted Grass", "O-1a"),
        40: ("Boreal Mixedwood - Leafless", "M-1"),
        77: ("Something", None),
    }
    model = _model_curves()

    resolution = resolve_fuel_curves(model, project, codes=[2, 650, 425, 680, 6, 31, 40, 77, 999])

    assert resolution.derived == {425: "M-1 (25 PC)", 680: "M-1/M-2 (80 PC)"}
    assert resolution.replaced == {}
    assert resolution.unresolved == {
        6: "C-6: fuel type C-6 needs curves from the R script",
        31: "O-1a: fuel type O-1a is not in the model's training data",
        40: "M-1: mixedwood needs a percent conifer, e.g. M-1 (25 PC)",
        77: "'Something' has no FBP code in the crosswalk",
        999: "not in the model's fuel table and not listed in hexNN_FuelTypes.csv / hexNN_FuelCodeCrosswalk.csv",
    }
    curves = resolution.curves
    pd.testing.assert_frame_equal(curves[curves.fbp_code.isin([2, 650])].reset_index(drop=True), model)
    m1_80 = curves[(curves.fbp_code == 680) & (curves.SeasonState == "leafless")].sort_values("ISI").ROS
    np.testing.assert_allclose(m1_80, R_CURVES[("M-1", 80.0)], rtol=1e-12)
    assert set(curves[curves.fbp_code == 680].SeasonState) == {"leafless", "green"}


def test_a_known_code_defined_as_another_fuel_uses_the_project_definition():
    project = {2: ("Boreal Mixedwood - Leafless (25% Conifer)", "M-1 (25 PC)"), 650: ("Mature Jack or Lodgepole Pine", "C-3")}

    resolution = resolve_fuel_curves(_model_curves(), project, codes=[2, 650])

    assert resolution.replaced == {2: "M-1 (25 PC)"}
    np.testing.assert_allclose(resolution.curves.query("fbp_code == 2").sort_values("ISI").ROS, R_CURVES[("M-1", 25.0)], rtol=1e-12)
    # C-3 is not in this model's training data: the model's curve is kept, and the conflict is reported.
    assert resolution.unverified == {650: "C-3: fuel type C-3 is not in the model's training data"}
    assert len(resolution.curves.query("fbp_code == 650")) == 2 * len(ISI)


def test_without_project_fuel_tables_only_the_model_table_is_used():
    resolution = resolve_fuel_curves(_model_curves(), None, codes=[2, 425])

    assert not resolution.changed
    assert resolution.unresolved == {
        425: "not in the model's fuel table and no hexNN_FuelTypes.csv / hexNN_FuelCodeCrosswalk.csv to define it"
    }


def test_hfi_curves_are_never_derived():
    resolution = resolve_fuel_curves(_model_curves(), {425: ("x", "M-1 (25 PC)")}, codes=[425], feature_col="HFI")

    assert resolution.unresolved == {425: "not in the model's fuel table; HFI curves must come from the R script"}


def write_project_fuel_tables(project: Path, hex_id: str, codes: dict[int, tuple[str, str | None]]) -> None:
    tabular = project / f"hex{hex_id}" / "tabular"
    tabular.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Name": [name for name, _ in codes.values()], "Description": "", "ID": list(codes), "Color": ""}).to_csv(
        tabular / f"hex{hex_id}_FuelTypes.csv", index=False
    )
    crosswalk = [(name, code) for name, code in codes.values() if code is not None]
    pd.DataFrame(crosswalk, columns=["FuelType", "Code"]).to_csv(tabular / f"hex{hex_id}_FuelCodeCrosswalk.csv", index=False)


def test_project_fuel_codes_join_the_fuel_types_and_crosswalk_tables(tmp_path: Path):
    write_project_fuel_tables(
        tmp_path, "07", {425: ("Boreal Mixedwood - Leafless (25% Conifer)", "M-1 (25 PC)"), 100: ("Not Available", None)}
    )

    codes = read_project_fuel_codes(Paths(hex_id="07", root_dir=tmp_path), "07")

    assert codes == {425: ("Boreal Mixedwood - Leafless (25% Conifer)", "M-1 (25 PC)"), 100: ("Not Available", None)}
    assert read_project_fuel_codes(Paths(hex_id="08", root_dir=tmp_path), "08") is None


@pytest.mark.parametrize("bad_id", ["1.9", "abc", "inf"])
def test_project_fuel_ids_must_be_whole_numbers(tmp_path: Path, bad_id: str):
    write_project_fuel_tables(tmp_path, "07", {425: ("Boreal Mixedwood - Leafless (25% Conifer)", "M-1 (25 PC)")})
    types_path = tmp_path / "hex07" / "tabular" / "hex07_FuelTypes.csv"
    types_path.write_text(types_path.read_text().replace("425", bad_id))

    with pytest.raises(ValueError, match="is not a whole number"):
        read_project_fuel_codes(Paths(hex_id="07", root_dir=tmp_path), "07")
