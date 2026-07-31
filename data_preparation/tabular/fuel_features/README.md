To generate the initial rate of spread data and Head fire intensity for current fuel types in the national map, as a function of ISI (initial spread index),
directly run the R script: source("compute_vector_values_national.R"), the output csv file will be under "fbp_curves_national_fuel.csv"
or BETTER run it from python using the script: `generate_fuel_vectors_national.py`
`python -m data_preparation.tabular.fuel_features.generate_fuel_vectors_national --output-dir /path/to/raw_data/data_samples`

The curves are generated for the fuel inside `Fuel_Types.csv`

[NOT USED FOR NOW] For FBP Raster function, run the python script:
`python -m data_preparation/feature_processing/fuel_features/generate`
This calls the main R script (fbp_features.R) and saves output under `data/fuel_data`.
