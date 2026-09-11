import pandas as pd

# features of fire weather list to include
weather_column_names = [
    "WeatherZone",
    "Season",
    "Temperature",
    "RelativeHumidity",
    "Precipitation",
    "FineFuelMoistureCode",
    "DuffMoistureCode",
    "DroughtCode",
    "InitialSpreadIndex",
    "BuildupIndex",
    "FireWeatherIndex",
    "WindSpeed",
    "WindDirection",
]

# Old-style weather column names mapped to their expected canonical names.
weather_column_aliases: dict[str, str] = {
    "FRU": "WeatherZone",
    "temp": "Temperature",
    "rh": "RelativeHumidity",
    "ws": "WindSpeed",
    "wd": "WindDirection",
    "prec": "Precipitation",
    "ffmc": "FineFuelMoistureCode",
    "dmc": "DuffMoistureCode",
    "dc": "DroughtCode",
    "isi": "InitialSpreadIndex",
    "bui": "BuildupIndex",
    "fwi": "FireWeatherIndex",
}


def check_column_format(df: pd.DataFrame, col_name: str) -> bool:
    # Regex Explanation:
    # ^   = Start of string
    # s   = Literal letter 's'
    # \d+ = One or more digits
    # $   = End of string
    pattern = r"^[a-zA-Z]+\d+$"

    # 1. Coerce to string (in case some are ints)
    # 2. Check match
    # 3. .all() ensures EVERY row matches
    is_valid = df[col_name].astype(str).str.match(pattern).all()
    return bool(is_valid)


def check_weather_list(weather_list: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the weather list is of the required format (columns) and the season and WeatherZone column
    """
    # Rename old-style column names to their expected names before validating.
    rename_map = {
        src: dst for src, dst in weather_column_aliases.items() if src in weather_list.columns and dst not in weather_list.columns
    }
    if rename_map:
        weather_list = weather_list.rename(columns=rename_map)

    if not set(list(weather_column_names)).issubset(weather_list.columns):
        raise ValueError("Missing columns/ weather df not in required format")

    if check_column_format(weather_list, "Season"):
        weather_list["Season"] = weather_list["Season"].astype(str).str.extract(r"(\d+)").astype(int)

    if check_column_format(weather_list, "WeatherZone"):
        weather_list["WeatherZone"] = weather_list["WeatherZone"].astype(str).str.extract(r"(\d+)").astype(int)

    return weather_list
