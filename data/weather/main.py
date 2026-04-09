import math
from pathlib import Path

import polars as pl

# Keys are columns we want; values are individual or list of values that indicate missing data for that column
COLUMNS_TO_MISSING = {
    "DATE": [],
    "REPORT_TYPE": ["9999"],
    "WND": [],  # Wind speed and direction
    "TMP": [],  # Temperature
    "DEW": [],  # Dew point
    "VIS": [],  # Visibility
    "CIG": [],  # Ceiling (cloud height)
    "SLP": [],  # Sea level pressure
    "AA1": [],  # Liquid precipitation
    "AA2": [],  # Liquid precipitation
    "AA3": [],  # Liquid precipitation
    "AH1": [],  # Precipitation type codes
    "AH2": [],  # Precipitation type codes
    "AH3": [],  # Precipitation type codes
    "AH4": [],  # Precipitation type codes
    "AH5": [],  # Precipitation type codes
    "AH6": [],  # Precipitation type codes
}

REPORT_TYPES = [
    "AUTO",  # Report from an automatic station
    "FM-12",  # SYNOP Report of surface observation from a fixed land station
    "FM-15",  # METAR Aviation routine weather report
    "FM-16",  # SPECI Aviation selected special weather report
    "S-S-A",  # Synoptic, airways, and auto merged report
    "SA-AU",  # Airways and auto merged report
    "SY-AU",  # Synoptic and auto merged report
    "SY-MT",  # Synoptic and METAR merged report
]


def load_and_filter_weather_data(csv_path: str) -> pl.DataFrame:
    """
    Load ISD weather data and keep only essential columns.

    Args:
        csv_path: Path to the ISD CSV file

    Returns:
        Polars DataFrame with only the weather columns we need
    """
    # Read the CSV
    df = pl.read_csv(Path(__file__).parent / csv_path)

    # Keep only columns that exist in dataset
    available_cols = [col for col in COLUMNS_TO_MISSING.keys() if col in df.columns]

    # Select only the columns you need
    df_filtered = df.select(available_cols)

    # Select only rows with wanted report types
    if "REPORT_TYPE" in df_filtered.columns:
        df_filtered = df_filtered.filter(pl.col("REPORT_TYPE").is_in(REPORT_TYPES))

    # Treat missing values as nulls
    for col in available_cols:
        missing_values = COLUMNS_TO_MISSING.get(col, [])
        if missing_values:
            df_filtered = df_filtered.with_columns(
                pl.when(pl.col(col).is_in(missing_values))
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            )

    # ====================================
    # WIND cleanup
    ## WND format: "DIR,DIR_Q,TYPE,SPEED,SPEED_Q" (e.g. "270,1,1,10,1")
    df_filtered = df_filtered.with_columns(
        [pl.col("WND").str.split(",").alias("wnd_parts")]
    ).with_columns(
        [
            pl.col("wnd_parts").list.get(0).cast(pl.Int32).alias("wind_dir"),
            pl.col("wnd_parts").list.get(1).alias("wind_dir_q"),  # direction quality
            pl.col("wnd_parts").list.get(2).alias("wind_type"),
            pl.col("wnd_parts").list.get(3).cast(pl.Int32).alias("wind_speed"),
            pl.col("wnd_parts").list.get(4).alias("wind_speed_q"),  # speed quality
        ]
    )

    ## Drop missing values, if other columns are "bad", make all wind columns null for that row
    df_filtered = df_filtered.with_columns(
        [
            (
                (pl.col("wind_dir") != 999)
                & (pl.col("wind_speed") != 9999)
                & (pl.col("wind_dir_q").is_in(["0", "1", "4", "5", "9"]))
                & (pl.col("wind_speed_q").is_in(["0", "1", "4", "5", "9"]))
                & (pl.col("wind_type").is_in(["C", "N"]))  # Keep calm and normal
            ).alias("wind_valid")
        ]
    )

    ## If wind is invalid, set wind columns to null
    df_filtered = df_filtered.with_columns(
        [
            pl.when(pl.col("wind_valid"))
            .then(pl.col("wind_dir"))
            .otherwise(None)
            .alias("wind_dir"),
            pl.when(pl.col("wind_valid"))
            .then(pl.col("wind_speed"))
            .otherwise(None)
            .alias("wind_speed"),
        ]
    )

    ## If wind type is "C" (calm), set speed to 0 and direction to null (since calm has no direction)
    df_filtered = df_filtered.with_columns(
        [
            pl.when(pl.col("wind_type") == "C")
            .then(0)
            .otherwise(pl.col("wind_speed"))
            .alias("wind_speed")
        ]
    )
    df_filtered = df_filtered.with_columns(
        [
            pl.when(pl.col("wind_type") == "C")
            .then(None)
            .otherwise(pl.col("wind_dir"))
            .alias("wind_dir")
        ]
    )

    ## Convert wind speed to m/s and direction to radians for easier modeling later on
    df_filtered = df_filtered.with_columns(
        [
            (pl.col("wind_speed") / 10).alias("wind_speed_mps"),
            (pl.col("wind_dir") * math.pi / 180).alias("wind_rad"),
        ]
    )

    ## Create u and v components of wind vector (u = speed * cos(direction), v = speed * sin(direction))
    df_filtered = df_filtered.with_columns(
        [
            (pl.col("wind_speed_mps") * pl.col("wind_rad").cos()).alias("wind_u"),
            (pl.col("wind_speed_mps") * pl.col("wind_rad").sin()).alias("wind_v"),
        ]
    )

    ## Drop intermediate columns we don't need anymore
    df_filtered = df_filtered.drop(
        [
            "WND",
            "wnd_parts",
            "wind_dir",
            "wind_rad",
            "wind_valid",
            "wind_speed",
            "wind_speed_q",
            "wind_dir_q",
            "wind_type",
        ]
    )

    # ====================================
    # TEMPERATURE cleanup
    ## TMP format: "TEMP,TEMP_Q" (e.g. "150,1" for 15.0°C)

    df_filtered = df_filtered.with_columns(
        [pl.col("TMP").str.split(",").alias("tmp_parts")]
    ).with_columns(
        [
            pl.col("tmp_parts").list.get(0).cast(pl.Int32).alias("temp_c"),
            pl.col("tmp_parts").list.get(1).alias("temp_q"),  # quality
        ]
    )

    ## Set temp to null if quality is bad or value is 9999
    df_filtered = df_filtered.with_columns(
        [
            pl.when(
                (pl.col("temp_c") != 9999)
                & pl.col("temp_q").is_in(["0", "1", "4", "5", "9"])
            )
            .then(pl.col("temp_c") / 10)  # Convert to °C
            .otherwise(None)
            .alias("temp_c")
        ]
    )

    # ====================================
    # DEW cleanup
    ## DEW format: "DEW,DEW_Q" (e.g. "+10,1")
    df_filtered = df_filtered.with_columns(
        [pl.col("DEW").str.split(",").alias("dew_parts")]
    ).with_columns(
        [
            pl.col("dew_parts").list.get(0).cast(pl.Int32).alias("dew_point"),
            pl.col("dew_parts").list.get(1).alias("dew_point_q"),  # dew point quality
        ]
    )

    ## Drop missing values, if other columns are "bad", make all dew columns null for that row
    df_filtered = df_filtered.with_columns(
        [
            (
                (pl.col("dew_point") != 9999)
                & (
                    pl.col("dew_point_q").is_in(
                        ["0", "1", "4", "5", "9", "A", "C", "I", "M", "P", "R", "U"]
                    )
                )  # quality codes that indicate valid data
            ).alias("dew_valid")
        ]
    )

    ## If dew is invalid, set dew columns to null
    df_filtered = df_filtered.with_columns(
        [
            pl.when(pl.col("dew_valid"))
            .then(pl.col("dew_point"))
            .otherwise(None)
            .alias("dew_point"),
        ]
    )

    ## Convert dew point to degrees Celsius (divide by 10)
    df_filtered = df_filtered.with_columns(
        [(pl.col("dew_point") / 10).alias("dew_point_c")]
    )

    ## Drop intermediate columns we don't need anymore
    df_filtered = df_filtered.drop(
        ["DEW", "dew_parts", "dew_point_q", "dew_valid", "dew_point"]
    )
    # ====================================

    # Drop all columns where the whole column is null (e.g. if a column didn't exist in the original dataset, it will be all nulls after selection)
    df_filtered = df_filtered[
        [s.name for s in df_filtered if not (s.null_count() == df_filtered.height)]
    ]

    print(f"Original shape: {df.shape}")
    print(f"Filtered shape: {df_filtered.shape}")
    print(f"Columns kept: {df_filtered.columns}")

    return df_filtered


def process_multiple_files(
    directory: str | Path, output_path: str | None | Path = None
) -> pl.DataFrame:
    """
    Process multiple ISD CSV files and combine them.

    Args:
        directory: Path to directory containing ISD CSV files
        output_path: Optional path to save the combined dataset

    Returns:
        Combined Polars DataFrame
    """
    csv_files = list(Path(directory).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")

    print(f"Found {len(csv_files)} CSV files")

    # Load and filter each file
    dfs = []
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        df = load_and_filter_weather_data(str(csv_file))
        dfs.append(df)

    # Combine all dataframes
    combined_df = pl.concat(dfs)

    print(f"\nCombined dataset shape: {combined_df.shape}")

    # Optionally save to CSV or Parquet
    if output_path:
        if str(output_path).endswith(".parquet"):
            combined_df.write_parquet(output_path)
            print(f"Saved to {output_path} (Parquet)")
        else:
            combined_df.write_csv(output_path)
            print(f"Saved to {output_path} (CSV)")

    return combined_df


def basic_data_inspection(df: pl.DataFrame) -> None:
    """
    Print basic statistics about the weather data.

    Args:
        df: Polars DataFrame with weather data
    """
    print("\n=== Data Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"\nData types:\n{df.schema}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=-1):
        print(f"\nFirst few rows:\n{df.head(20)}")
    print(f"\nNull counts:\n{df.null_count()}")


# Example usage
if __name__ == "__main__":
    # Process multiple CSV files from current directory and save combined dataset
    df = process_multiple_files(
        directory=Path(__file__).parent,
        output_path=Path(__file__).parent / "combined_weather_data.parquet",
    )
    basic_data_inspection(df)
