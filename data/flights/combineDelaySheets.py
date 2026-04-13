import polars as pl
from pathlib import Path

DCA_AIRPORT_ID = 11278

START_DATE = pl.datetime(2024, 1, 1)
END_DATE = pl.datetime(2025, 8, 27)

def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col(col).cast(pl.Float64, strict=False)
        if col.startswith("DIV_") else pl.col(col)
        for col in df.columns
    ])


def load_and_filter_flight_data(csv_path: str) -> pl.DataFrame:
    """
    Load DoT flight data and filter for DCA + date range.
    Keeps ALL columns.
    """

    df = pl.read_csv(csv_path, try_parse_dates=False)

    # -----------------------------
    # Parse FL_DATE properly
    df = df.with_columns(
        pl.col("FL_DATE").str.strptime(
            pl.Datetime,
            format="%m/%d/%Y %I:%M:%S %p",
            strict=False
        )
    )

    # -----------------------------
    # Filter for DCA (origin OR destination)
    df = df.filter(
        (pl.col("ORIGIN_AIRPORT_ID") == DCA_AIRPORT_ID) |
        (pl.col("DEST_AIRPORT_ID") == DCA_AIRPORT_ID)
    )

    # -----------------------------
    # Filter date range
    df = df.filter(
        (pl.col("FL_DATE") >= START_DATE) &
        (pl.col("FL_DATE") <= END_DATE)
    )

    # -----------------------------
    # Optional (but useful): clean cancellation codes
    # Replace empty strings with nulls
    if "CANCELLATION_CODE" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("CANCELLATION_CODE") == "")
            .then(None)
            .otherwise(pl.col("CANCELLATION_CODE"))
            .alias("CANCELLATION_CODE")
        )

    print(f"{Path(csv_path).name}: {df.shape}")

    return df


def process_multiple_files(directory: str | Path, output_path: str | Path | None = None) -> pl.DataFrame:
    """
    Combine multiple flight CSVs.
    """


    csv_files = [
        f for f in Path(directory).glob("*.csv")
        if "Flight_Delay" in f.name
    ]
    #csv_files = list(Path(directory).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found.")

    dfs = []

    for file in csv_files:
        print(f"Processing {file.name}...")
        df = load_and_filter_flight_data(file)
        dfs.append(df)
        
    for df in dfs:
        df = normalize_schema(df)

    combined = pl.concat(dfs, how="vertical_relaxed")

    print(f"\nFinal shape: {combined.shape}")

    if output_path:
        combined.write_parquet(output_path)
        print(f"Saved to {output_path}")

    return combined


def basic_data_inspection(df: pl.DataFrame):
    print("\n=== Data Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns}")
    print(f"\nSchema:\n{df.schema}")
    print(f"\nNull counts:\n{df.null_count()}")
    print(f"\nSample:\n{df.head(10)}")


if __name__ == "__main__":
    df = process_multiple_files(
        directory=Path(__file__).parent,
        output_path=Path(__file__).parent / "dca_flights.parquet"
    )

    basic_data_inspection(df)