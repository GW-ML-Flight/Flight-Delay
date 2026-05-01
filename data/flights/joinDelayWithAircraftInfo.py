import polars as pl
from pathlib import Path


def load_data():
    # Read registration data with schema overrides to treat all columns as strings initially
    # This prevents Polars from incorrectly inferring numeric types for mixed columns
    reg = pl.read_csv(
        "data/flights/flightRegistration.csv",
        infer_schema_length=0,  # Read all columns as strings
        ignore_errors=False,
        quote_char=None  # Disable quote parsing since quotes aren't properly escaped
    )
    
    lookup = pl.read_csv(
        "data/flights/N_Number_Lookup.csv",
        infer_schema_length=0,  # Read all columns as strings
        quote_char=None  # Disable quote parsing - fields have embedded quotes that aren't escaped
    )
    
    flights = pl.read_parquet("data/flights/dca_flights.parquet")
    
    return reg, lookup, flights


def clean_registration(reg: pl.DataFrame) -> pl.DataFrame:
    # Select only the columns we need and clean them
    return reg.select([
        "N-NUMBER",
        "SERIAL NUMBER", 
        "MFR MDL CODE",
        "ENG MFR MDL",
        "YEAR MFR"
    ]).with_columns([
        # Clean raw fields
        pl.col("N-NUMBER").str.strip_chars().alias("N_NUMBER_CLEAN"),
        pl.col("SERIAL NUMBER").str.strip_chars(),
        pl.col("MFR MDL CODE").str.strip_chars().alias("MFR_CODE_CLEAN"),
        pl.col("ENG MFR MDL").str.strip_chars(),
        
        # Convert YEAR_MFR to numeric, handling empty strings
        pl.col("YEAR MFR").str.strip_chars().cast(pl.Int32, strict=False),

        # Construct actual tail number 
        (pl.lit("N") + pl.col("N-NUMBER").str.strip_chars()).alias("TAIL_NUM_CLEAN"),
    ])


def clean_lookup(lookup: pl.DataFrame) -> pl.DataFrame:
    return lookup.with_columns([
        pl.col("CODE").str.strip_chars().alias("CODE_CLEAN")
    ])


def clean_flights(flights: pl.DataFrame) -> pl.DataFrame:
    return flights.with_columns([
        pl.col("TAIL_NUM").str.strip_chars()
    ])


def join_all(reg, lookup, flights):
    # Join registration + lookup (aircraft metadata)
    reg_lookup = reg.join(
        lookup,
        left_on="MFR_CODE_CLEAN",
        right_on="CODE_CLEAN",
        how="inner"
    )

    # Join flights + registration
    full = flights.join(
        reg_lookup,
        left_on="TAIL_NUM",
        right_on="TAIL_NUM_CLEAN",
        how="left"
    )

    return full


def print_data(df: pl.DataFrame):
    print("\n=== Data Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns}")
    print(f"\nSchema:\n{df.schema}")
    print(f"\nNull counts:\n{df.null_count()}")
    print(f"\nSample:\n{df.head(10)}")


def main():
    reg, lookup, flights = load_data()

    reg = clean_registration(reg)
    lookup = clean_lookup(lookup)
    flights = clean_flights(flights)

    df = join_all(reg, lookup, flights)

    output_path = Path("data/flights/Full_Flight_Data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    print(f"\nSaved to: {output_path}")
    print_data(df)


if __name__ == "__main__":
    main()