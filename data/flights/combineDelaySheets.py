from pathlib import Path

import polars as pl

DCA_AIRPORT_ID = 11278

START_DATE = pl.datetime(2024, 1, 1)
END_DATE = pl.datetime(2025, 8, 27)

# FL_DATE = Date and time for the flight
# MKT_UNIQUE_CARRIER, = The marketing carrier id
# MKT_CARRIER_FL_NUM = The marketing carrier's flight number
# ,OP_UNIQUE_CARRIER = The unique carrier code
# ,OP_CARRIER_AIRLINE_ID = Operating Unique Carrier airline ID
# ,TAIL_NUM = The tail number of the plane (Can be joined with other table for plane information)
# ,OP_CARRIER_FL_NUM = The operating carrier flight number
# ,ORIGIN_AIRPORT_ID = The origin airport ID
# ,ORIGIN_CITY_MARKET_ID = The market (City or land area) that an airport serves
# ,ORIGIN = The origin airport code IATA code
# ,DEST_AIRPORT_ID = The destination airport ID
# ,CRS_DEP_TIME = The scheduled departure time
# ,DEP_TIME = The actual departure time
# ,DEP_DELAY = The delay in minutes
# ,DEP_DELAY_GROUP = A grouped delay in 15 minute blocks
# ,TAXI_OUT = The time the plane taxied out
# ,WHEELS_OFF, = The time a plane actually took off
# WHEELS_ON, = The time a plane actually landed at the airport
# TAXI_IN, = When a flight landed and taxied to it's destination
# CRS_ARR_TIME = The scheduled arrival time
# ,ARR_TIME = The actual arrival time
# ,ARR_DELAY = The arrival delay in minutes
# ,CANCELLED = The flight being canceled
# ,CANCELLATION_CODE = The reason for the flight being canceled
# ,DIVERTED = The flight was diverted
# ,CRS_ELAPSED_TIME, = The scheduled elapsed flight time
# ACTUAL_ELAPSED_TIME = The actual elapsed flight time
# ,AIR_TIME = The amount of time the plane is in the air
# ,FLIGHTS = Number of flights
# ,DISTANCE = The distance between destination and origin airport in miles
# ,CARRIER_DELAY = The minutes of delay due to the carrier
# ,WEATHER_DELAY = The minutes of delay due to weather
# ,NAS_DELAY = The minutes of delay due to national aviation system
# ,SECURITY_DELAY = The minutes of delay due to security
# ,LATE_AIRCRAFT_DELAY = The minutes of delay due to late aircraft arrival
# ,FIRST_DEP_TIME = The minutes of delay due to weather
# ,DIV_AIRPORT_LANDINGS = The number of diverted flights that landed
# ,DIV_REACHED_DEST = The diversion reached it's destination or not
# ,DIV_ACTUAL_ELAPSED_TIME = The time a flight spent diverted


def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col(col).cast(pl.Float64, strict=False)
            if col.startswith("DIV_")
            else pl.col(col)
            for col in df.columns
        ]
    )


def load_and_filter_flight_data(csv_path: str) -> pl.DataFrame:
    """
    Load DoT flight data and filter for DCA + date range.
    Keeps ALL columns except sequence ID
    """

    df = pl.read_csv(csv_path, try_parse_dates=False)

    # Parse FL_DATE properly
    df = df.with_columns(
        pl.col("FL_DATE").str.strptime(
            pl.Datetime, format="%m/%d/%Y %I:%M:%S %p", strict=False
        )
    )

    # Filter for DCA (origin OR destination)
    df = df.filter(
        (pl.col("ORIGIN_AIRPORT_ID") == DCA_AIRPORT_ID)
        | (pl.col("DEST_AIRPORT_ID") == DCA_AIRPORT_ID)
    )

    # Filter date range
    df = df.filter((pl.col("FL_DATE") >= START_DATE) & (pl.col("FL_DATE") <= END_DATE))

    # Clean cancellation codes. Replace empty strings with nulls
    if "CANCELLATION_CODE" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("CANCELLATION_CODE") == "")
            .then(None)
            .otherwise(pl.col("CANCELLATION_CODE"))
            .alias("CANCELLATION_CODE")
        )

    # Remove the sequence columns
    df = df.drop("DEST_AIRPORT_SEQ_ID")
    df = df.drop("ORIGIN_AIRPORT_SEQ_ID")

    print(f"{Path(csv_path).name}: {df.shape}")

    return df


def process_multiple_files(
    directory: str | Path, output_path: str | Path | None = None
) -> pl.DataFrame:
    """
    Combine multiple flight CSVs.
    """

    csv_files = [f for f in Path(directory).glob("*.csv") if "Flight_Delay" in f.name]
    # csv_files = list(Path(directory).glob("*.csv"))

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


def printData(df: pl.DataFrame):
    print("\n=== Data Inspection ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns}")
    print(f"\nSchema:\n{df.schema}")
    print(f"\nNull counts:\n{df.null_count()}")
    print(f"\nSample:\n{df.head(10)}")


if __name__ == "__main__":
    df = process_multiple_files(
        directory=Path(__file__).parent,
        output_path=Path(__file__).parent / "flight_delay.parquet",
    )

    printData(df)
