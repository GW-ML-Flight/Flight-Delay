1. Add historical/derived features
2. Fix train/test split to be time-based, not random
3. Hyperparameter tuning (RandomizedSearchCV)
4. Report
5. Maybe UI for input


Some slop notes:

Good feature buckets

- [x] Time of day features from FL_DATE and CRS_DEP_TIME: hour, day_of_week, month, weekend, departure_time_bucket, and optionally sine/cosine encodings for time of day. Use the scheduled departure time, not actual departure.
- [x] Airport congestion features from the flight table: count of departures from the same origin airport in the last 1h, 3h, 6h, and 24h; same for arrivals into the destination airport if you have them. Derive these by sorting flights by FL_DATE and counting prior rows for the same airport.
- [X] Carrier history features from the flight table: rolling mean delay, rolling cancellation rate, and rolling late-flight rate for MKT_UNIQUE_CARRIER over 7d and 30d. Use only flights that happened before the current row.
- [x] Origin airport history features from the flight table: rolling mean delay, median delay, and cancellation rate for ORIGIN_AIRPORT_ID over 7d and 30d.
- [X] Destination airport history features from the flight table: same idea for DEST_AIRPORT_ID. This often captures arrival-bank congestion and destination-specific patterns.
- [x] Route features from the flight table: combine ORIGIN_AIRPORT_ID + DEST_AIRPORT_ID and compute rolling delay stats for that route.
- [ ] Tail-number features from the flight table: prior delay of the same TAIL_NUM, plus time since the aircraft’s previous flight and previous arrival delay if available. This is a strong proxy for aircraft knock-on delay.
- [ ] Delay-cause history from the flight table: rolling shares or counts of past CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, and LATE_AIRCRAFT_DELAY. These should be aggregated over prior flights only, not used directly on the current row.
- [ ] Weather threshold flags from the weather parquet joined in model-training.ipynb: low_visibility, heavy_wind, precipitation_present, freezing_temp, low_ceiling. These are simple boolean versions of your current weather columns.
- [ ] Weather trend features from isd_weather_data.parquet: change in temp, pressure, windspeed, and visibility_m over the last 1h or 3h. If the weather is getting worse quickly, that matters more than the absolute value.
- [x] Distance transforms from DISTANCE: log(distance), distance buckets, and maybe an interaction like distance * weather_severity.
- [ ] Carrier-route interactions from the flight table: MKT_UNIQUE_CARRIER + route, since some carriers handle specific routes much better or worse.
- [ ] Aircraft model features from MODEL: model frequency, model-specific average delay, and model-specific cancellation rate. This is mostly a historical lookup table.

Easy work split

- Person 1: time features and distance transforms.
- Person 2: carrier, origin airport, and route rolling stats.
- Person 3: tail-number and aircraft-model history.
- Person 4: weather flags and weather trend features.
- Person 5: historical delay-cause mix features.

Two implementation rules matter:

- Compute every rolling feature using only rows strictly before the current flight, or you will leak the answer.
- Build the features first, then split train/test by time if possible, because a random split can make historical features look better than they really are.
