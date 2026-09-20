# Data Directory

This directory contains the UCI Bike Sharing data used by the active training workflow.

`src/data_ingestion.py` downloads the public UCI archive reproducibly, validates `day.csv`, and creates `data/bike_sharing.db` with table `bike_rentals`. Training extracts rows from SQLite with SQL and excludes `casual` and `registered` because those columns leak the target `cnt`.

The older Iris and synthetic artifacts are retained for repository history but are not used by the active Bike Sharing pipeline.
