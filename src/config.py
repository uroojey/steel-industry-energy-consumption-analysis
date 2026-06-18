# src/config.py

# Simple string path to your dataset file
DATA_PATH = "../data/raw/Steel_industry_data.csv"

# The name of the column we want to predict
TARGET = "Usage_kWh"

# Categorical text columns we want to convert to numbers
CATEGORICAL_COLUMNS = ["WeekStatus", "Day_of_week", "Load_Type"]

# All the feature columns we will use to train our models
FEATURES = [
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'NSM',
    'WeekStatus_enc',
    'Day_of_week_enc',
    'Load_Type_enc',
    'hour_sin',
    'hour_cos',
    'is_weekend'
]
