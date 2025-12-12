import os
import pandas as pd 
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import data_process

# data_process.build_agri_causal_dataset(crop_csv="D:\data\causalgraph2025\data\datasets\crop_yield.csv",
# soil_csv="D:\data\causalgraph2025\data\datasets\state_soil_data.csv",
# weather_csv="data/datasets/state_weather_data_1997_2020.csv",
# output_csv="data/datasets/process.csv")
df = pd.read_csv(r"D:\data\causalgraph2025\data\datasets\process.csv")
df_conti_cols = data_process.get_continuous_columns(df)

df[df_conti_cols].to_csv("D:\data\causalgraph2025\data\datasets\process_conti.csv",index=False)