import pandas as pd
DATA_FILE_PATH = 'data/cleaned_data_no_zero_periods_filtered.csv'

data_full = pd.read_csv(DATA_FILE_PATH)
all_place_ids = data_full['placeId'].unique()[:30]

print(all_place_ids)