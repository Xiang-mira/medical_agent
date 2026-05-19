# Author: Qilong Wu
# Institute: JHU CCVL, NUS
# Description: It is used to filter the duplicated rows in the csv file.

# Use case: python filter_csv.py -t /path/to/your/csv/file.csv
#############################################################################
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.t)
df_unique = df.drop_duplicates(subset=["sample", "organ"], keep="first")

original_row_count = len(df)
unique_row_count = len(df_unique)

print(f"Original Lines: {original_row_count}")
print(f"After Removing Duplicates: {unique_row_count}")

df_unique.to_csv(args.t, index=False)