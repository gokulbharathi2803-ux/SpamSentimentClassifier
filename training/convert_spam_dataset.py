import pandas as pd

input_file = "../data/SMSSpamCollection"
output_file = "../data/spam.csv"

data = pd.read_csv(
    input_file,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

data = data[["text", "label"]]

data.to_csv(output_file, index=False)

print("Dataset converted successfully!")
print("Rows:", len(data))
print("\nColumns:", list(data.columns))
print("\nFirst 5 rows:")
print(data.head())