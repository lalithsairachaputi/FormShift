from datasets import load_dataset
import pandas as pd

# Stream without downloading the whole dataset
dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)

# Take only the first 100 rows
small_dataset = dataset.take(100)

# Convert to pandas and export to CSV/JSON
df = pd.DataFrame(list(small_dataset))
df.to_csv("cnn_dailymail_100.csv", index=False)
# df.to_json("cnn_dailymail_100.json", orient="records", indent=2)