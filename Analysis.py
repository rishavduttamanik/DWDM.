import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('eng_-french.csv')

# Size before cleaning
print("Original size:", len(df))

# Drop rows with missing values
df.dropna(inplace=True)

# Size after cleaning
print("After cleaning:", len(df))

# Split into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Training entries:", len(train_df))
print("Testing entries:", len(test_df))
# Save the splits
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
