import pandas as pd
import os

STYLES_CSV_PATH = r"C:\Users\dell\Desktop\FINAL-YEAR\product_categorization\data\raw\myntra_fashion_products\styles.csv"

try:
    styles_df = pd.read_csv(STYLES_CSV_PATH, on_bad_lines='skip')

    # Filter out categories with only one sample first, as before
    category_counts = styles_df['masterCategory'].value_counts()
    single_sample_categories = category_counts[category_counts == 1].index.tolist()
    if single_sample_categories:
        styles_df = styles_df[~styles_df['masterCategory'].isin(single_sample_categories)].copy()

    # Get the value counts of masterCategory after filtering
    category_counts_filtered = styles_df['masterCategory'].value_counts()
    print("Filtered Category Counts:")
    print(category_counts_filtered.to_string())

except FileNotFoundError:
    print(f"Error: styles.csv not found at {STYLES_CSV_PATH}")
except Exception as e:
    print(f"Error reading or processing styles.csv: {e}")