import pandas as pd
import ast
from collections import Counter
from pathlib import Path
import os

# Define Paths (ensure this path is correct for your local machine)
csv_path = Path("C:/Users/dell/Desktop/FINAL-YEAR/PRODUCT_CATEGORIZATION/data/raw/myntra_fashion_products/styles.csv")

# Load the full CSV
try:
    df_full = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
    print(f"CSV loaded from: {csv_path}")
except Exception as e:
    print(f"Error reading CSV from {csv_path}: {e}")
    exit(1)

# Function to parse p_attributes and get categories
def get_category_from_p_attributes(attr_string):
    if pd.isna(attr_string) or attr_string == "NA":
        return "NoAttributesCategory"
    try:
        attr_dict = ast.literal_eval(attr_string)

        if 'Top Type' in attr_dict and attr_dict['Top Type'] != 'NA':
            return attr_dict['Top Type']
        if 'Bottom Type' in attr_dict and attr_dict['Bottom Type'] != 'NA':
            return attr_dict['Bottom Type']
        if 'Dupatta' in attr_dict and attr_dict['Dupatta'] != 'NA':
            return attr_dict['Dupatta']
        if 'Apparel Type' in attr_dict and attr_dict['Apparel Type'] != 'NA':
            return attr_dict['Apparel Type']
        if 'Category' in attr_dict and attr_dict['Category'] != 'NA':
            return attr_dict['Category']

        if 'Occasion' in attr_dict and attr_dict['Occasion'] != 'NA':
            return attr_dict['Occasion'] + " Wear"

        return "OtherApparel"
    except (ValueError, SyntaxError):
        return "ParsingErrorCategory"

# Apply the function to the full DataFrame
if "p_attributes" in df_full.columns:
    df_full["category"] = df_full["p_attributes"].apply(get_category_from_p_attributes)
    print("Parsed 'p_attributes' for categories from the full dataset.")
else:
    df_full["category"] = "GenericCategory"
    print("Neither 'masterCategory' nor 'p_attributes' found in full dataset. Assigning 'GenericCategory'.")

# Calculate and print the full category distribution
full_category_distribution = Counter(df_full["category"])
print("\nFull Dataset Category Distribution (from p_attributes):")
for category, count in sorted(full_category_distribution.items(), key=lambda item: item[1], reverse=True):
    print(f"- {category}: {count} samples")
print(f"\nTotal samples in full dataset with parsed categories: {len(df_full)}")