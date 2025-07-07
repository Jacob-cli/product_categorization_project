import pandas as pd
from PIL import Image
import os
import ast
from collections import Counter
from datasets import Dataset
from transformers import AutoProcessor
import random
from pathlib import Path
import torch

# --- Configuration ---
# Define Paths (ensure this path is correct for your local machine)
# The base directory where your 'data' folder is located
project_root = Path("C:/Users/dell/Desktop/FINAL-YEAR/PRODUCT_CATEGORIZATION")
csv_path = project_root / "data" / "raw" / "myntra_fashion_products" / "styles.csv"
# Image directory is now 'images' (lowercase) as per the new dataset
image_dir = project_root / "data" / "raw" / "myntra_fashion_products" / "images"
processed_data_path = project_root / "data" / "processed" / "processed_balanced_dataset"

# --- Parameters for Class Selection and Balancing (Optimized for Academic Rigor & System Performance) ---
# Minimum samples a class must have in the original dataset to be considered
MIN_SAMPLES_PER_CLASS_FOR_SELECTION = 200
# Target number of samples to take from EACH selected top class
TARGET_SAMPLES_PER_STABLE_CLASS = 200
# Number of most frequent stable classes to include in the final balanced dataset
NUM_TOP_CLASSES_TO_SELECT = 5  # Aiming for 5 classes * 200 samples/class = ~1000 total samples


# --- Function to extract category from new dataset columns ---
def get_category_from_columns(row):
    # Prioritize specific article types for more granular classification and diversity
    if pd.notna(row['articleType']):
        return str(row['articleType']).replace(" ", "")  # Clean up spaces

    # Fallback to subCategory if articleType is not available or suitable
    if pd.notna(row['subCategory']):
        return str(row['subCategory']).replace(" ", "")

    # Fallback to masterCategory as a broad category if neither above is suitable
    if pd.notna(row['masterCategory']):
        return str(row['masterCategory']).replace(" ", "")

    # Fallback if no specific category is found across prioritized columns
    return "Other"


# --- Function to load and preprocess images and text ---
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


def preprocess_data_batch(examples):
    images = []
    texts = []

    # Filter out samples where image path is invalid or image cannot be opened
    valid_indices = []
    for i, product_id in enumerate(examples["id"]):
        img_filename = f"{product_id}.jpg"
        img_path = os.path.join(image_dir, img_filename)

        try:
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            texts.append(examples["productDisplayName"][i] if pd.notna(examples["productDisplayName"][i]) else "")
            valid_indices.append(i)
        except FileNotFoundError:
            pass
        except Exception as e:
            pass

    if not images:
        return {
            "input_ids": [],
            "pixel_values": [],
            "attention_mask": [],
            "category": [],
            "id": [],
            "productDisplayName": []
        }

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)

    return {
        "input_ids": inputs.input_ids.tolist(),
        "pixel_values": inputs.pixel_values.tolist(),
        "attention_mask": inputs.attention_mask.tolist(),
        "category": [examples["category"][i] for i in valid_indices],
        "id": [examples["id"][i] for i in valid_indices],
        "productDisplayName": [examples["productDisplayName"][i] for i in valid_indices],
        "masterCategory": [examples["masterCategory"][i] for i in valid_indices],
        "subCategory": [examples["subCategory"][i] for i in valid_indices],
        "articleType": [examples["articleType"][i] for i in valid_indices],
    }


# --- Main Preprocessing Script ---
if __name__ == "__main__":
    print(f"CSV loaded from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False)
        print(f"Columns in the loaded DataFrame: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        exit(1)

    df["category"] = df.apply(get_category_from_columns, axis=1)
    print("Extracted categories from dataset columns.")

    df['id_numeric'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id_numeric']).copy()
    df['id'] = df['id_numeric'].astype(int)
    df = df.drop(columns=['id_numeric'])

    print(f"Attempting to find images in: {image_dir}")
    pre_image_filter_len = len(df)
    df['img_path_exists'] = df['id'].apply(lambda x: os.path.exists(os.path.join(image_dir, f"{x}.jpg")))
    df = df[df['img_path_exists']].copy()
    print(f"Filtered to {len(df)} samples with valid image paths. (Before image path filter: {pre_image_filter_len})")

    if len(df) == 0:
        print("No samples with valid image paths found after filtering. Exiting.")
        exit(1)

    sample_id = df['id'].iloc[0]
    sample_img_path = os.path.join(image_dir, f"{sample_id}.jpg")
    print(f"Checking existence of sample image: {sample_img_path} -> {os.path.exists(sample_img_path)}")

    # 3. Identify "stable" categories and select the top N most frequent ones
    full_category_counts = Counter(df["category"])

    # Filter categories that meet the minimum sample requirement
    candidate_stable_categories = {
        cat: count for cat, count in full_category_counts.items()
        if count >= MIN_SAMPLES_PER_CLASS_FOR_SELECTION
    }

    if not candidate_stable_categories:
        print(
            f"No categories found with at least {MIN_SAMPLES_PER_CLASS_FOR_SELECTION} samples. Adjust MIN_SAMPLES_PER_CLASS_FOR_SELECTION if needed.")
        exit(1)

    # Sort candidates by count (most frequent first) and select the top N
    sorted_stable_categories = sorted(candidate_stable_categories.items(), key=lambda item: item[1], reverse=True)
    selected_top_categories = [cat for cat, _ in sorted_stable_categories[:NUM_TOP_CLASSES_TO_SELECT]]

    print(
        f"\nIdentified {len(candidate_stable_categories)} candidate stable categories (>= {MIN_SAMPLES_PER_CLASS_FOR_SELECTION} samples).")
    print(f"Selecting top {NUM_TOP_CLASSES_TO_SELECT} categories for balancing:")
    for cat in selected_top_categories:
        print(f"- {cat} (Original count: {full_category_counts[cat]} samples)")

    # 4. Create a new DataFrame by sampling TARGET_SAMPLES_PER_STABLE_CLASS from each selected top category
    balanced_samples_list = []
    total_target_samples = 0
    for category in selected_top_categories:
        category_df = df[df["category"] == category]
        num_samples_to_take = min(len(category_df), TARGET_SAMPLES_PER_STABLE_CLASS)
        sampled_df = category_df.sample(n=num_samples_to_take, random_state=42)
        balanced_samples_list.append(sampled_df)
        total_target_samples += num_samples_to_take

    if not balanced_samples_list:
        print("No samples selected for balancing. Exiting.")
        exit(1)

    df_balanced = pd.concat(balanced_samples_list).reset_index(drop=True)

    print(
        f"\nSuccessfully sampled {len(df_balanced)} images to create a balanced dataset from {len(selected_top_categories)} top categories.")
    print(
        f"Targeting {TARGET_SAMPLES_PER_STABLE_CLASS} samples per selected class. Total expected: {len(selected_top_categories) * TARGET_SAMPLES_PER_STABLE_CLASS}")

    final_preprocessed_distribution = Counter(df_balanced["category"])
    print("\nFinal Preprocessed Dataset Category Distribution:")
    for category, count in sorted(final_preprocessed_distribution.items(), key=lambda item: item[1], reverse=True):
        print(f"- {category}: {count} samples")
    print(f"Total samples in final preprocessed dataset: {len(df_balanced)}")

    dataset_columns = ['id', 'productDisplayName', 'category', 'masterCategory', 'subCategory', 'articleType', 'gender',
                       'baseColour', 'season', 'year', 'usage']
    dataset = Dataset.from_pandas(df_balanced[dataset_columns])

    print(f"Using {os.cpu_count()} CPU cores for preprocessing...")
    num_cpu_cores = os.cpu_count() or 1
    processed_dataset = dataset.map(
        preprocess_data_batch,
        batched=True,
        num_proc=num_cpu_cores,
        remove_columns=['productDisplayName', 'masterCategory', 'subCategory', 'articleType', 'gender', 'baseColour',
                        'season', 'year', 'usage']
    )

    original_len_post_map = len(processed_dataset)
    processed_dataset = processed_dataset.filter(
        lambda example: example['input_ids'] is not None and example['pixel_values'] is not None)
    filtered_len_post_map = len(processed_dataset)
    if original_len_post_map > filtered_len_post_map:
        print(
            f"Filtered out {original_len_post_map - filtered_len_post_map} samples that failed image or text processing after map.")

    processed_dataset.set_format("torch")

    processed_dataset.save_to_disk(processed_data_path)
    print(f"\nPreprocessed dataset saved to: {processed_data_path}")

    if len(processed_dataset) > 0:
        sample = processed_dataset[0]
        print("\nSample data from preprocessed dataset:")
        print(f"Product ID: {sample['id']}")
        print(f"Input IDs shape (text tokens): {sample['input_ids'].shape}")
        print(f"Image shape (pixel values): {sample['pixel_values'].shape}")
        print(f"Category (Parsed from new columns): {sample['category']}")
    else:
        print("\nNo samples in the final preprocessed dataset.")