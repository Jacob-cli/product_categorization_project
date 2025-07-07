import os
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd  # Used for temporary conversion if needed for stratified split


def main():
    project_dir = Path(os.getcwd())
    processed_data_path = project_dir / "data" / "processed" / "processed_subset_dataset"

    print(f"Loading preprocessed dataset from: {processed_data_path}")
    try:
        # Load the dataset. We'll set the format to pandas temporarily for stratified split
        # as sklearn's train_test_split works well with pandas DataFrames or numpy arrays.
        dataset = load_from_disk(processed_data_path)
        print(f"Dataset loaded with {len(dataset)} samples.")

        # Ensure the dataset is in a format suitable for pandas conversion if it's not already
        dataset.set_format("pandas")
        df = dataset.to_pandas()

        # Define categories for stratification
        categories = df['category'].tolist()

        # --- Handle extremely rare classes for stratified split ---
        # Stratified split can fail if a class has only 1 sample and you try to split it
        # into multiple sets. We'll identify such classes and handle them.

        # Count occurrences of each category
        category_counts = df['category'].value_counts()

        # Identify categories with only 1 sample
        single_sample_categories = category_counts[category_counts == 1].index.tolist()

        if single_sample_categories:
            print(
                f"Warning: The following categories have only 1 sample and cannot be perfectly stratified across all splits: {single_sample_categories}")
            print(
                "These samples will be assigned to a single split (likely training or validation/test based on random state).")

        # First, split into train and temp (validation + test)
        # Use a consistent random_state for reproducibility
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,  # 15% for val, 15% for test = 30% for temp
            stratify=categories,
            random_state=42
        )

        # Now, split temp into validation and test
        # Calculate new stratify targets for the temp_df
        temp_categories = temp_df['category'].tolist()
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,  # 15% of total / 30% of total_temp = 0.50
            stratify=temp_categories,
            random_state=42
        )

        # Convert pandas DataFrames back to Hugging Face Dataset objects
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Set format to torch for subsequent use
        train_dataset.set_format("torch")
        val_dataset.set_format("torch")
        test_dataset.set_format("torch")

        # Create a DatasetDict
        dataset_splits = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })

        # Save the splits
        output_split_path = project_dir / "data" / "processed" / "split_dataset"
        dataset_splits.save_to_disk(output_split_path)

        print("\nDataset split completed successfully:")
        print(f"  Training samples: {len(dataset_splits['train'])}")
        print(f"  Validation samples: {len(dataset_splits['validation'])}")
        print(f"  Test samples: {len(dataset_splits['test'])}")
        print(f"Splits saved to: {output_split_path}")

        # Optional: Verify category distribution in splits (can be time-consuming for large datasets)
        print("\nVerifying category distribution in splits (first 5 categories):")
        for split_name, ds in dataset_splits.items():
            ds_df = ds.to_pandas()
            print(f"  {split_name} distribution:")
            print(ds_df['category'].value_counts(normalize=True).head())  # Show top 5 proportions
            print("-" * 20)

    except Exception as e:
        print(f"Error during dataset splitting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()