import os
import pandas as pd
from PIL import Image
import torch
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, CLIPForImageClassification
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T  # For data augmentation
import random  # For random sampling for augmentation
import traceback  # For detailed error logging
from tqdm import tqdm  # For progress bars

# --- Configuration and Paths ---
# Log and Model Save Directories
LOG_DIR = r"C:\Users\dell\Desktop\FINAL-YEAR\product_categorization\logs"
os.makedirs(LOG_DIR, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"clip_training_log_{current_time}.json")
CONFUSION_MATRIX_PATH_PREFIX = os.path.join(LOG_DIR, f"clip_confusion_matrix_{current_time}_epoch")
model_save_directory = "./trained_models_clip"  # Specific directory for CLIP models
os.makedirs(model_save_directory, exist_ok=True)

# Data Paths
DATA_DIR = r"C:\Users\dell\Desktop\FINAL-YEAR\product_categorization\data\raw\myntra_fashion_products"
IMAGE_FOLDER = os.path.join(DATA_DIR, "images")
STYLES_CSV_PATH = os.path.join(DATA_DIR, "styles.csv")

# Training parameters
BATCH_SIZE = 8  # Small batch size for CPU and smaller dataset
NUM_EPOCHS = 10  # Increased epochs for smaller, augmented dataset
LEARNING_RATE = 2e-5
DEVICE = torch.device("cpu")  # Sticking to CPU as per your constraint
NUM_WORKERS = 0  # Set to 0 for Windows for stability

# --- Dataset and Augmentation Parameters ---
TARGET_SAMPLES_PER_CLASS = 600  # AIM: 600 samples per selected class (Changed from 200)
NUM_CLASSES_TO_SELECT = 5  # Number of top classes to use for training

# --- Global Variables for Category Mapping (Loaded from Data) ---
image_id_to_category = {}  # Maps image_id (string) to category_name (string)
unique_categories = []  # List of unique category_names (strings)
category_to_id = {}  # Maps category_name (string) to numerical ID (int)
id_to_category = {}  # Maps numerical ID (int) to category_name (string)
NUM_CLASSES = 0  # Total number of unique classes selected

# --- Data Augmentation Transforms ---
# These transforms will be applied to images chosen for augmentation
train_augment_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Resize while maintaining aspect ratio, then crop
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(15),
])


# --- Data Loading and Preprocessing (CRITICAL CHANGES FOR BALANCED SUBSET) ---
def load_and_preprocess_data():
    global image_id_to_category, unique_categories, category_to_id, id_to_category, NUM_CLASSES

    print(f"Loading labels from: {STYLES_CSV_PATH}")
    try:
        styles_df = pd.read_csv(STYLES_CSV_PATH, on_bad_lines='skip')
        styles_df['id'] = styles_df['id'].astype(str)

        # 1. Filter out categories with only one sample (as before)
        category_counts_initial = styles_df['masterCategory'].value_counts()
        single_sample_categories = category_counts_initial[category_counts_initial == 1].index.tolist()
        if single_sample_categories:
            styles_df = styles_df[~styles_df['masterCategory'].isin(single_sample_categories)].copy()
            print(
                f"Removed {len(single_sample_categories)} categories with only one sample: {single_sample_categories}")

        # 2. Select top N categories based on counts (as discussed)
        selected_category_names = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items']

        # Ensure these selected categories actually exist in the filtered data
        selected_category_names = [cat for cat in selected_category_names if
                                   cat in styles_df['masterCategory'].unique()]

        if len(selected_category_names) < NUM_CLASSES_TO_SELECT:
            print(
                f"Warning: Could only select {len(selected_category_names)} categories that exist in the filtered data out of {NUM_CLASSES_TO_SELECT} desired.")
            if not selected_category_names:
                raise ValueError("No suitable categories found after filtering for the specified criteria.")

        styles_df_filtered_by_selection = styles_df[styles_df['masterCategory'].isin(selected_category_names)].copy()

        print(f"\nSelected {len(selected_category_names)} categories for training:")
        for cat in selected_category_names:
            original_count = styles_df['masterCategory'].value_counts().get(cat, 0)
            print(f"- {cat} (Original Samples: {original_count})")

        # 3. Create a balanced dataset (exactly TARGET_SAMPLES_PER_CLASS per category)
        # This list will contain tuples: (image_id, category_name, is_augmented_flag)
        balanced_image_data_tuples = []
        augmented_summary = {}  # To log how many samples were augmented

        for category_name in selected_category_names:
            category_df = styles_df_filtered_by_selection[
                styles_df_filtered_by_selection['masterCategory'] == category_name]
            original_sample_ids_in_category = category_df['id'].tolist()

            num_original_available = len(original_sample_ids_in_category)
            augmented_added = 0
            current_category_data = []

            if num_original_available >= TARGET_SAMPLES_PER_CLASS:
                # If enough original samples, randomly sample TARGET_SAMPLES_PER_CLASS unique ones
                sampled_ids = random.sample(original_sample_ids_in_category, k=TARGET_SAMPLES_PER_CLASS)
                for img_id in sampled_ids:
                    current_category_data.append((img_id, category_name, False))  # False: not augmented
                total_for_class = TARGET_SAMPLES_PER_CLASS
            elif num_original_available > 0:
                # If not enough, take all original and augment the rest
                for img_id in original_sample_ids_in_category:
                    current_category_data.append((img_id, category_name, False))  # False: not augmented

                num_to_augment = TARGET_SAMPLES_PER_CLASS - num_original_available
                # Randomly choose from original samples (with replacement) to augment
                augmented_ids_to_add = random.choices(original_sample_ids_in_category, k=num_to_augment)
                for img_id in augmented_ids_to_add:
                    current_category_data.append((img_id, category_name, True))  # True: augmented

                augmented_added = num_to_augment
                total_for_class = num_original_available + augmented_added
            else:  # If a selected category somehow has 0 samples, it will not contribute
                print(f"Warning: Category '{category_name}' has 0 samples and cannot be included/augmented. Skipping.")
                total_for_class = 0

            balanced_image_data_tuples.extend(current_category_data)

            augmented_summary[category_name] = {
                'original': num_original_available,
                'augmented_added': augmented_added,
                'total_after_aug': total_for_class
            }

        print("\nDataset Balancing Summary (Original -> After Augmentation):")
        for cat, counts in augmented_summary.items():
            print(
                f"- {cat}: {counts['original']} original, {counts['augmented_added']} augmented samples added -> Total: {counts['total_after_aug']}")

        # Shuffle the balanced data to mix categories
        random.shuffle(balanced_image_data_tuples)

        # Update global mappings based on the FINAL set of selected categories
        # (This handles cases where a selected_category_name might have been empty after sampling/filtering)
        final_active_categories = sorted(list(set(item[1] for item in balanced_image_data_tuples)))
        unique_categories = final_active_categories
        category_to_id = {category: i for i, category in enumerate(unique_categories)}
        id_to_category = {i: category for category, i in category_to_id.items()}
        NUM_CLASSES = len(unique_categories)

        print(f"\nFinal Number of Classes for Training: {NUM_CLASSES}")
        print(f"Final Categories Mapping: {category_to_id}")
        print(f"Total images after balancing and augmentation: {len(balanced_image_data_tuples)}")

        return balanced_image_data_tuples, augmented_summary  # Return the processed data for splitting

    except FileNotFoundError:
        print(f"Error: styles.csv not found at {STYLES_CSV_PATH}. Please check the path.")
        exit()
    except Exception as e:
        print(f"Error loading or processing styles.csv: {e}")
        traceback.print_exc()
        exit()


# --- Custom Dataset Class (Adapted for Augmentation and CLIP) ---
class ImageTextDataset(Dataset):
    def __init__(self, data_tuples, processor, image_folder, augment_transforms=None, balance_classes=True):
        # data_tuples: list of (image_id_string, category_ID_int, is_augmented_bool)
        self.image_data = data_tuples
        self.processor = processor
        self.image_folder = image_folder
        self.augment_transforms = augment_transforms  # Apply if a sample is marked as augmented for training
        self.balance_classes = balance_classes

        # Prepare for WeightedRandomSampler if balancing is enabled
        if balance_classes:
            labels_for_sampler = [item[1] for item in self.image_data]  # item[1] is already the numerical label ID

            class_counts = np.bincount(labels_for_sampler, minlength=NUM_CLASSES)  # Ensure minlength for all classes
            class_weights = np.array([1. / c if c > 0 else 0 for c in class_counts])
            self.sampler_weights = torch.DoubleTensor([class_weights[label_id] for label_id in labels_for_sampler])
            self.sampler = WeightedRandomSampler(
                weights=self.sampler_weights,
                num_samples=len(self.image_data),
                replacement=True
            )
        else:
            self.sampler = None

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        # item here is (image_id_string, category_ID_int, is_augmented_bool)
        img_id, label_id, is_augmented = self.image_data[idx]
        image_path = None

        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(self.image_folder, f"{img_id}{ext}")
            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            print(f"Error: Image file not found for ID {img_id} at any common extension. Skipping sample.")
            return None

        try:
            image = Image.open(image_path).convert("RGB")

            # Apply augmentation if this sample is marked as augmented for training
            # Only apply if augment_transforms are provided AND the sample is marked as augmented
            if is_augmented and self.augment_transforms:
                image = self.augment_transforms(image)

            # CLIP processor only takes images, text is not directly used in input for classification
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

            if pixel_values is None or pixel_values.numel() == 0:
                print(
                    f"Error: Processor generated empty/missing pixel_values for ID {img_id}, Path: {image_path}. Skipping sample.")
                return None

            return {
                'pixel_values': pixel_values,
                'labels': torch.tensor(label_id, dtype=torch.long)
            }
        except Exception as e:
            print(
                f"Error processing sample {img_id} (Path: {image_path}, Label: {id_to_category.get(label_id, 'N/A')}): {str(e)}")
            traceback.print_exc()
            return None


# --- Collate Function for DataLoader (Simplified for CLIP) ---
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'labels': labels
    }


# --- TrainingLogger, compute_metrics, plot_confusion_matrix (Re-integrated for academic rigor) ---
class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics = {
            'config': {},
            'training': [],
            'validation': [],
            'test': None,
            'ethical_checks': {
                'class_balance': {},
                'bias_analysis': {},
                'fairness_metrics': {}
            }
        }

    def log_config(self, config):
        self.metrics['config'] = config
        self._save()

    def log_epoch(self, epoch, train_metrics, val_metrics):
        epoch_data = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['training'].append(epoch_data)
        self._save()

    def log_final_metrics(self, test_metrics):
        self.metrics['test'] = test_metrics
        self._save()

    def log_class_balance(self, class_distribution):
        self.metrics['ethical_checks']['class_balance'] = class_distribution
        self._save()

    def _save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def compute_metrics(y_true, y_pred, classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',
                                                                                 zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                                          average='weighted',
                                                                                          zero_division=0)

    per_class_report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fairness = {}
    for i, class_name in enumerate(classes):
        class_mask = (y_true == i)
        if class_mask.sum() > 0:
            class_accuracy = accuracy_score(y_true[class_mask], y_pred[class_mask])
            fairness[class_name] = {
                'accuracy': class_accuracy,
                'precision': per_class_report[class_name]['precision'],
                'recall': per_class_report[class_name]['recall'],
                'f1-score': per_class_report[class_name]['f1-score'],
                'support': int(class_mask.sum())
            }
        else:
            fairness[class_name] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0
            }

    metrics = {
        'overall_accuracy': overall_accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,  # CORRECTED from weighted_recall to recall_weighted
        'weighted_f1': f1_weighted,
        'per_class_report': per_class_report,
        'confusion_matrix': cm.tolist(),
        'fairness_metrics': fairness
    }
    return metrics


def plot_confusion_matrix(cm, classes, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    if len(classes) < 20:
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center", fontsize=8,
                         color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# --- Main Training Loop for CLIP ---
def train_clip_model():
    # Load and preprocess data (now returns the balanced_image_data_tuples and augmented_summary)
    balanced_image_data_tuples, augmented_summary = load_and_preprocess_data()

    # Print device info
    print(f"\nUsing device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        if DEVICE == torch.device("cpu"):
            print("WARNING: GPU is available but DEVICE is set to 'cpu'. Change DEVICE to 'cuda' for faster training.")
    else:
        print(
            "CUDA is NOT available. Training will be on CPU, which can be very slow, but we've optimized dataset size.")

    # --- Load CLIP Processor ---
    # CLIP processor is global for the dataset class
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP processor loaded.")

    # --- Load CLIP Model for Image Classification ---
    print("\nLoading CLIP model for image classification...")
    clip_model = CLIPForImageClassification.from_pretrained(
        "openai/clip-vit-base-patch32",
        num_labels=NUM_CLASSES,
        id2label=id_to_category,
        label2id=category_to_id
    )
    clip_model.to(DEVICE)
    print("CLIP model loaded and moved to CPU.")

    # --- Data Splitting (80-20 validation split on the balanced dataset) ---
    print("\nSplitting data into training and validation sets (80-20 split with stratification)...")
    # balanced_image_data_tuples: list of (image_id_string, category_name_string, is_augmented_bool)
    # For train_test_split, we need numerical labels for stratification

    # Convert category_name (string) to category_ID (int) for splitting purposes
    data_for_split_with_ids = []
    labels_for_split = []
    for img_id, category_name, is_augmented in balanced_image_data_tuples:
        label_id = category_to_id[category_name]
        data_for_split_with_ids.append((img_id, label_id, is_augmented))
        labels_for_split.append(label_id)

    train_data, val_data, _, _ = train_test_split(
        data_for_split_with_ids, labels_for_split, test_size=0.2, random_state=42, stratify=labels_for_split
    )

    print(f"Total samples for training and validation: {len(train_data) + len(val_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Initialize logger
    logger = TrainingLogger(LOG_FILE)

    # Log configuration
    logger.log_config({
        'model_name': 'CLIPForImageClassification',
        'base_model_id': "openai/clip-vit-base-patch32",
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'device': str(DEVICE),
        'class_mapping': category_to_id,
        'dataset_info': {
            'total_original_csv_entries': pd.read_csv(STYLES_CSV_PATH, on_bad_lines='skip').shape[0],
            'selected_categories_original_counts': {cat: augmented_summary[cat]['original'] for cat in
                                                    unique_categories},
            'target_samples_per_class': TARGET_SAMPLES_PER_CLASS,
            'total_samples_after_augmentation': len(train_data) + len(val_data),
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'augmentation_applied': augmented_summary
        },
        'ethical_considerations': {
            'balanced_sampling': True,
            'stratified_split': True,
            'data_augmentation_for_imbalance': True,
            'fairness_metrics_tracked': True
        },
        'data_augmentation_transforms': str(train_augment_transforms)
    })

    # DataLoaders: Train with WeightedRandomSampler for imbalance; Val without.
    train_dataset = ImageTextDataset(train_data, clip_processor, IMAGE_FOLDER,
                                     augment_transforms=train_augment_transforms, balance_classes=True)
    val_dataset = ImageTextDataset(val_data, clip_processor, IMAGE_FOLDER, augment_transforms=None,
                                   balance_classes=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_dataset.sampler,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                num_workers=NUM_WORKERS)

    print(f"Train DataLoader created with {len(train_dataloader)} batches (using WeightedRandomSampler).")
    print(f"Validation DataLoader created with {len(val_dataloader)} batches.")

    # Optimizer and Learning Rate Scheduler
    optimizer = AdamW(clip_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE * 10, total_steps=total_steps, anneal_strategy='cos')

    # Early stopping parameters
    best_val_f1_macro = -1.0  # Using macro F1 for early stopping due to potential imbalance
    patience = 3
    patience_counter = 0

    print("\nStarting CLIP training...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        # --- Training Phase ---
        clip_model.train()
        total_train_loss = 0
        train_preds = []
        train_labels_list = []

        print(f"Training Epoch {epoch + 1}/{NUM_EPOCHS}")
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            if batch is None:
                continue

            optimizer.zero_grad()

            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = clip_model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_metrics = compute_metrics(train_labels_list, train_preds, unique_categories)

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  Overall Loss: {avg_train_loss:.4f}")
        print(f"  Overall Accuracy: {train_metrics['overall_accuracy']:.4f}")
        print(f"  Macro F1: {train_metrics['macro_f1']:.4f}")
        print("  Per-Class Training Metrics (F1-score):")
        for category, metrics_data in train_metrics['fairness_metrics'].items():
            print(f"    {category}: F1={metrics_data['f1-score']:.4f} (Support: {metrics_data['support']})")

        # --- Validation Phase ---
        clip_model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels_list = []

        print(f"Validation Epoch {epoch + 1}/{NUM_EPOCHS}")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}")):
                if batch is None:
                    continue

                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = clip_model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_metrics = compute_metrics(val_labels_list, val_preds, unique_categories)

        print(f"\nEpoch {epoch + 1} Validation Summary:")
        print(f"  Overall Loss: {avg_val_loss:.4f}")
        print(f"  Overall Accuracy: {val_metrics['overall_accuracy']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print("  Per-Class Validation Metrics (F1-score):")
        for category, metrics_data in val_metrics['fairness_metrics'].items():
            print(f"    {category}: F1={metrics_data['f1-score']:.4f} (Support: {metrics_data['support']})")

        # Log epoch data to JSON
        logger.log_epoch(
            epoch=epoch,
            train_metrics={
                'loss': avg_train_loss,
                **train_metrics
            },
            val_metrics={
                'loss': avg_val_loss,
                **val_metrics
            }
        )

        # Plot and save Confusion Matrix for validation set
        plot_confusion_matrix(
            np.array(val_metrics['confusion_matrix']),
            unique_categories,
            f"{CONFUSION_MATRIX_PATH_PREFIX}{epoch + 1}.png"
        )
        print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PATH_PREFIX}{epoch + 1}.png")

        # --- Early Stopping Check ---
        current_val_f1_macro = val_metrics['macro_f1']
        if current_val_f1_macro > best_val_f1_macro:
            best_val_f1_macro = current_val_f1_macro
            patience_counter = 0
            model_filename = f"clip_best_model_epoch_{epoch + 1}_val_f1_{best_val_f1_macro:.4f}.pth"
            torch.save(clip_model.state_dict(), os.path.join(model_save_directory, model_filename))
            print(f"--> Saved best model checkpoint to {os.path.join(model_save_directory, model_filename)}")
        else:
            patience_counter += 1
            print(f"Validation Macro F1 did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in validation Macro F1.")
                break

    print("\n--- CLIP training complete! ---")
    print(f"Final Best Validation Macro F1 achieved: {best_val_f1_macro:.4f}")
    print(f"Training logs saved to {LOG_FILE}")


# --- Run Training ---
if __name__ == "__main__":
    train_clip_model()