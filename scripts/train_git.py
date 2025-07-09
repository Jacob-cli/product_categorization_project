import os
import pandas as pd
from PIL import Image
import torch
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoProcessor, \
    GitModel  # Using GitModel for classification tasks, as it provides the base model's embeddings
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)  # Added classification_report for compute_metrics
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
import random
import traceback
import gc  # For garbage collection

# --- Configuration and Paths ---
# Log and Model Save Directories
LOG_DIR = r"C:\Users\dell\Desktop\FINAL-YEAR\product_categorization\logs"
os.makedirs(LOG_DIR, exist_ok=True)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"git_training_log_{current_time}.json")

# Corrected path for Confusion Matrix output to 'results' folder
RESULTS_DIR = r"C:\Users\dell\Desktop\FINAL-YEAR\product_categorization\results"
os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists
CONFUSION_MATRIX_PATH_PREFIX = os.path.join(RESULTS_DIR, f"git_confusion_matrix_{current_time}_epoch")

model_save_directory = "./trained_models_git"
os.makedirs(model_save_directory, exist_ok=True)

# Data paths
DATA_DIR = r"C:\Users\dell\Desktop\FINAL-YEAR\product_categorization\data\raw\myntra_fashion_products"
IMAGE_FOLDER = os.path.join(DATA_DIR, "images")
STYLES_CSV_PATH = os.path.join(DATA_DIR, "styles.csv")

# Training parameters
BATCH_SIZE = 8  # Changed back to 8 as it ran before
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
PATIENCE = 3  # Early stopping patience
MODEL_NAME = "microsoft/git-base"  # Corrected Model Name

# --- Dataset Balancing Configuration ---
TARGET_SAMPLES_PER_CLASS = 600
AUGMENTATION_FACTOR = 2  # This variable is not used in the current balancing logic, but kept for consistency
MIN_SAMPLES_FOR_AUGMENTATION = 50

# --- CPU Optimization ---
# Set number of CPU threads for PyTorch operations
torch.set_num_threads(8)
print(f"PyTorch using {torch.get_num_threads()} CPU threads.")

# --- Data Augmentation Transforms ---
# GIT's processor will handle resizing and normalization based on its pre-training
# These are general transforms for augmentation before processor.
train_augment_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomRotation(10),
    T.ToTensor(),
])

# Global variables for category mapping (will be populated by load_and_preprocess_data)
unique_categories = []
category_to_id = {}
id_to_category = {}
NUM_CLASSES = 0


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data():
    global unique_categories, category_to_id, id_to_category, NUM_CLASSES

    print(f"Loading labels from: {STYLES_CSV_PATH}")
    try:
        styles_df = pd.read_csv(STYLES_CSV_PATH, on_bad_lines='skip')
        styles_df['id'] = styles_df['id'].astype(str)

        # 1. Filter out categories with only one sample
        category_counts_initial = styles_df['masterCategory'].value_counts()
        single_sample_categories = category_counts_initial[category_counts_initial == 1].index.tolist()
        if single_sample_categories:
            styles_df = styles_df[~styles_df['masterCategory'].isin(single_sample_categories)].copy()
            print(
                f"Removed {len(single_sample_categories)} categories with only one sample: {single_sample_categories}")

        # 2. Select specific categories
        selected_category_names = ['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items']
        selected_category_names = [cat for cat in selected_category_names if
                                   cat in styles_df['masterCategory'].unique()]

        if len(selected_category_names) < len(['Apparel', 'Accessories', 'Footwear', 'Personal Care', 'Free Items']):
            print(f"Warning: Not all desired categories found. Using: {selected_category_names}")
            if not selected_category_names:
                raise ValueError("No suitable categories found after filtering.")

        styles_df_filtered_by_selection = styles_df[styles_df['masterCategory'].isin(selected_category_names)].copy()

        print(f"\nSelected {len(selected_category_names)} categories for training:")
        for cat in selected_category_names:
            original_count = styles_df['masterCategory'].value_counts().get(cat, 0)
            print(f"- {cat} (Original Samples: {original_count})")

        # 3. Create a balanced dataset
        balanced_image_data_tuples = []
        augmented_summary = {}

        for category_name in selected_category_names:
            category_df = styles_df_filtered_by_selection[
                styles_df_filtered_by_selection['masterCategory'] == category_name]
            original_sample_ids_in_category = category_df['id'].tolist()

            num_original_available = len(original_sample_ids_in_category)
            augmented_added = 0
            current_category_data = []

            if num_original_available >= TARGET_SAMPLES_PER_CLASS:
                sampled_ids = random.sample(original_sample_ids_in_category, k=TARGET_SAMPLES_PER_CLASS)
                for img_id in sampled_ids:
                    current_category_data.append((img_id, category_name, False))
                total_for_class = TARGET_SAMPLES_PER_CLASS
            elif num_original_available > 0:
                for img_id in original_sample_ids_in_category:
                    current_category_data.append((img_id, category_name, False))

                num_to_augment = TARGET_SAMPLES_PER_CLASS - num_original_available
                augmented_ids_to_add = random.choices(original_sample_ids_in_category, k=num_to_augment)
                for img_id in augmented_ids_to_add:
                    current_category_data.append((img_id, category_name, True))

                augmented_added = num_to_augment
                total_for_class = num_original_available + augmented_added
            else:
                print(f"Warning: Category '{category_name}' has 0 samples. Skipping.")
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

        random.shuffle(balanced_image_data_tuples)

        final_active_categories = sorted(list(set(item[1] for item in balanced_image_data_tuples)))
        unique_categories = final_active_categories
        category_to_id = {category: i for i, category in enumerate(unique_categories)}
        id_to_category = {i: category for category, i in category_to_id.items()}
        NUM_CLASSES = len(unique_categories)

        print(f"\nFinal Number of Classes for Training: {NUM_CLASSES}")
        print(f"Final Categories Mapping: {category_to_id}")
        print(f"Total images after balancing and augmentation: {len(balanced_image_data_tuples)}")

        return balanced_image_data_tuples, augmented_summary

    except FileNotFoundError:
        print(f"Error: styles.csv not found at {STYLES_CSV_PATH}. Please check the path.")
        exit()
    except Exception as e:
        print(f"Error loading or processing styles.csv: {e}")
        traceback.print_exc()
        exit()


# --- Custom Dataset Class ---
class ProductCategorizationDataset(Dataset):
    def __init__(self, data_tuples, image_folder, processor, id_to_category, category_to_id, augment_transforms=None):
        self.data_tuples = data_tuples  # Now takes data_tuples directly
        self.image_folder = image_folder
        self.processor = processor
        self.id_to_category = id_to_category
        self.category_to_id = category_to_id
        self.augment_transforms = augment_transforms

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        img_id, category_name, is_augmented = self.data_tuples[idx]
        label_id = self.category_to_id[category_name]

        image_path = os.path.join(self.image_folder, f"{img_id}.jpg")

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found for ID {img_id}. Skipping.")
            return None  # Return None for missing images

        # Apply augmentation if this sample is marked as augmented AND transforms are provided
        if is_augmented and self.augment_transforms:
            image = self.augment_transforms(image)

        text_prompt = f"a photo of a {self.id_to_category[label_id].replace('_', ' ').lower()}"

        processed_inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
            padding="max_length",  # Ensure consistent padding
            truncation=True,
            max_length=77  # Common max length for text in VLMs
        )

        return {
            'pixel_values': processed_inputs['pixel_values'].squeeze(0),  # Remove batch dim added by processor
            'input_ids': processed_inputs['input_ids'].squeeze(0),
            'attention_mask': processed_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


# --- Custom Collate Function for DataLoader ---
def custom_git_collate_fn(batch):
    # Filter out None samples (from missing images)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# --- GIT Model for Classification ---
class GitClassificationModel(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Load the base GIT model (GitModel provides the core encoder for features)
        # Using AutoModelForCausalLM as the base model for GIT, as it's the underlying architecture
        # that GitModel wraps for its multimodal capabilities.
        self.git_model = GitModel.from_pretrained(model_name)

        # GIT's encoder output hidden size for 'git-base' is typically 768.
        hidden_size = self.git_model.config.hidden_size

        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        print(
            f"GIT model loaded and custom classifier added (input features: {hidden_size}, output classes: {num_labels}).")

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # GIT forward pass to get features from the multimodal encoder
        outputs = self.git_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # Ensure hidden states are returned
        )

        # For classification, we need a single vector representation from the multimodal output.
        # GitModel's outputs typically have a 'last_hidden_state' from the final encoder layer.
        # We'll take the representation corresponding to the first token (often a CLS token equivalent)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return (loss, logits) if loss is not None else (None, logits)  # Return (None, logits) in eval mode


# --- TrainingLogger, compute_metrics, plot_confusion_matrix (Re-used from CLIP) ---
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
        'weighted_recall': recall_weighted,
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

    if len(classes) < 20:  # Only add text if classes are few enough to avoid clutter
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


# --- Main Training Function ---
def train_git_model():
    # --- Load Labels and Create Mapping ---
    # This part remains mostly the same, ensuring global variables are set.
    balanced_image_data_tuples, augmented_summary = load_and_preprocess_data()

    # --- Load Processor and Model ---
    global _global_processor  # Access the global processor for collate_fn
    print(f"Loading GIT processor from: {MODEL_NAME} (will download if not present)...")
    _global_processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print("GIT processor loaded.")

    DEVICE = torch.device("cpu")
    print(f"\nUsing device: {DEVICE}")
    if DEVICE.type == 'cpu':
        print("CUDA is NOT available. Training will be on CPU, which will be slower and potentially unstable.")

    print(f"Loading GIT model from: {MODEL_NAME} (will download if not present)...")
    model = GitClassificationModel(MODEL_NAME, NUM_CLASSES)  # Use the custom class
    model.to(DEVICE)
    print("GIT model adapted for classification and moved to CPU.")

    # --- Data Splitting ---
    # Convert category_name (string) to category_ID (int) for splitting purposes
    data_for_split_with_ids = []
    labels_for_split = []
    for img_id, category_name, is_augmented in balanced_image_data_tuples:
        label_id = category_to_id[category_name]
        data_for_split_with_ids.append((img_id, category_name, is_augmented))  # Keep category_name for dataset
        labels_for_split.append(label_id)  # Use label_id for stratification

    train_data, val_data, _, _ = train_test_split(
        data_for_split_with_ids, labels_for_split, test_size=0.2, stratify=labels_for_split, random_state=42
    )

    print("\nSplitting data into training and validation sets (80-20 split with stratification)...")
    print(f"Total samples for training and validation: {len(train_data) + len(val_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Initialize logger
    logger = TrainingLogger(LOG_FILE)

    # Log configuration
    logger.log_config({
        'model_name': 'GITClassificationModel',
        'base_model_id': MODEL_NAME,
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
        'data_augmentation_transforms': str(train_augment_transforms),
        'git_text_prompt_strategy': 'Using class name in prompt'
    })

    train_dataset = ProductCategorizationDataset(train_data, IMAGE_FOLDER, _global_processor, id_to_category,
                                                 category_to_id, augment_transforms=train_augment_transforms)
    val_dataset = ProductCategorizationDataset(val_data, IMAGE_FOLDER, _global_processor, id_to_category,
                                               category_to_id, augment_transforms=None)

    # Sampler for training data
    train_labels_for_sampler = [category_to_id[item[1]] for item in train_data]  # Get numerical labels for sampler
    class_counts = np.bincount(train_labels_for_sampler)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    train_sample_weights = class_weights[train_labels_for_sampler]
    train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  collate_fn=custom_git_collate_fn, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=custom_git_collate_fn, num_workers=0)

    print(f"Train DataLoader created with {len(train_dataloader)} batches (using WeightedRandomSampler).")
    print(f"Validation DataLoader created with {len(val_dataloader)} batches.")

    # --- Optimizer and Scheduler ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE * 10,  # Max LR can be higher than base LR for OneCycleLR
        steps_per_epoch=len(train_dataloader),
        epochs=NUM_EPOCHS,
        anneal_strategy='cos'  # Cosine annealing is often good
    )

    # --- Training Loop ---
    print("\nStarting GIT training...")
    best_val_macro_f1 = -1.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")):
            if batch is None:
                continue

            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            loss, logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            if loss.dim() > 0:
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            del pixel_values, input_ids, attention_mask, labels, loss, logits, preds
            gc.collect()

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_metrics = compute_metrics(all_train_labels, all_train_preds, unique_categories)

        print(f"\nEpoch {epoch + 1} Training Summary:")
        print(f"  Overall Loss: {avg_train_loss:.4f}")
        print(f"  Overall Accuracy: {train_metrics['overall_accuracy']:.4f}")
        print(f"  Macro F1: {train_metrics['macro_f1']:.4f}")
        print("  Per-Class Training Metrics (F1-score):")
        for category, metrics_data in train_metrics['fairness_metrics'].items():
            print(f"    {category}: F1={metrics_data['f1-score']:.4f} (Support: {metrics_data['support']})")

        # --- Validation ---
        model.eval()
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}")):
                if batch is None:
                    continue

                pixel_values = batch['pixel_values'].to(DEVICE)
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                loss, logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                if loss is not None and loss.dim() > 0:  # Check if loss is not None
                    loss = loss.mean()
                    total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=-1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

                del pixel_values, input_ids, attention_mask, labels, loss, logits, preds
                gc.collect()

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_metrics = compute_metrics(all_val_labels, all_val_preds, unique_categories)

        print(f"\nEpoch {epoch + 1} Validation Summary:")
        print(f"  Overall Loss: {avg_val_loss:.4f}")
        print(f"  Overall Accuracy: {val_metrics['overall_accuracy']:.4f}")
        print(f"  Macro F1: {val_metrics['macro_f1']:.4f}")
        print("  Per-Class Validation Metrics (F1-score):")
        for category, metrics_data in val_metrics['fairness_metrics'].items():
            print(f"    {category}: F1={metrics_data['f1-score']:.4f} (Support: {metrics_data['support']})")

        # Log epoch data to JSON
        # Assuming TrainingLogger is defined in a separate utils.py or similar,
        # or we define it here for self-containment.
        # For now, let's define it within this script for simplicity, as per previous logs.
        logger = TrainingLogger(LOG_FILE)  # Re-instantiate or make global if needed
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

        plot_confusion_matrix(
            np.array(val_metrics['confusion_matrix']),
            unique_categories,
            f"{CONFUSION_MATRIX_PATH_PREFIX}{epoch + 1}.png"
        )
        print(f"Confusion Matrix saved to {CONFUSION_MATRIX_PATH_PREFIX}{epoch + 1}.png")

        # --- Early Stopping Check ---
        current_val_macro_f1 = val_metrics['macro_f1']
        if current_val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = current_val_macro_f1
            patience_counter = 0
            model_filename = f"git_best_model_epoch_{epoch + 1}_val_f1_{best_val_macro_f1:.4f}.pt"
            torch.save(model.state_dict(), os.path.join(model_save_directory, model_filename))
            print(f"--> Saved best model checkpoint to {os.path.join(model_save_directory, model_filename)}")
        else:
            patience_counter += 1
            print(f"Validation Macro F1 did not improve. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in validation Macro F1.")
                break

    print("\n--- GIT training complete! ---")
    print(f"Final Best Validation Macro F1 achieved: {best_val_macro_f1:.4f}")

    # Final log save
    with open(LOG_FILE, 'w') as f:
        json.dump(training_logs, f, indent=4)
    print(f"Training logs saved to {LOG_FILE}")


# --- Run Training ---
if __name__ == "__main__":
    train_git_model()