import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define BLIPClassificationModel at the top level
class BLIPClassificationModel(torch.nn.Module):
    def __init__(self, blip_model, num_labels, id2label, label2id):
        super().__init__()
        self.blip_model = blip_model
        self.classifier = torch.nn.Linear(self.blip_model.config.text_config.hidden_size, num_labels)
        self.num_labels = num_labels
        self.config = blip_model.config
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        vision_output = self.blip_model.vision_model(pixel_values=pixel_values)[0]
        pooled_output = vision_output[:, 0, :]  # CLS token output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return type('Outputs', (object,), {'loss': loss, 'logits': logits})()


class ProductImageDataset(Dataset):
    def __init__(self, dataset_info, processor, image_dir, split='train'):
        self.image_dir = image_dir
        self.split = split
        self.processor = processor
        self.data = []
        self.category_to_id = dataset_info['category_mapping']
        self.id_to_category = {v: k for k, v in self.category_to_id.items()}

        split_data = dataset_info.get(split, [])
        for item in split_data:
            image_path = os.path.join(image_dir, item['image_path'])
            if os.path.exists(image_path):
                self.data.append({
                    'image_path': image_path,
                    'category': item['category'],
                    'label': self.category_to_id[item['category']]
                })
            else:
                logger.warning(f"Image not found: {image_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        label = item['label']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a placeholder or skip
            image = Image.new('RGB', (224, 224), color='gray')
            label = -1  # Indicate invalid sample

        # Process image and text
        text = "a photo of"  # Simple prompt
        inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True,
                                max_length=77)

        # Flatten the inputs to remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0) if 'attention_mask' in inputs else None

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_dataset_info(dataset_info_path):
    if not os.path.exists(dataset_info_path):
        logger.error(f"Dataset info file not found at: {dataset_info_path}")
        raise FileNotFoundError(f"Dataset info file not found at: {dataset_info_path}")
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info


def train_blip_model(args):
    dataset_info_path = args.dataset_info_path
    image_dir = args.image_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    max_grad_norm = args.max_grad_norm

    # Create output directories
    model_dir = os.path.join(output_dir, 'trained_models', 'blip')
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load dataset info
    dataset_info = load_dataset_info(dataset_info_path)
    category_to_id = dataset_info['category_mapping']
    id_to_category = {v: k for k, v in category_to_id.items()}
    NUM_CLASSES = len(category_to_id)

    # Initialize processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BLIPClassificationModel(blip_base_model, NUM_CLASSES, id_to_category, category_to_id)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_model.to(device)
    logger.info(f"Using device: {device}")

    # Initialize datasets
    train_dataset = ProductImageDataset(dataset_info, processor, image_dir, split='train')
    val_dataset = ProductImageDataset(dataset_info, processor, image_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(blip_model.parameters(), lr=learning_rate)

    # Training loop
    best_val_f1 = 0.0
    best_model_path = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"blip_training_log_{timestamp}.json")
    training_logs = {
        "config": {
            "model": "Salesforce/blip-image-captioning-base",
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "category_mapping": category_to_id
        },
        "epochs": []
    }

    for epoch in range(num_epochs):
        blip_model.train()
        total_train_loss = 0.0
        train_predictions = []
        train_labels = []

        for batch_idx, batch in enumerate(train_loader):
            if batch['labels'].item() == -1:  # Skip invalid samples
                continue

            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = blip_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            if loss is not None:
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(blip_model.parameters(), max_grad_norm)
                optimizer.step()

            # Collect predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            train_predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_f1 = f1_score(train_labels, train_predictions, average='weighted')

        # Validation
        blip_model.eval()
        total_val_loss = 0.0
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if batch['labels'].item() == -1:
                    continue

                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device) if batch['attention_mask'] is not None else None
                labels = batch['labels'].to(device)

                outputs = blip_model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                if loss is not None:
                    total_val_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                val_predictions.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_f1 = f1_score(val_labels, val_predictions, average='weighted')

        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")

        # Save epoch metrics
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1
        }
        training_logs["epochs"].append(epoch_log)

        # Save model if validation F1 improves
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_filename = f"blip_best_model_epoch_{epoch + 1}_val_f1_{val_f1:.4f}.pth"
            best_model_path = os.path.join(model_dir, model_filename)
            torch.save(blip_model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path}")

        # Save training log after each epoch
        with open(log_file, 'w') as f:
            json.dump(training_logs, f, indent=4)

    # Save final best model path in log
    training_logs["final_best_model_path"] = best_model_path
    with open(log_file, 'w') as f:
        json.dump(training_logs, f, indent=4)
    logger.info(f"Training completed. Best model saved at: {best_model_path}")
    logger.info(f"Best Validation F1: {best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train BLIP model for product categorization")
    parser.add_argument('--dataset_info_path', type=str, default='data/processed/dataset_info.json',
                        help='Path to dataset info JSON file')
    parser.add_argument('--image_dir', type=str, default='data/images',
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save models and logs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    args = parser.parse_args()

    train_blip_model(args)


if __name__ == "__main__":
    main()