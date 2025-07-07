import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import CLIPProcessor, CLIPModel, SwinModel
from timm import create_model # This import is present in your script but not used for SwinModel from transformers
from tqdm import tqdm
import torch.nn as nn

# Argument parser
parser = argparse.ArgumentParser(description="Test CLIP, SWIN, or BLIP model")
parser.add_argument("--model", type=str, required=True, choices=["clip", "swin", "blip"], help="Model to test")
parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
parser.add_argument("--data_dir", type=str, required=True, help="Path to test dataset")
parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
parser.add_argument("--clip_prompt", type=str, default="category", choices=["category", "product", "description"],
                    help="CLIP prompt style")
args = parser.parse_args()

# Define parameters
device = torch.device(args.device)
label_mapping = {
    "Accessories": 0,
    "Apparel": 1,
    "Footwear": 2,
    "Free Items": 3,
    "Personal Care": 4
}
categories = list(label_mapping.keys())
num_classes = len(categories)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Corrected load_test_data function
def load_test_data(data_dir_arg):
    # Resolve the provided data_dir_arg to an absolute path.
    # This assumes data_dir_arg (e.g., "data/myntradataset/train_data")
    # is the directory that *directly contains* your category folders (Accessories, Apparel, etc.).
    base_data_path = os.path.abspath(data_dir_arg)

    images = []
    labels = []

    print(f"Attempting to load data from: {base_data_path}")

    for category in categories:
        category_dir = os.path.join(base_data_path, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found: {category_dir}. Skipping...")
            continue
        img_count = 0
        for img_name in os.listdir(category_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): # Added more image extensions
                img_path = os.path.join(category_dir, img_name)
                images.append(img_path)
                labels.append(label_mapping[category])
                img_count += 1
        print(f"Loaded {img_count} images for category {category} from {category_dir}")
    if not images:
        raise FileNotFoundError(f"No valid images found in '{base_data_path}' or its category subdirectories. Please check your --data_dir argument and folder structure.")
    return images, labels


# Custom SwinClassifier
class SwinClassifier(nn.Module):
    def __init__(self, num_labels):
        super(SwinClassifier, self).__init__()
        self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.classifier = nn.Linear(self.swin.config.hidden_size, num_labels)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


# Load model
def load_model(model_name, weights_path):
    if model_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        state_dict = torch.load(weights_path, map_location=device)
        # Handle missing position_ids if they were not saved with the state_dict
        if "text_model.embeddings.position_ids" not in state_dict:
            state_dict["text_model.embeddings.position_ids"] = model.text_model.embeddings.position_ids
        if "vision_model.embeddings.position_ids" not in state_dict:
            state_dict["vision_model.embeddings.position_ids"] = model.vision_model.embeddings.position_ids
        # Handle potential 'module.' prefix from DataParallel saving
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Successfully loaded CLIP model weights from {weights_path} with strict=True.")
        except RuntimeError as e:
            print(f"Strict loading failed for CLIP: {e}. Attempting non-strict loading...")
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded CLIP model weights from {weights_path} with strict=False.")
        return model, processor
    elif model_name == "swin":
        model = SwinClassifier(num_labels=num_classes)
        state_dict = torch.load(weights_path, map_location=device)
        # Handle potential 'module.' prefix from DataParallel saving
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Successfully loaded SWIN model weights from {weights_path} with strict=True.")
        except RuntimeError as e:
            print(f"Strict loading failed for SWIN: {e}. Attempting non-strict loading...")
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded SWIN model weights from {weights_path} with strict=False.")
        return model, None
    elif model_name == "blip":
        from transformers import BlipProcessor, BlipForConditionalGeneration # Corrected import for base BlipForConditionalGeneration
        # Load the base BLIP model first, then wrap it with the classifier head
        base_blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForImageClassification(base_blip_model, num_labels=num_classes) # Use your custom classifier wrapper
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        state_dict = torch.load(weights_path, map_location=device)
        # Handle potential 'module.' prefix from DataParallel saving
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"Successfully loaded BLIP model weights from {weights_path} with strict=True.")
        except RuntimeError as e:
            print(f"Strict loading failed for BLIP: {e}. Attempting non-strict loading...")
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded BLIP model weights from {weights_path} with strict=False.")
        return model, processor
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Custom BlipForImageClassification (assuming this class is defined elsewhere or should be included)
# Including it here for completeness, as it was in previous BLIP scripts
class BlipForImageClassification(torch.nn.Module):
    def __init__(self, blip_model, num_labels):
        super().__init__()
        self.blip = blip_model
        # The vision_model.pooler_output provides a pooled representation of the image.
        # Its dimension is blip_model.config.vision_config.hidden_size (e.g., 1024 for blip-large).
        self.classifier = torch.nn.Linear(self.blip.config.vision_config.hidden_size, num_labels)

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        # Get image features from BLIP's vision encoder
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs.pooler_output # Shape: (batch_size, hidden_size)

        # Pass through the classification head
        logits = self.classifier(image_features)

        # For testing, we only need logits.
        return {'logits': logits}


# Run inference
def test_model(model, model_name, processor, images, labels):
    model.eval()
    predictions = []
    true_labels = []

    # Text inputs for CLIP
    if model_name == "clip":
        if args.clip_prompt == "category":
            text_inputs = [f"{category}" for category in categories]
        elif args.clip_prompt == "product":
            text_inputs = [f"A {category.lower()} product" for category in categories]
        else:  # description
            text_inputs = [
                "Jewelry, bags, or other fashion accessories" if category == "Accessories" else
                "Clothing items like shirts, pants, or dresses" if category == "Apparel" else
                "Shoes, sneakers, or boots" if category == "Footwear" else
                "Promotional or complimentary items" if category == "Free Items" else
                "Skincare, haircare, or hygiene products" for category in categories
            ]
        print(f"Using CLIP prompts: {text_inputs}")

    with torch.no_grad():
        for img_path, label in tqdm(zip(images, labels), total=len(images), desc=f"Testing {model_name}"):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
                continue
            if model_name == "clip":
                # For CLIP, image is passed directly to processor along with text
                inputs = processor(text=text_inputs, images=img, return_tensors="pt", padding=True).to(device)
                outputs = model(**inputs)
                logits = outputs.logits_per_image # CLIP outputs logits_per_image for image-text similarity
                pred = torch.argmax(logits, dim=1).cpu().item()
            else:  # SWIN or BLIP
                img_tensor = transform(img).unsqueeze(0).to(device)
                # For SWIN, model.forward expects pixel_values=images
                # For BLIP (custom BlipForImageClassification), model.forward expects pixel_values=images
                outputs = model(img_tensor)
                if model_name == "blip":
                    logits = outputs['logits'] # Your custom BlipForImageClassification returns a dict
                else: # Swin
                    logits = outputs # Your custom SwinClassifier returns logits directly
                pred = torch.argmax(logits, dim=1).cpu().item()

            predictions.append(pred)
            true_labels.append(label)

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")
    # Ensure all target names are present, including those with zero support
    unique_true_labels_in_test = sorted(list(set(true_labels)))
    target_names_for_report = [categories[i] for i in unique_true_labels_in_test]

    # Handle cases where true_labels might be empty
    if not true_labels:
        print("No true labels found for classification report. Skipping report generation.")
        class_report = "No data to generate classification report."
    else:
        class_report = classification_report(true_labels, predictions,
                                            target_names=target_names_for_report, zero_division=0) # Add zero_division

    # Generate confusion matrix
    if not true_labels or not predictions:
        print("No data to generate confusion matrix. Skipping confusion matrix generation.")
    else:
        cm = confusion_matrix(true_labels, predictions, labels=range(num_classes))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
        plt.title(f"Confusion Matrix for {model_name.upper()}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir, f"confusion_matrix_{model_name.lower()}.png"))
        plt.close()


    # Save results to log
    log_path = os.path.join("logs", f"{model_name}_test_log.txt")
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"Results saved to {log_path} and confusion matrix saved to {args.output_dir}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", class_report)


# Main execution
if __name__ == "__main__":
    # Ensure output and log directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)  # Ensure the 'logs' directory exists

    try:
        images, labels = load_test_data(args.data_dir)
        model, processor = load_model(args.model, args.weights)
        model.to(device)
        test_model(model, args.model, processor, images, labels)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")