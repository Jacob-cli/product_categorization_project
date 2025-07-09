import torch
from PIL import Image
import os
import json
import argparse
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPForImageClassification, BlipForConditionalGeneration, \
    BlipProcessor
from torch.nn import functional as F
import sys

# Add the project root to the sys.path to import custom model classes
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- Import your custom model classes ---
# GitClassificationModel is defined in train_git.py
from scripts.train_git import GitClassificationModel

# --- Configuration ---
DEVICE = "cpu"  # Set the device to 'cpu' as per your setup

# Path to your processed dataset info (for category mapping)
DATASET_INFO_PATH = os.path.join(project_root, r"data\processed\processed_balanced_dataset\dataset_info.json")

# Base paths for your trained models
TRAINED_MODELS_DIR = os.path.join(project_root, r"trained_models")

# --- Model Definitions (for dynamic loading) ---
# Each config now correctly reflects the specific Hugging Face model class used
# and the expected file extension.
MODEL_CONFIGS = {
    "git": {
        "model_name": "microsoft/git-base",
        "processor_class": AutoProcessor,
        "model_class": GitClassificationModel,  # Custom wrapper
        "hf_base_model_class": AutoModelForCausalLM,  # Base model for GitClassificationModel
        "model_file_prefix": "git_best_model_epoch_",
        "file_ext": ".pt",
        "input_features": 768,  # Output features from the base GitModel encoder
        "default_epoch": 4  # As per your logs
    },
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "processor_class": AutoProcessor,
        "model_class": CLIPForImageClassification,  # HF model with classification head
        "model_file_prefix": "clip_best_model_epoch_",
        "file_ext": ".pth",
        "default_epoch": 10  # As per your logs
    },
    "blip": {
        "model_name": "Salesforce/blip-large",
        "processor_class": BlipProcessor,  # BLIP uses BlipProcessor
        "model_class": BlipForConditionalGeneration,  # This is used for generation, as per your train_blip_large.py
        "model_file_prefix": "blip_best_model_epoch_",
        "file_ext": ".pt",  # Or .pth, check your saving if it's consistent
        "default_epoch": 4  # As per your logs
    }
}


# --- Function to load model and processor ---
def load_model_and_processor(model_type, num_labels, device, epoch=None):
    config = MODEL_CONFIGS.get(model_type.lower())
    if not config:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")

    model_name = config["model_name"]
    processor_class = config["processor_class"]
    model_class = config["model_class"]
    hf_base_model_class = config.get("hf_base_model_class")  # Used for GitClassificationModel

    # Construct the model path
    epoch_str = str(epoch if epoch is not None else config["default_epoch"])
    model_filename = f"{config['model_file_prefix']}{epoch_str}{config['file_ext']}"
    model_path = os.path.join(TRAINED_MODELS_DIR, model_type.lower(), model_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please check the path and filename.")

    print(f"Loading processor for {model_name}...")
    processor = processor_class.from_pretrained(model_name)
    print("Processor loaded.")

    print(f"Loading {model_type} model from {model_path}...")

    # Load model based on its specific class and saving method
    if model_type.lower() == "git":
        base_model = hf_base_model_class.from_pretrained(model_name)
        model = model_class(base_model, config["input_features"], num_labels)
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type.lower() == "clip":
        model = model_class.from_pretrained(model_name,
                                            num_labels=num_labels)  # CLIPForImageClassification takes num_labels
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif model_type.lower() == "blip":
        # BLIP is used for text generation (category name as token sequence)
        # It needs to be initialized with is_decoder=True if not already, and text vocab size
        # However, it's safer to load directly from_pretrained and then load state_dict.
        # Your train_blip_large.py implies loading AutoModelForVision2Seq directly.
        model = model_class.from_pretrained(model_name)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise ValueError(f"Prediction logic not implemented for model type: {model_type}")

    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded and set to evaluation mode.")
    return processor, model


# --- Main prediction function ---
def predict_image(image_path, processor, model, model_type, id_to_category, category_to_id, device):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    # Process the image and get inputs
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Perform inference based on model type
    with torch.no_grad():  # Ensure no gradient computation for inference
        if model_type.lower() == "git":
            _, logits = model(pixel_values=inputs.pixel_values, labels=None)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            predicted_category = id_to_category.get(str(predicted_class_id),
                                                    "Unknown Category (ID: " + str(predicted_class_id) + ")")
            # For GIT/CLIP, probabilities are direct classification probabilities
            prob_dict = {id_to_category.get(str(idx), f"ID_{idx}"): f"{prob:.4f}" for idx, prob in
                         enumerate(probabilities.squeeze().tolist())}
            return predicted_category, prob_dict

        elif model_type.lower() == "clip":
            outputs = model(**inputs)
            logits = outputs.logits  # CLIPForImageClassification directly outputs logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(probabilities, dim=-1).item()
            predicted_category = id_to_category.get(str(predicted_class_id),
                                                    "Unknown Category (ID: " + str(predicted_class_id) + ")")
            prob_dict = {id_to_category.get(str(idx), f"ID_{idx}"): f"{prob:.4f}" for idx, prob in
                         enumerate(probabilities.squeeze().tolist())}
            return predicted_category, prob_dict

        elif model_type.lower() == "blip":
            # For BLIP, we generate text and then map it to a category
            # We need to provide a prompt for BLIP's generation, e.g., "the category of this image is"
            # However, your training script directly uses category labels as `labels_input_ids`.
            # So, we'll try to generate the category name directly.
            # You might need to set `max_new_tokens` and `num_beams` for better generation.
            # Assuming the model was fine-tuned to output the category name directly after seeing the image.

            # The BlipProcessor can handle image and text inputs.
            # For pure image-to-text generation of a label, no explicit text input might be needed
            # as the model learns to generate the category based on the image.

            # Ensure the processor has a tokenizer (BlipProcessor usually does)
            if not hasattr(processor, 'tokenizer'):
                raise AttributeError("BLIP processor does not have a tokenizer. Cannot perform text generation.")

            # Generate tokens (category name)
            generated_ids = model.generate(pixel_values=inputs.pixel_values, max_new_tokens=10, num_beams=1)

            # Decode generated tokens to text
            generated_text = processor.decode(generated_ids[0],
                                              skip_special_tokens=True).strip().lower()  # Convert to lower for matching

            # Map generated text to official category names
            # This requires a robust mapping, possibly fuzzy matching or ensuring generated text is exact.
            # For now, a direct lookup after converting to canonical format (e.g., lowercasing).

            # Assuming your `id_to_category` values are the "canonical" category names
            # and that `generated_text` will closely match one of these.

            predicted_category = "Unknown Category (Generated: " + generated_text + ")"
            # Simple direct match
            for category_name_val in id_to_category.values():
                if generated_text == category_name_val.lower():  # Compare lowercase generated text with lowercase canonical names
                    predicted_category = category_name_val
                    break

            # BLIP doesn't give "probabilities per class" in the same way for generation.
            # It gives token probabilities. We can return an empty dict for probabilities or log the generated text.
            return predicted_category, {"generated_text": generated_text}
        else:
            return "Error: Unknown model type for prediction.", {}


# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict product category using a specified VLM.")
    parser.add_argument("model_type", type=str, choices=list(MODEL_CONFIGS.keys()),
                        help=f"Type of model to use for prediction. Choose from: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to the image file for prediction. If not provided, uses a default example.")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch number of the saved model to load. Uses default if not specified.")

    args = parser.parse_args()

    # Load dataset info for category mapping
    with open(DATASET_INFO_PATH, 'r') as f:
        dataset_info = json.load(f)
    id_to_category = {str(v): k for k, v in dataset_info['category_mapping'].items()}
    category_to_id = {k: str(v) for k, v in dataset_info['category_mapping'].items()}  # Also needed for BLIP mapping
    num_labels = len(id_to_category)

    try:
        # Load the specified model and processor
        processor, model = load_model_and_processor(args.model_type, num_labels, DEVICE, args.epoch)
    except (FileNotFoundError, ValueError, AttributeError) as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Determine image path
    image_path_to_predict = args.image_path
    if image_path_to_predict is None:
        # Fallback to a default example image if none provided
        example_image_path = os.path.join(project_root, r"data\raw\myntra_fashion_products\images\10001.jpg")
        if os.path.exists(example_image_path):
            image_path_to_predict = example_image_path
        else:
            print("No image path provided and default example image not found.")
            print("Please provide an image path using --image_path argument or ensure default image exists.")
            sys.exit(1)

    print(f"\n--- Predicting for image: {image_path_to_predict} using {args.model_type.upper()} model ---")
    predicted_category, probabilities_or_generated_text = predict_image(
        image_path_to_predict, processor, model, args.model_type, id_to_category, category_to_id, DEVICE
    )

    if predicted_category:
        print(f"Predicted Category: {predicted_category}")
        if args.model_type.lower() in ["git", "clip"]:
            print("Probabilities per class (Top 5):")
            # Sort probabilities for cleaner output and limit to top 5
            sorted_probs = sorted(probabilities_or_generated_text.items(), key=lambda item: float(item[1]),
                                  reverse=True)[:5]
            for category, prob in sorted_probs:
                print(f"  {category}: {prob}")
        elif args.model_type.lower() == "blip":
            print(f"BLIP Generated Text: {probabilities_or_generated_text.get('generated_text', 'N/A')}")
            if predicted_category.startswith("Unknown Category"):
                print("Note: BLIP's generated text did not directly match a known category.")
    else:
        print("Prediction failed.")