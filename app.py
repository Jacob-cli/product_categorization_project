import streamlit as st
import torch
from PIL import Image
import os
import json
from transformers import AutoProcessor, CLIPForImageClassification, BlipForConditionalGeneration, BlipProcessor
from torch.nn import functional as F
import sys
import numpy as np

# --- Configuration and Paths ---
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

# --- Import your custom model classes ---
try:
    from scripts.train_git import GitClassificationModel
except ImportError:
    st.error("Could not import GitClassificationModel from scripts.train_git. "
             "Please ensure train_git.py is correctly defined and accessible.")
    st.stop()

try:
    from scripts.train_blip import BLIPClassificationModel
except ImportError:
    st.error("Could not import BLIPClassificationModel from scripts.train_blip. "
             "Please ensure train_blip.py is correctly defined and accessible.")
    st.stop()

DEVICE = "cpu"
DATASET_INFO_PATH = os.path.join(project_root, "data", "processed", "dataset_info.json")

# --- Helper Functions ---

@st.cache_resource
def load_model_and_processor(model_type, model_path, log_path):
    processor = None
    model = None
    id_to_category = {}
    category_to_id = {}

    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                config = log_data.get('config', {})
            if 'category_mapping' in config:
                category_to_id = config['category_mapping']
            elif 'class_mapping' in config:
                category_to_id = config['class_mapping']
            else:
                if model_type.lower() == "git":
                    st.warning(f"Category mapping not found in config for {model_type} model in {log_path}. "
                               "Using a default mapping for GIT model as a fallback based on observed categories.")
                    category_to_id = {
                        'Accessories': 0,
                        'Apparel': 1,
                        'Footwear': 2,
                        'Free Items': 3,
                        'Personal Care': 4
                    }
                else:
                    st.error(
                        f"Error: Neither 'category_mapping' nor 'class_mapping' found in config for {model_type} model in {log_path}.")
                    st.stop()
            id_to_category = {v: k for k, v in category_to_id.items()}
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from {log_path}. Please check file integrity.")
            st.stop()
    else:
        st.error(f"Log file not found at: {log_path}. Please ensure it exists.")
        st.stop()

    num_classes = len(category_to_id) if category_to_id else 0

    try:
        if model_type.lower() == "blip":
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BLIPClassificationModel(base_model, num_classes, id_to_category, category_to_id)
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=DEVICE)
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    st.error(f"Error loading BLIP state dictionary: {e}")
                    st.stop()
            else:
                st.warning(f"BLIP model checkpoint not found at {model_path}. Using untrained BLIP classification model. "
                           "Prediction quality will be poor.")

        elif model_type.lower() == "clip":
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPForImageClassification.from_pretrained("openai/clip-vit-base-patch32", num_labels=num_classes)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            else:
                st.warning(f"CLIP model checkpoint not found at {model_path}. Using base CLIP model. "
                           "Prediction quality may be affected.")

        elif model_type.lower() == "git":
            processor = AutoProcessor.from_pretrained("microsoft/git-base")
            model = GitClassificationModel(model_name="microsoft/git-base", num_labels=num_classes)
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            else:
                st.warning(f"GIT model checkpoint not found at {model_path}. Using untrained GIT model. "
                           "Prediction quality will be poor.")

        else:
            st.error("Unsupported model type selected.")
            st.stop()

        model.to(DEVICE)
        model.eval()

    except Exception as e:
        st.error(f"Error loading {model_type} model or processor: {e}")
        st.stop()

    return processor, model, id_to_category, category_to_id


def predict_image(image_file, processor, model, model_type, id_to_category, category_to_id, device):
    if image_file is None:
        return None, None

    image = Image.open(image_file).convert("RGB")

    try:
        if model_type.lower() == "clip":
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_category = id_to_category.get(predicted_class_id, "Unknown Category")
            probs_dict = {id_to_category[i]: f"{p.item():.4f}" for i, p in enumerate(probabilities)}
            return predicted_category, probs_dict

        elif model_type.lower() == "git":
            text_input = "a photo of"  # Generic prompt, consistent with BLIP
            inputs = processor(images=image, text=text_input, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                # Handle tuple output; assume logits are in outputs[1] or check tuple structure
                if isinstance(outputs, tuple):
                    if len(outputs) > 1 and outputs[1] is not None:
                        logits = outputs[1]  # Try second element for logits
                    elif outputs[0] is not None:
                        logits = outputs[0]  # Fallback to first element
                    else:
                        raise ValueError(f"GIT model output tuple {outputs} contains no valid logits")
                else:
                    raise ValueError(f"Expected tuple from GIT model, got {type(outputs)}")
            probabilities = F.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_category = id_to_category.get(predicted_class_id, "Unknown Category")
            probs_dict = {id_to_category[i]: f"{p.item():.4f}" for i, p in enumerate(probabilities)}
            return predicted_category, probs_dict

        elif model_type.lower() == "blip":
            text_input = "a photo of"
            inputs = processor(images=image, text=text_input, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            with torch.no_grad():
                outputs = model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)[0]
            predicted_class_id = torch.argmax(probabilities).item()
            predicted_category = id_to_category.get(predicted_class_id, "Unknown Category")
            probs_dict = {id_to_category[i]: f"{p.item():.4f}" for i, p in enumerate(probabilities)}
            return predicted_category, probs_dict

        else:
            st.error("Prediction for this model type is not implemented yet.")
            return None, None

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.exception(e)
        return None, None


# --- Streamlit UI ---

st.set_page_config(page_title="Product Categorization App", layout="centered")

st.title("👗 Product Categorization Application 👟")
st.markdown("Upload an image of a product and let the AI categorize it!")

# Model Selection
st.sidebar.header("Model Configuration")
model_options = {
    "CLIP": {
        "model_dir": "trained_models/clip",
        "log_pattern": "clip_training_log_",
        "model_pattern": "clip_best_model_epoch_",
        "extension": ".pth"
    },
    "GIT": {
        "model_dir": "trained_models/git",
        "log_pattern": "git_training_log_",
        "model_pattern": "git_best_model_epoch_",
        "extension": ".pt"
    },
    "BLIP": {
        "model_dir": "trained_models/blip",
        "log_pattern": "blip_training_log_",
        "model_pattern": "blip_best_model_epoch_",
        "extension": ".pth"
    }
}

selected_model_type = st.sidebar.selectbox(
    "Select Model Type:", list(model_options.keys())
)

current_model_config = model_options[selected_model_type]
log_files = [f for f in os.listdir(os.path.join(project_root, "logs")) if
             f.startswith(current_model_config["log_pattern"]) and f.endswith(".json")]
log_files.sort(reverse=True)

if not log_files:
    st.sidebar.error(f"No log files found for {selected_model_type} in the 'logs' directory.")
    st.stop()

selected_log_file = st.sidebar.selectbox("Select Training Log File:", log_files)

log_path = os.path.join(project_root, "logs", selected_log_file)
model_path_dir = os.path.join(project_root, current_model_config["model_dir"])

best_model_filename = None
try:
    with open(log_path, 'r') as f:
        log_data = json.load(f)
        if 'final_best_model_path' in log_data:
            best_model_filename = os.path.basename(log_data['final_best_model_path'])
        elif selected_model_type.lower() == "git" and "epochs" in log_data and log_data["epochs"]:
            available_models = [f for f in os.listdir(model_path_dir) if
                                f.startswith(current_model_config["model_pattern"]) and f.endswith(
                                    current_model_config["extension"])]
            available_models.sort(reverse=True)
            if available_models:
                best_model_filename = available_models[0]
            else:
                st.sidebar.warning(f"No trained model files found for {selected_model_type} in '{model_path_dir}'. "
                                   "The base model will be loaded, which may lead to poor performance.")
                best_model_filename = None
        else:
            available_models = [f for f in os.listdir(model_path_dir) if
                                f.startswith(current_model_config["model_pattern"]) and f.endswith(
                                    current_model_config["extension"])]
            available_models.sort(reverse=True)
            if available_models:
                best_model_filename = available_models[0]
            else:
                st.sidebar.warning(f"No trained model files found for {selected_model_type} in '{model_path_dir}'. "
                                   "The base model will be loaded, which may lead to poor performance.")
                best_model_filename = None

except Exception as e:
    st.sidebar.warning(f"Could not determine best model from log file {selected_log_file}: {e}. "
                       "Please manually select a model if available.")
    best_model_filename = None

model_files = [f for f in os.listdir(model_path_dir) if f.endswith(current_model_config["extension"])]
model_files.sort(reverse=True)

if best_model_filename and best_model_filename in model_files:
    selected_model_file = st.sidebar.selectbox("Select Model Checkpoint:", model_files,
                                               index=model_files.index(best_model_filename))
elif model_files:
    selected_model_file = st.sidebar.selectbox("Select Model Checkpoint:", model_files)
else:
    selected_model_file = st.sidebar.selectbox("Select Model Checkpoint:", ["No models found"])
    if selected_model_file == "No models found":
        st.warning(f"No model checkpoints available for {selected_model_type} in '{model_path_dir}'. "
                   "The application will load an untrained base model, which might not perform well.")
        model_path = None
    else:
        model_path = os.path.join(model_path_dir, selected_model_file)

processor, model, id_to_category, category_to_id = None, None, None, None
if selected_model_file != "No models found":
    model_path = os.path.join(model_path_dir, selected_model_file)
    with st.spinner(f"Loading {selected_model_type} model from {model_path}..."):
        try:
            processor, model, id_to_category, category_to_id = load_model_and_processor(
                selected_model_type, model_path, log_path
            )
            st.sidebar.success("Model and processor loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            st.stop()
else:
    with st.spinner(f"Attempting to load base {selected_model_type} model..."):
        try:
            processor, model, id_to_category, category_to_id = load_model_and_processor(
                selected_model_type, None, log_path
            )
            st.sidebar.info(f"Loaded base {selected_model_type} model and category mappings.")
        except Exception as e:
            st.sidebar.error(f"Failed to load base model components: {e}")
            st.stop()

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    if st.button("Predict Category"):
        if processor and model and id_to_category and category_to_id:
            with st.spinner("Predicting..."):
                predicted_category, probabilities_or_generated_text = predict_image(
                    uploaded_file, processor, model, selected_model_type, id_to_category, category_to_id, DEVICE
                )

            if predicted_category:
                st.success(f"**Predicted Category:** {predicted_category}")

                if selected_model_type.lower() in ["git", "clip", "blip"]:
                    st.write("Probabilities per class (Top 5):")
                    sorted_probs = sorted(probabilities_or_generated_text.items(), key=lambda item: float(item[1]),
                                          reverse=True)[:5]
                    for category, prob in sorted_probs:
                        st.write(f"  - {category}: {prob}")
            else:
                st.error("Prediction failed. Please check the logs for errors.")
        else:
            st.error("Model, processor, or mappings are not loaded. Please check the sidebar for loading errors.")

st.markdown("---")
st.markdown("Developed for Final Year Project")