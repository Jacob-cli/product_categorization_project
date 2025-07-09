# Comparative Study of Attention Mechanisms in Vision-Language Models for Product Categorization

This repository contains the official implementation for the final year project "A Comparative Study of Attention Mechanisms in Vision-Language Models for Product Categorization." The project investigates the performance of state-of-the-art vision-language models (VLMs) — specifically CLIP, BLIP, and GIT (Generative Image-to-text Transformer) — in accurately categorizing fashion products from the Myntra dataset. This README provides detailed instructions for environment setup, data preparation, model training, evaluation, and accessing the interactive demonstration interface, ensuring full reproducibility and academic rigor.

## 1. Project Overview

### 1.1. Objective
The primary objective of this research is to conduct a comparative study on the efficacy of different attention mechanisms within Vision-Language Models (VLMs) for the task of fine-grained product categorization. We specifically evaluate CLIP (Contrastive Language-Image Pre-training), BLIP (Bootstrapping Language-Image Pre-training), and GIT (Generative Image-to-text Transformer) on a curated subset of the Myntra fashion product dataset. The research aims to identify which VLM architecture, particularly their attention mechanisms, offers superior performance in accurately classifying products into predefined categories, thereby contributing to enhanced e-commerce search and recommendation systems.

### 1.2. Key Contributions
* **Empirical Comparison**: Comprehensive performance evaluation of CLIP, BLIP, and GIT models on a real-world product categorization task.
* **Reproducibility**: Detailed instructions and scripts for dataset preparation, model training, and evaluation to ensure the reproducibility of reported results.
* **Ethical Considerations**: Implementation of data balancing techniques (oversampling for minority classes) and tracking of per-class fairness metrics (precision, recall, F1-score) to mitigate bias.
* **Interactive Interface**: A Streamlit-based web application for real-time product categorization inference, demonstrating practical applicability.

### 1.3. Dataset
* **Source**: A curated subset of the Myntra Fashion Products dataset, derived from an original CSV containing 44,424 entries. link: [text](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
* **Selected Categories**: Five major product categories were chosen for fine-grained classification: `Accessories`, `Apparel`, `Footwear`, `Free Items`, and `Personal Care`.
* **Balancing and Augmentation**: To address class imbalance and enhance model generalization, the dataset was balanced to contain 600 samples per class, resulting in a total of 3,000 images. This involved oversampling minority classes (e.g., 'Free Items' from 105 to 600 samples) and undersampling majority classes where necessary.
* **Splitting**: The balanced dataset was split into training and validation sets with an 80/20 ratio (2400 training samples, 600 validation samples). Stratified splitting was employed to maintain class distribution across splits.
* **Data Augmentation**: Standard image augmentation techniques (e.g., `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomRotation`, `RandomPerspective`, `ToTensor`, `Normalize`) were applied during training to improve model robustness.

### 1.4. Models
The following Vision-Language Models (VLMs) were fine-tuned for product categorization:
* **CLIP (Contrastive Language-Image Pre-training)**: Utilized `openai/clip-vit-base-patch32` as the base model, fine-tuned with a classification head.
* **BLIP (Bootstrapping Language-Image Pre-training)**: Employed `Salesforce/blip-image-captioning-base` for its image-to-text capabilities, adapted with a custom classification head.
* **GIT (Generative Image-to-text Transformer)**: Based on `microsoft/git-base`, similarly adapted for image classification.

### 1.5. Hardware
All training and inference procedures were conducted on a CPU-based system, leveraging PyTorch with 8 CPU threads, demonstrating the feasibility of deploying such models without high-end GPU resources.

## 2. Repository Structure

```markdown
├── data/
│   ├── raw/
│   │   └── myntra\_fashion\_products/
│   │       ├── images/               \# Original images (full dataset not included)
│   │       └── styles.csv            \# Original dataset metadata
│   └── processed/
│       ├── dataset\_info.json         \# Contains class mappings, dataset statistics, and augmentation details
│       └── images/                   \# Processed, balanced, and augmented images (ready for training)
├── scripts/
│   ├── train\_blip.py                 \# Script for training BLIP classification model
│   ├── train\_blip\_large.py           \# Script for training larger BLIP model variant
│   ├── train\_clip.py                 \# Script for training CLIP classification model
│   ├── train\_git.py                  \# Script for training GIT classification model
│   └── utils.py                      \# Utility functions (e.g., plot\_confusion\_matrix)
├── trained\_models\_blip/              \# Directory to save trained BLIP model checkpoints
├── trained\_models\_clip/              \# Directory to save trained CLIP model checkpoints
├── trained\_models\_git/               \# Directory to save trained GIT model checkpoints
├── logs/
│   ├── blip\_training\_log\_*.json      \# Training logs for BLIP, including config, metrics, and fairness metrics
│   ├── clip\_training\_log\_*.json      \# Training logs for CLIP
│   └── git\_training\_log\_*.json       \# Training logs for GIT
├── results/
│   ├── blip\_confusion\_matrix\_*.png   \# Confusion matrices for BLIP
│   ├── clip\_confusion\_matrix\_*.png   \# Confusion matrices for CLIP
│   └── git\_confusion\_matrix\_*.png    \# Confusion matrices for GIT
├── app.py                            \# Streamlit application for interactive inference
├── README.md                         \# This README file
└── requirements.txt                  \# Python dependencies
```

## 3. Environment Setup

To set up the environment and install necessary dependencies, follow these steps:

### 3.1. Prerequisites
* Python 3.8 or higher.
* `git` (for cloning the repository).

### 3.2. Installation Steps
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
    (Replace `your-username/your-repository-name.git` with your actual repository URL)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include: `torch`, `torchvision`, `transformers`, `scikit-learn`, `matplotlib`, `numpy`, `pandas`, `Pillow`, `tqdm`, `streamlit`.

## 4. Dataset Preparation

The Myntra dataset is not fully included in the repository due to its size and potential licensing considerations. However, the `scripts` are designed to work with a properly structured dataset.

### 4.1. Obtaining and Structuring the Dataset
1.  **Download `styles.csv`**: Obtain the `styles.csv` file from the Myntra Fashion Products dataset. Place it in `data/raw/myntra_fashion_products/`.
2.  **Download Images**: Download the corresponding images (as referenced in `styles.csv`). It is assumed that these images will be placed in `data/raw/myntra_fashion_products/images/`.

### 4.2. Processing and Balancing the Dataset
The training scripts (`train_*.py`) automatically handle dataset loading, balancing, and splitting based on the configurations specified within them and the `dataset_info.json` file. The `dataset_info.json` file is crucial for defining class mappings and augmentation parameters.

An example `dataset_info.json` structure (generated during an initial data processing step if not already present) is as follows:

```json
{
  "total_original_csv_entries": 44424,
  "selected_categories_original_counts": {
    "Accessories": 11274,
    "Apparel": 21397,
    "Footwear": 9219,
    "Free Items": 105,
    "Personal Care": 2403
  },
  "target_samples_per_class": 600,
  "total_samples_after_augmentation": 3000,
  "training_samples": 2400,
  "validation_samples": 600,
  "augmentation_applied": {
    "Apparel": { "original": 21397, "augmented_added": 0, "total_after_aug": 600 },
    "Accessories": { "original": 11274, "augmented_added": 0, "total_after_aug": 600 },
    "Footwear": { "original": 9219, "augmented_added": 0, "total_after_aug": 600 },
    "Personal Care": { "original": 2403, "augmented_added": 0, "total_after_aug": 600 },
    "Free Items": { "original": 105, "augmented_added": 495, "total_after_aug": 600 }
  },
  "ethical_considerations": {
    "balanced_sampling": true,
    "stratified_split": true,
    "data_augmentation_for_imbalance": true,
    "fairness_metrics_tracked": true
  },
  "class_mapping": {
    "Accessories": 0,
    "Apparel": 1,
    "Footwear": 2,
    "Free Items": 3,
    "Personal Care": 4
  }
}

**Note:** Ensure that your `styles.csv` contains the "Home" category if you want to explicitly remove it, as seen in the `train_git.py` log (`Removed 1 categories with only one sample: ['Home']`).

## 5\. Model Training

Each model (CLIP, BLIP, GIT) has a dedicated training script. Training logs and model checkpoints are saved in their respective directories.

### 5.1. Training Configuration

All training scripts are configured with:

  * `BATCH_SIZE`: 8
  * `EPOCHS`: 10 (with early stopping patience of 3)
  * `LEARNING_RATE`: 2e-05
  * `DEVICE`: `cpu` (adaptable to `cuda` if a GPU is available and configured).
  * `PYTORCH_NUM_THREADS`: 8

### 5.2. Running Training Scripts

To reproduce the training of each model, execute the following commands from the project root directory:

  * **Train CLIP Model**:

    ```bash
    python scripts/train_clip.py
    ```

    This script will save checkpoints to `./trained_models_clip/` and logs to `./logs/`.

  * **Train BLIP Model**:

    ```bash
    python scripts/train_blip.py
    ```

    This script will save checkpoints to `./trained_models_blip/` and logs to `./logs/`.

  * **Train GIT Model**:

    ```bash
    python scripts/train_git.py
    ```

    This script will save checkpoints to `./trained_models_git/` and logs to `./logs/`.

**Output**: During training, detailed logs for each epoch (loss, accuracy, macro F1-score, and per-class F1-scores) will be printed to the console and saved as JSON files in the `logs/` directory. The best performing model checkpoint (based on validation Macro F1-score) will be saved in the respective `trained_models_*/` directory. Confusion matrices are generated and saved after each epoch's validation phase in the `results/` directory.

### 5.3. Example Training Log Excerpt (GIT Model - Epoch 4)

```json
{
  "epoch": 4,
  "training_summary": {
    "overall_loss": 0.1460,
    "overall_accuracy": 0.9579,
    "macro_f1": 0.9587,
    "per_class_f1": {
      "Accessories": 0.9084,
      "Apparel": 0.9858,
      "Footwear": 0.9967,
      "Free Items": 0.9217,
      "Personal Care": 0.9808
    }
  },
  "validation_summary": {
    "overall_loss": 0.3115,
    "overall_accuracy": 0.9150,
    "macro_f1": 0.9102,
    "per_class_f1": {
      "Accessories": 0.7716,
      "Apparel": 0.9266,
      "Footwear": 0.9874,
      "Free Items": 0.8906,
      "Personal Care": 0.9750
    },
    "confusion_matrix": [
        [76, 16, 1, 25, 2],
        [0, 120, 0, 0, 0],
        [0, 2, 118, 0, 0],
        [1, 0, 0, 118, 1],
        [0, 1, 0, 2, 117]
    ]
  },
  "early_stopping_check": "Validation Macro F1 did not improve. Patience: 2/3"
}
```

## 6\. Evaluation and Reproducibility

The training scripts automatically perform validation and log metrics. To verify the performance of the models, refer to the `logs/` directory.

### 6.1. Analyzing Training Logs

The JSON log files (`blip_training_log_*.json`, `clip_training_log_*.json`, `git_training_log_*.json`) contain detailed epoch-wise metrics, including:

  * `overall_loss`: Training/Validation loss.
  * `overall_accuracy`: Training/Validation accuracy.
  * `macro_f1`: Macro-averaged F1-score, a key metric for multi-class classification, especially with potential class imbalances.
  * `per_class_f1`: F1-score for each individual class, providing insights into fairness and performance across categories.
  * `confusion_matrix`: Raw confusion matrix for the validation set, useful for detailed error analysis.
  * `fairness_metrics`: Detailed precision, recall, and F1-score for each class.

These logs allow for direct reproduction and verification of the reported model performances.

### 6.2. Visualizing Confusion Matrices

Confusion matrices are automatically generated and saved as PNG images in the `results/` directory after each validation epoch during training. These provide a visual representation of the classification performance across different classes, highlighting misclassifications.

Example file naming convention: `git_confusion_matrix_20250708_081148_epoch_4.png`

## 7\. Interactive Interface (Streamlit)

An interactive web application built with Streamlit (`app.py`) is provided to demonstrate the real-time product categorization capabilities of the trained models.

### 7.1. Running the Streamlit Application

1.  **Ensure models are trained and saved**: Make sure you have at least one trained model (`.pth` or `.pt` file) in its respective `trained_models_*/` directory. The `app.py` script automatically attempts to load the latest or a specified model.
2.  **Ensure `dataset_info.json` exists**: This file is critical for mapping class IDs to names.
3.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

### 7.2. Interface Features

  * **Model Selection**: Users can select which fine-tuned VLM (CLIP, BLIP, or GIT) to use for inference.
  * **Image Upload**: Upload an image of a product for categorization.
  * **Real-time Prediction**: The application will display the predicted category and, for CLIP and GIT, the top 5 predicted probabilities across classes. For BLIP, it will display the generated text description.
  * **BLIP Specific Note**: For BLIP, if the generated text doesn't directly map to a known category, the application will attempt to find the closest matching category or indicate "Unknown Category," highlighting the generative nature of BLIP compared to direct classification models.

## 8\. Ethical Considerations and Responsible AI

Throughout this project, conscious efforts were made to align with AI ethics and responsible AI guidelines:

  * **Bias Mitigation**:
      * **Balanced Sampling**: The dataset was rigorously balanced by oversampling minority classes to prevent models from being biased towards majority categories.
      * **Stratified Splitting**: Data splits ensured that class distributions were maintained, preventing skewed training or validation sets.
      * **Data Augmentation for Imbalance**: Augmentation techniques were specifically leveraged to expand the representation of underrepresented classes.
  * **Transparency and Fairness**:
      * **Fairness Metrics**: Per-class precision, recall, and F1-scores were tracked and logged for both training and validation phases, allowing for a detailed understanding of how each model performs across different product categories. This helps identify if any specific category is consistently underperforming, indicating potential bias.
      * **Confusion Matrices**: Visual confusion matrices provide clear insights into misclassifications, further aiding in the analysis of model fairness.

These measures contribute to building more robust and equitable AI systems for product categorization.

## 9\. Future Work

Potential directions for future research include:

  * Exploring more advanced multimodal fusion techniques.
  * Investigating the impact of larger pre-trained models and longer training durations on performance.
  * Evaluating the models on more diverse and larger e-commerce datasets.
  * Developing methods for continuous learning and adaptation to new product categories.
  * Implementing explainability techniques to understand model decisions better.

## 10\. Contact

For any questions, issues, or feedback regarding this implementation, please feel free to open an issue on the GitHub repository or contact me at ekemodeadesola.xyz@gmail.com / michaelkelly4573@gmail.com