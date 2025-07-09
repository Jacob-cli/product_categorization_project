# Product Categorization Thesis

This repository contains the implementation for the thesis "A Comparative Study of Attention Mechanisms on Vision Language Models for Product Categorization." It includes scripts, models, and data to train and test CLIP, SWIN, and BLIP models on the Myntra dataset for product categorization.

## Project Overview
- **Objective**: Compare the performance of attention mechanisms in CLIP, SWIN, and BLIP for categorizing products into five classes: Accessories, Apparel, Footwear, Free Items, Personal Care.
- **Dataset**: Curated subset from Myntra (500 samples, 100 per category, sourced from a 44,424-row CSV with preprocessing).
- **Models**: Fine-tuned CLIP, SWIN, and BLIP models, trained and tested on June 21-22, 2025.
- **Hardware**: CPU-based implementation.
## Folder Structure
- `data/myntradataset/train_data`: 500 samples organized by category (Accessories, Apparel, Footwear, Free Items, Personal Care).
- `scripts`: Python scripts for preprocessing, training, and testing.
- `models`: Fine-tuned model weights (`clip_optimized.pth`, `swin_best.pth`, `blip_large_finetuned.pth`).
- `logs`: Training and test logs (e.g., `clip_optimized_test_log.txt`, `swin_test_log.txt`, `training_log_blip_large_20250621`).
- `results`: Visual outputs (e.g., `confusion_matrix_blip_large.png`).

## Environment Setup
1. **Install Python**: Use Python 3.8 or higher.
2. **Install Dependencies**:
   ```bash
   pip install torch torchvision transformers timm scikit-learn matplotlib seaborn pandas numpy pillow
   ```
3. **Verify Dataset**: Ensure `data/myntradataset/train_data` contains 500 images across five categories.
4. **Verify Models**: Place `clip_optimized.pth`, `swin_best.pth`, and `blip_large_finetuned.pth` in `models`.

## Running Scripts
### 1. Verify Dataset
```bash
python scripts/verify_dataset.py
```
- Checks integrity of `data/myntradataset/train_data` against `styles.csv`.

### 2. Run Inference (Test Models)
To reproduce test results for CLIP, SWIN, and BLIP:
```bash
python scripts/test_model.py --model clip --weights models/clip_optimized.pth --data_dir data/myntradataset/train_data
python scripts/test_model.py --model swin --weights models/swin_best.pth --data_dir data/myntradataset/train_data
python scripts/test_model.py --model blip --weights models/blip_large_finetuned.pth --data_dir data/myntradataset/train_data
```
- Outputs: Accuracy, F1 scores, and per-class metrics saved to `logs`.
- Example results:
  - CLIP: Accuracy 0.8400, F1 0.8399
  - SWIN: Accuracy 0.9200, F1 0.9144
  - BLIP: Accuracy 0.7800, F1 0.7622

### 3. Generate Confusion Matrices
To generate confusion matrices for CLIP and SWIN:
```bash
python scripts/generate_confusion_matrices.py
```
- Outputs: `confusion_matrix_clip.png` and `confusion_matrix_swin.png` in `results`.

### 4. (Optional) Retrain Models
To retrain models:
```bash
python scripts/train_clip.py --data_dir data/myntradataset/train_data --output_dir models
python scripts/train_swin.py --data_dir data/myntradataset/train_data --output_dir models
python scripts/train_blip_large.py --data_dir data/myntradataset/train_data --output_dir models
```

## Expected Outputs
- **Logs**: Test results in `logs` (e.g., `clip_optimized_test_log.txt`).
- **Results**: Confusion matrices in `results` (e.g., `confusion_matrix_blip_large.png`).
- **Metrics**:
  - CLIP: Accuracy 0.84, F1 0.8399 (with per-class metrics)
  - SWIN: Accuracy 0.92, F1 0.9144 (with per-class metrics)
  - BLIP: Accuracy 0.78, F1 0.7622 (per-class metrics in `training_log_blip_large_20250621`)

## Notes
- All scripts are designed to run on CPU. Adjust `--device cuda` in scripts if GPU is available.
- The Myntra dataset is not included in full due to licensing; `train_data` contains a 500-sample subset.
- Contact the author for issues running scripts or reproducing results.