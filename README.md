# DeepDive LoRA – GPT-2 SMS Spam Classification

This project fine-tunes GPT-2 for SMS spam classification on the UCI SMS Spam Collection dataset, using **LoRA** (Low-Rank Adaptation) and several alternative fine-tuning strategies.

The main script:
- Downloads and preprocesses the SMS Spam dataset  
- Builds train/validation/test splits  
- Initializes a GPT-style classifier from saved base weights  
- Applies LoRA (or other trainable layer settings)  
- Trains the model and logs metrics  
- Saves per-experiment results to CSV

---

## 1. Requirements

Python 3.9+ is recommended.

Key Python packages (see code):

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `transformers`
- `tiktoken`
- `tensorflow` (logged, not directly used here)

Install dependencies, for example:

```bash
pip install torch numpy pandas matplotlib transformers tiktoken
```

## 2. Data

The script automatically downloads the SMS Spam Collection from UCI:

URL: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip

- Extracts and converts it into a TSV/CSV format
- Balances the dataset and maps labels:
 - ham → 0
 - spam → 1

- Preprocessing utilities are in preprocess_data.py:
- download_data
- create_balanced_dataset
- randomly_split_dataset
- makespamDataset

- Train/validation/test splits are saved as:
  - train.csv
  - val.csv
  - test.csv

## 3. Models

The script uses GPT-2-style models and configuration defined in:

- preprocess.py 
- Model_setup.py (GPT, calc_accuracy, train_classifier, plot_metrics)
- Lora_setup.py (LoRAInjectedLinear)
- Supported GPT-2 sizes (via MODEL_NAMES):gpt2-small (124M)

## 4. LoRA and Trainable Layers

The script supports different adaptation strategies:

- trainable_layers="lora"
    - Freeze all base parameters, inject LoRA into every nn.Linear, and train:LoRA parameters and classification head
- trainable_layers="last_block"
    - Train only the last Transformer block and the classification head.
- trainable_layers="last_layer"
    - Train only the classification head.
- trainable_layers="all"
    - Train all model parameters (full fine-tuning).
- LoRA injection is performed using:-
   ```bash
    apply_lora_to_linear_layers(model, rank=rank, alpha=alpha)
   ```
## 5. Running Experiments

```bash
python exp2.py \
  --model_size "gpt2-small (124M)" \
  --lora_rank 4 \
  --lora_alpha 4 \
  --trainable_layers lora last_block last_layer all \
  --num_epochs 20 \
  --batch_size 8 \
  --eval_iter 50 \
  --eval_iter_val 5

  Key arguments

-  --model_size
  - GPT-2 variant. Default: "gpt2-small (124M)"

-  --lora_rank
  - LoRA rank r. Default: 4

-  --lora_alpha
  - LoRA scaling factor α. Default: 4

-  --trainable_layers
  - One or more of: lora, last_block, last_layer, all
  - (loops over the provided list)

-  --num_epochs
  - Training epochs per run. Default: 20

-  --batch_size
  - Batch size for train/val/test. Default: 8

-  --eval_iter, --eval_iter_val
  - Frequency of training and validation logging.
```
## 6. Outputs
For each `trainable_layers` setting, the script:

- Trains a fresh model and evaluates:
  - Pre-train accuracies: train/val/test
  - Final accuracies: train/val/test
  - Final train/validation losses
- Saves model checkpoints to:
  - `trainable_layers_checkpoints/llmgpt_rank{r}_alpha{a}_tl_{tl}.pth`
  - Logs per-run metrics to CSV files:
  - `tl_{trainable_layers}_results.csv`
- If plotting succeeds, generates training/validation loss curves using `plot_metrics()`.

