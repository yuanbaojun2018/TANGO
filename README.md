# TANGO: Task-subtask Disentanglement and Coupling for Relation Triplet Extraction

This repository is built upon the LLaMA-Factory framework and is used for instruction-tuning and prediction on Relation Triplet Extraction (RTE) tasks.

---

# 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install the project in editable mode:

```bash
pip install -e .
```

---

# 2. Prepare Dataset

Place your dataset file under the `data/` directory.

Example:

```text
data/
├── dataset_info.json
├── NYT_train.json
└── NYT_test.json
```

---

# 3. Register Your Dataset

Open:

```bash
data/dataset_info.json
```

Add a new dataset entry:

Example:

```json
{
  "NYT_train": {
    "file_name": "NYT_train.json",
    "file_sha1": ""
  },

  "NYT_test": {
    "file_name": "NYT_test.json",
    "file_sha1": ""
  }
}
```

The dataset name (e.g., `NYT_test`) will be used in the `--dataset` argument.

---

# 4. Run Training on NYT

```bash
python src/train_bash.py 
--stage sft 
--do_train True 
--model_name_or_path /home/nlp/LLaMA-Factory/model/llama-2-7b-hf 
--finetuning_type lora 
--template default 
--dataset_dir data 
--dataset NYT
--cutoff_len 1024
--learning_rate 3e-4 
--num_train_epochs 20.0 
--per_device_train_batch_size 4 
--gradient_accumulation_steps 64 
--lr_scheduler_type cosine 
--logging_steps 10
--warmup_steps 100  
--optim adamw_torch  
--output_dir train_NYT
--lora_target q_proj,k_proj,v_proj,o_proj 
--lora_dropout 0.10
--lora_rank 16
--lora_alpha 32 
--plot_loss  
--fp16 

```

Prediction results will be saved to:

```text
/home/nlp/LLaMA-Factory/train_NYT
```

---

# 5. Run Prediction on NYT

```bash
python src/train_bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path /home/nlp/LLaMA-Factory/model/llama-2-7b-hf \
    --adapter_name_or_path train_NYT \
    --dataset NYT_test \
    --dataset_dir data \
    --template default \
    --finetuning_type lora \
    --output_dir /home/nlp/LLaMA-Factory/predict_result_NYT_test \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate
```

Prediction results will be saved to:

```text
/home/nlp/LLaMA-Factory/predict_result_NYT_test
```

---

# 6. Parameter Description

| Parameter                      | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `--model_name_or_path`         | Path to the base LLaMA model                   |
| `--adapter_name_or_path`       | Path to the trained LoRA adapter               |
| `--dataset`                    | Dataset name registered in `dataset_info.json` |
| `--dataset_dir`                | Dataset directory                              |
| `--template`                   | Prompt template                                |
| `--finetuning_type`            | Fine-tuning method (LoRA)                      |
| `--output_dir`                 | Directory for prediction outputs               |
| `--cutoff_len`                 | Maximum sequence length                        |
| `--preprocessing_num_workers`  | Number of preprocessing workers                |
| `--per_device_eval_batch_size` | Evaluation batch size                          |
| `--predict_with_generate`      | Enable generation-based prediction             |

---


