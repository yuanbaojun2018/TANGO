# TANGO
Paper: Subtasks Matter! TANGO: Task-subtask Disentanglement and Coupling for Generative Relation Triplet Extraction


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
├── new_dataset_name.json
├── WebNLG_test_new.json
└── WebNLG_star_test_new.json
```

---

# 3. Register Your Dataset

Open:

```bash
data/dataset_info.json
```

Add a new dataset entry:

```json
{
  "new_dataset": {
    "file_name": "new_dataset_name.json",
    "file_sha1": ""
  }
}
```

Example:

```json
{
  "WebNLG_test_new": {
    "file_name": "WebNLG_test_new.json",
    "file_sha1": ""
  },

  "WebNLG_star_test_new": {
    "file_name": "WebNLG_star_test_new.json",
    "file_sha1": ""
  }
}
```

The dataset name (e.g., `WebNLG_test_new`) will be used in the `--dataset` argument.

---

# 4. Run Prediction on WebNLG

```bash
python src/train_bash.py 
--stage sft 
--do_train True 
--model_name_or_path /home/nlp/LLaMA-Factory/model/llama-2-7b-hf 
--finetuning_type lora 
--template default 
--dataset_dir data 
--dataset NYT_train
--cutoff_len 1024
--learning_rate 3e-4 
--num_train_epochs 20.0 
--per_device_train_batch_size 4 
--gradient_accumulation_steps 64 
--lr_scheduler_type cosine 
--logging_steps 10
--warmup_steps 100  
--optim adamw_torch  
--output_dir train_2024-05-18_NYT
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

# 5. Run Prediction on WebNLG-Star

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
    --output_dir /home/nlp/LLaMA-Factory/predict_result \
    --cutoff_len 1536 \
    --preprocessing_num_workers 16 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate
```

Prediction results will be saved to:

```text
/home/nlp/LLaMA-Factory/predict_result
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

# Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{tango2026,
  title={Subtasks Matter! TANGO: Task-subtask Disentanglement and Coupling for Generative Relation Triplet Extraction},
  author={Anonymous},
  journal={},
  year={2026}
}
```
