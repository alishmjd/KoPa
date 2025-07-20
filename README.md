# Making Large Language Models Perform Better in Knowledge Graph Completion


## Data Explanation
Due to the size of the data, you need to download and unzip the data file data.zip from [this link](https://drive.google.com/file/d/1J1Ioi23jTMaBkBDYzfIy2MAZYMUIjFWW/view?usp=drive_link) and put them in the data. 

Sample data:
```json
  {
    "instruction": "Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity. Please determine the correctness of the triple and response True or False.",
    "input": "\nThe input triple: \n( Bo Burnham, member of, International Bank for Reconstruction and Development )\n",
    "output": "False",
    "embedding_ids": [
      1973, # Bo Burnham
      2, # member of
      123 # International Bank for Reconstruction and Development
    ]
  }
```

## Training & Test
- run KoPA tuning
```shell
# For CoDeX-S dataset
export WANDB_DISABLED=true
wandb offline
CUDA_VISIBLE_DEVICES=0 python finetune_kopa.py \
    --base_model 'baffo32/decapoda-research-llama-7B-hf' \
    --data_path 'data/CoDeX-S-train.json' \
    --output_dir './mymodel' \
    --num_epochs 3 \
    --lora_r 64 \
    --learning_rate 3e-4 \
    --batch_size 12 \
    --micro_batch_size 12 \
    --num_prefix 1 \
    --kge_model 'data/CoDeX-S-rotate.pth' \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' > log.txt &
```

- run inference
```shell
CUDA_VISIBLE_DEVICES=0 python inference_kopa.py
```

## Fine-tuned model
Fine-tuned model can be downloaded from [this link](https://drive.google.com/drive/folders/1Y5789tyVtfAV_Z03OxbeWfkgbStRxWfp?usp=sharing)

|Parameters| Value |
|----------|-------|
|epoches| 10|
|learning rate| 3e-4|
|batch size| 12|
|lora r| 8 |
|lora alpha| 16|

## Evaluation
|Parameters| Value |
|----------|-------|
|Accuracy|  |
|Presicion|   |
|Recall|    |
|F1-score|  |
