# LoRA StyleGAN Ablation Study

## 1. Download pretrained StyleGAN2 model

Place the model in the repository root:

```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-256x256.pkl
```

---

## 2. Create environment

```bash
conda create -n lora python=3.11.14 -y
conda activate lora

pip install -r requirements.txt
```

---

## 3. Verify GPU availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:

```text
True
```

---

## 4. Quick test

```bash
python train_lora_parametric.py     --run-name test     --steps 100     --rank 8     --batch-size 16
```

---

## 5. Ablation Study

Learning rate is fixed to:

```text
1e-4
```

Run the experiments listed below.

### Baseline

```bash
python train_lora_parametric.py     --run-name baseline     --steps 1000     --rank 8     --batch-size 16
```
### Baseline

```bash
python train_lora_parametric.py \
    --run-name baseline \
    --steps 1000 \
    --rank 8 \
    --batch-size 16
```

### Experiment 1

```bash
python train_lora_parametric.py \
    --run-name steps_500 \
    --steps 500 \
    --rank 8 \
    --batch-size 16
```

### Experiment 2

```bash
python train_lora_parametric.py \
    --run-name steps_5000 \
    --steps 5000 \
    --rank 8 \
    --batch-size 16
```

### Experiment 3

```bash
python train_lora_parametric.py \
    --run-name rank_2 \
    --steps 1000 \
    --rank 2 \
    --batch-size 16
```

### Experiment 4

```bash
python train_lora_parametric.py \
    --run-name rank_4 \
    --steps 1000 \
    --rank 4 \
    --batch-size 16
```

### Experiment 5

```bash
python train_lora_parametric.py \
    --run-name rank_16 \
    --steps 1000 \
    --rank 16 \
    --batch-size 16
```

### Experiment 6

```bash
python train_lora_parametric.py \
    --run-name batch_32 \
    --steps 1000 \
    --rank 8 \
    --batch-size 32
```

### Experiment 7

```bash
python train_lora_parametric.py \
    --run-name batch_64 \
    --steps 1000 \
    --rank 8 \
    --batch-size 64
```

### Experiment 8

```bash
python train_lora_parametric.py \
    --run-name lr_5e5 \
    --steps 1000 \
    --rank 8 \
    --batch-size 16 \
    --lr 5e-5
```

### Experiment 9

```bash
python train_lora_parametric.py \
    --run-name lr_5e4 \
    --steps 1000 \
    --rank 8 \
    --batch-size 16 \
    --lr 5e-4
```

---

## 6. Outputs

Each run creates:

```text
runs/<run-name>/
├── training_log.csv
├── samples/
└── checkpoints/
```

Important checkpoints:

```text
lora_best.pt
lora_final.pt
```

Intermediate checkpoints are saved every 100 training steps.
