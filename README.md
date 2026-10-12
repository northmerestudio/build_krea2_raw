# Krea 2 Turbo to RAW

Apply a Krea 2 Turbo fine-tune to the RAW model and export a complete Diffusers pipeline.

## Usage

You need three transformer `.safetensors` files (original Turbo, fine-tuned Turbo, original RAW) and the complete Krea 2 RAW Diffusers model directory.

```bash
pip install -r requirements.txt

python main.py \
  --turbo-original /path/to/turbo.safetensors \
  --turbo-finetuned /path/to/turbo-finetuned.safetensors \
  --raw-original /path/to/raw.safetensors \
  --raw-diffusers /path/to/raw-diffusers \
  --output /path/to/output
```

`--output` must be a new directory. To keep and reuse the intermediate checkpoint, add `--merged-checkpoint /path/to/merged.safetensors`.

The checkpoints must be unquantized FP16/BF16/FP32 with matching tensor layouts. The exported transformer is FP16; expect high RAM and disk usage, and check image quality before using it.
