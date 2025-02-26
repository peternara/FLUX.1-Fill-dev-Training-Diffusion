# Define variables for columns
export SOURCE_COLUMN="ghost_images"
export TARGET_COLUMN="target"
export MASK_COLUMN="binary_mask"
export CAPTION_COLUMN="prompt"
export MODEL_NAME="black-forest-labs/FLUX.1-Fill-dev"
export TRAIN_DATASET_NAME="raresense/BGData"
export TEST_DATASET_NAME="raresense/BGData_Validation"
export OUTPUT_DIR="trained-flux-inpaint"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --train_dataset_name=$TRAIN_DATASET_NAME \
  --test_dataset_name=$TEST_DATASET_NAME \
  --source_column=$SOURCE_COLUMN \
  --target_column=$TARGET_COLUMN \
  --mask_column=$MASK_COLUMN \
  --caption_column=$CAPTION_COLUMN \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=10 \
  --validation_steps=5 \
  --seed="42" \
  --height=768 \
  --width=576 \
  --max_sequence_length=512  \
  --checkpointing_steps=10  \
  --report_to="wandb" \
  --train_base_model
  # --resume_from_checkpoint="latest"  \