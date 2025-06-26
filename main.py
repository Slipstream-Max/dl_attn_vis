import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np
import os
import argparse

# Global model variables
model = None
feature_extractor = None
tokenizer = None

def load_model():
    """Loads the vision-encoder-decoder model, feature extractor, and tokenizer."""
    global model, feature_extractor, tokenizer
    if model is None:
        print("Loading model...")
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning",force_download=False)
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning",force_download=False)
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning",force_download=False)
        print("Model loaded.")

def predict_step(image_paths):
    """Generates captions and identifies indices of valid (non-special) tokens."""
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    
    gen_kwargs = {
        "max_length": 16,
        "num_beams": 1,
        "output_attentions": True,
        "return_dict_in_generate": True,
    }

    output = model.generate(pixel_values, **gen_kwargs)
    
    preds = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    
    token_ids = output.sequences[0]
    special_token_ids = set(tokenizer.all_special_ids)
    # Identify indices of non-special tokens. The index `i-1` corresponds to the
    # index in the `cross_attentions` tuple.
    valid_indices = [i - 1 for i, token_id in enumerate(token_ids) if i > 0 and token_id.item() not in special_token_ids]
    
    return preds, output, images[0], valid_indices

def rollout(attentions, discard_ratio, head_fusion):
    """Calculates the rollout for attention maps."""
    result = torch.eye(attentions[0].shape[-1])
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type Not supported")

            attention_heads_fused = attention_heads_fused.squeeze(0)
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            flat[0, indices] = 0

            Ii = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * Ii) / 2
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)
    
    mask = result[0, 1:] # Discard the CLS token attention
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask

def apply_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5) -> np.ndarray:
    """Applies a colored mask to an image."""
    mask = mask[..., np.newaxis]
    color_mask = np.full(image.shape, color, dtype=np.uint8)
    masked_image = image.astype(np.float32) * (1 - alpha * mask) + alpha * mask * color_mask
    return masked_image.astype(np.uint8)

def visualize_and_save_rollout(rollout_mask, image, output_path):
    """Visualizes the rollout mask on the image and saves it."""
    image_np = np.array(image)
    mask_resized = np.array(Image.fromarray((rollout_mask * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)) / 255.0
    masked_image = apply_mask(image_np, mask_resized, color=(255, 0, 0), alpha=0.5)
    Image.fromarray(masked_image).save(output_path)

def main():
    parser = argparse.ArgumentParser(description='Generate image captions and attention rollouts.')
    parser.add_argument('-i', '--image-path', type=str, help='Path to the input image.')
    args = parser.parse_args()

    load_model()

    print(f"\nProcessing image: {args.image_path}")
    preds, output, image_pil, valid_indices = predict_step([args.image_path])
    
    print(f"\nGenerated Caption: '{preds[0]}'")

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_dir = f"rollout_{image_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving rollout images to '{output_dir}/'")

    # Use the pre-calculated valid indices to loop through tokens and attentions
    cross_attentions = output.cross_attentions
    token_ids = output.sequences[0]
    saved_image_count = 0

    for attention_idx in valid_indices:
        # The token_id's index in the sequence is attention_idx + 1
        token_id = token_ids[attention_idx + 1].item()
        token_word = tokenizer.decode(token_id)
        
        # Sanitize filename and check if it's empty
        safe_token_word = "".join(c if c.isalnum() else "_" for c in token_word).strip('_')
        if not safe_token_word:
            continue

        attention_layer = cross_attentions[attention_idx]
        rollout_mask = rollout(attention_layer, discard_ratio=0.8, head_fusion='mean')
        
        output_path = os.path.join(output_dir, f"rollout_{saved_image_count}_{safe_token_word}.png")
        
        visualize_and_save_rollout(rollout_mask, image_pil, output_path)
        saved_image_count += 1

    print(f"\nProcessing complete. {saved_image_count} rollout images saved.")

if __name__ == "__main__":
    main()