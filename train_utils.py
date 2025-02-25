import torch
from einops import rearrange

def prepare_fill_with_mask(
        image_processor,
        mask_processor,
        vae,
        vae_scale_factor,
        image,
        mask,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
    ):
    """
    Prepares image and mask for fill operation with proper rearrangement.
    Focuses only on image and mask processing.
    """
    # Determine effective batch size
    effective_batch_size = batch_size * num_images_per_prompt
    
    # Prepare image
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)

    image_batch_size = image.shape[0]
    repeat_by = effective_batch_size if image_batch_size == 1 else num_images_per_prompt
    image = image.repeat_interleave(repeat_by, dim=0)
    image = image.to(device=device, dtype=dtype)

    # Prepare mask with specific processing
    if isinstance(mask, torch.Tensor):
        pass
    else:
        mask = mask_processor.preprocess(mask, height=height, width=width)

    mask = mask.repeat_interleave(repeat_by, dim=0)
    mask = mask.to(device=device, dtype=dtype)

    # Apply mask to image
    masked_image = image.clone()
    masked_image = masked_image * (1 - mask)

    # Encode to latents
    image_latents = vae.encode(masked_image.to(vae.dtype)).latent_dist.sample()
    image_latents = (
        image_latents - vae.config.shift_factor
    ) * vae.config.scaling_factor
    image_latents = image_latents.to(dtype)

    # Process mask following the example's specific rearrangement
    mask = mask[:, 0, :, :] if mask.shape[1] > 1 else mask[:, 0, :, :]
    mask = mask.to(torch.bfloat16)
    
    # First rearrangement: 8x8 patches
    mask = rearrange(
        mask,
        "b (h ph) (w pw) -> b (ph pw) h w",
        ph=8,
        pw=8,
    )
    
    # Second rearrangement: 2x2 patches
    mask = rearrange(
        mask, 
        "b c (h ph) (w pw) -> b (h w) (c ph pw)", 
        ph=2, 
        pw=2
    )

    # Rearrange image latents similarly
    image_latents = rearrange(
        image_latents,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2
    )

    # Combine image and mask
    image_cond = torch.cat([image_latents, mask], dim=-1)

    return image_cond, height, width

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_latents(
    vae_scale_factor,
    batch_size,
    height,
    width,
    dtype,
    device,
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))


    latent_image_ids = prepare_latent_image_ids(
        batch_size, height, width, device, dtype
    )
    return latent_image_ids

    
def encode_images_to_latents(vae, pixel_values, weight_dtype, height, width, image_processor=None):
    if image_processor is not None:
        pixel_values = image_processor.preprocess(pixel_values, height=height, width=width).to(dtype=vae.dtype, device=vae.device)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    return model_input
