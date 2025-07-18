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

    # (height // 2, width // 2, 3) 크기의 zero tensor 생성
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        
    # 두 번째 채널(인덱스 1)에 row 인덱스를 각 행에 반복해서 채움
    # 세부 분석 > 왜 각 (i,j) 위치에 i  ??
    #        → torch.arange(height // 2) : 0부터 (height//2 - 1)까지의 값을 담은 1D 텐서 
    #                → ex) height = 64 → torch.arange(32) = tensor([0, 1, 2, ..., 31])
    #        → [:, None]   
    #                → 위 1D 텐서를 (32, 1)로 reshape   
    #                → 그래서, 세로로 된 벡터 (column vector) 
    #        → 위의 두개를 합친 예를 보면, 
    #                → torch.arange(3)[:, None]  
    #                        → tensor([
    #                                  [0],
    #                                  [1],
    #                                  [2]
    #                                  ])    
    #        → 정리하면, 이는 broadcasting 연산을 위해서,
    #                → latent_image_ids[..., 1]의 shape은 (height//2, width//2) = (32, 32)    
    #                → torch.arange(height // 2)[:, None]의 shape은 (32, 1)
    #                → 이 두 텐서를 더하면 broadcasting에 의해 (32, 32) 텐서가 만들어지고,
    #                → 그 결과는 각 행마다 해당 행 인덱스(i) 값이 채워진 형태
    # 최종 결과 예) 실제 목적 
    # [[0, 0, 0, ..., 0],
    # [1, 1, 1, ..., 1],
    # [2, 2, 2, ..., 2],
    #  ...
    # [31, 31, 31, ..., 31]]
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    ) # (H/2, W/2), 각 (i,j) 위치에 i값이 들어감

    # 세 번째 채널(인덱스 2)에 column 인덱스를 각 열에 반복해서 채움
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    ) # (H/2, W/2), 각 (i,j) 위치에 j값이 들어감

    # 현재 latent_image_ids의 shape 저장
    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape # (H/2, W/2, 3)

    # (H/2 * W/2, 3)로 reshape: 각 pixel 위치가 하나의 3차원 벡터가 됨
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
    # height를 VAE scale factor에 맞춰서 2의 배수로 맞춤 (다운샘플링 대비)
    height = 2 * (int(height) // (vae_scale_factor * 2)) # 예: height=512, vae_scale_factor=8 → height = 2 * (512 // 16) = 64
    width = 2 * (int(width) // (vae_scale_factor * 2))   # 예: width=512, vae_scale_factor=8 → width = 2 * (512 // 16) = 64

    # latent image ID tensor 생성
    latent_image_ids = prepare_latent_image_ids(
        batch_size, height, width, device, dtype
    )
    return latent_image_ids # ((H/2)*(W/2), 3) = (height/2 * width/2, 3)
    
def encode_images_to_latents(vae, pixel_values, weight_dtype, height, width, image_processor=None):
    if image_processor is not None:
        pixel_values = image_processor.preprocess(pixel_values, height=height, width=width).to(dtype=vae.dtype, device=vae.device)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    return model_input
