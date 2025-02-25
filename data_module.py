from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class BgAiDataset(Dataset):
    """
    This class loads a dataset using the HuggingFace datasets library for model fine-tuning.

    Args:
        dataset_name (str): Identifier for the dataset to load.
        source_column_name (str): Column name containing the source images.
        mask_column_name (str): Column name containing the mask images.
        target_column_name (str): Column name containing the target images.
        caption_column_name (str): Column name containing the captions.
        static_caption (str, optional): Common caption used for all images.
        size (tuple, optional): Desired (width, height) to which images are resized.
    """

    def __init__(
        self,
        dataset_name,
        source_column_name,
        mask_column_name,
        target_column_name,
        caption_column_name,
        static_caption=None,
        size=(768,1024),
    ):
        self.dataset_name = dataset_name
        self.source_column_name = source_column_name
        self.mask_column_name = mask_column_name
        self.target_column_name = target_column_name
        self.caption_column_name = caption_column_name
        self.static_caption = static_caption
        self.size = size

        # Load the dataset directly
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. Please install the datasets library: `pip install datasets`."
            )

        # Load dataset reference but not the actual images
        self.dataset = load_dataset(self.dataset_name)
        self._length = len(self.dataset["train"])
        
        # Setup transformations
        width, height = self.size
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()  
            ]
        )
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Load images on demand
        sample = self.dataset["train"][index]
        source_image = sample[self.source_column_name]
        target_image = sample[self.target_column_name]
        mask_image = sample[self.mask_column_name]
        
        # Convert image modes if needed
        if not source_image.mode == "RGB":
            source_image = source_image.convert("RGB")
        if not target_image.mode == "RGB":
            target_image = target_image.convert("RGB")
        if not mask_image.mode == "L":
            mask_image = mask_image.convert("L")

        # Apply transformations
        source_tensor = self.train_transforms(source_image)
        target_tensor = self.train_transforms(target_image)
        mask_tensor = self.mask_transform(mask_image)
        
        # Get caption
        if self.static_caption:
            caption = self.static_caption
        else:
            caption = self.dataset["train"][self.caption_column_name][index]
        
        # Create example dictionary
        example = {
            "source_image": source_tensor,
            "target_image": target_tensor,
            "mask": mask_tensor,
            "caption": caption
        }
        
        return example

def collate_fn(batch):
    """
    Custom collate function for BgAiDataset to properly batch tensors and captions.
    
    Args:
        batch: List of samples from BgAiDataset
        
    Returns:
        Dictionary with batched tensors and list of captions
    """
    source_images = torch.stack([item['source_image'] for item in batch])
    target_images = torch.stack([item['target_image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    return {
        "source_image": source_images,
        "target_image": target_images,
        "mask": masks,
        "caption": captions
    }

if __name__ == "__main__":
    # Create dataset instance
    dataset = BgAiDataset(
        dataset_name="raresense/BGData",
        source_column_name="ghost_images",
        mask_column_name="binary_mask",  # You'll need to replace this with the actual mask column name
        target_column_name="target",
        caption_column_name="prompt"
    )
    
    # Print dataset length
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    print("Loading first sample...")
    sample = dataset[0]
    
    # Print tensor shapes
    print("\nTensor shapes:")
    print(f"Source image: {sample['source_image'].shape}")
    print(f"Target image: {sample['target_image'].shape}")
    print(f"Mask: {sample['mask'].shape}")
    
    # Print caption (truncated if too long)
    caption = sample["caption"]
    if len(caption) > 100:
        caption = caption[:100] + "..."
    print(f"\nCaption: {caption}")

    # Create and test DataLoader with custom collate function
    batch_size = 4
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("\nTesting batch processing:")
    print(f"Getting first batch of size {batch_size}...")
    batch = next(iter(dataloader))
    
    # Print tensor shapes for batch
    print("Tensor shapes for batch:")
    print(f"Source images: {batch['source_image'].shape}")
    print(f"Target images: {batch['target_image'].shape}")
    print(f"Masks: {batch['mask'].shape}")
    print(f"Number of captions: {len(batch['caption'])}")
    
    # Print first caption in batch
    if len(batch['caption'][0]) > 100:
        first_caption = batch['caption'][0][:100] + "..."
    else:
        first_caption = batch['caption'][0]
    print(f"First caption in batch: {first_caption}")
    