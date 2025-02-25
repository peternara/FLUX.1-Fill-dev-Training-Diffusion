from torch.utils.data import Dataset
from torchvision import transforms

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
        source_image = self.dataset["train"][self.source_column_name][index]
        target_image = self.dataset["train"][self.target_column_name][index]
        mask_image = self.dataset["train"][self.mask_column_name][index]
        
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