from torch.utils.data import Dataset
from torchvision import transforms
import itertools   

class BgAiDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
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
        repeats=1,
    ):

        self.custom_instance_prompts = None
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

        # Load the dataset from Hugging Face Hub
        dataset = load_dataset(self.dataset_name)
        source_images = dataset["train"][self.source_column_name]
        target_images = dataset["train"][self.target_column_name]
        mask_images = dataset["train"][self.mask_column_name]

        if self.static_caption:
            instance_prompts = [self.static_caption] * len(source_images) * repeats
        else:
            instance_prompts = dataset["train"][self.caption_column_name] * repeats

        # Repeat the images based on the repeats argument
        self.source_images = list(itertools.chain.from_iterable(itertools.repeat(img, repeats) for img in source_images))
        self.target_images = list(itertools.chain.from_iterable(itertools.repeat(img, repeats) for img in target_images))
        self.mask_images = list(itertools.chain.from_iterable(itertools.repeat(img, repeats) for img in mask_images))
        self.custom_instance_prompts = list(itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in instance_prompts))

        # Transformations
        width, height = self.size
        mask_transform = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()  
            ]
        )
        train_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.source_pixel_values = []
        self.target_pixel_values = []
        self.mask_pixel_values =[]
        # Iterate over both instance and source images
        for source_image, target_image, mask_image in zip(self.source_images, self.target_images, self.mask_images):
            if not source_image.mode == "RGB":
                source_image = source_image.convert("RGB")
            if not target_image.mode == "RGB":
                target_image = target_image.convert("RGB")
            if not mask_image.mode == "L":
                mask_image = mask_image.convert("L")

            instance_tensor = train_transforms(source_image)  # Source
            source_tensor = train_transforms(target_image)    # Target
            mask_tensor = mask_transform(mask_image)          # Apply transformations to mask_image

            self.source_pixel_values.append(instance_tensor)
            self.target_pixel_values.append(source_tensor)
            self.mask_pixel_values.append(mask_tensor)

        # Update the instance count and dataset length
        self.num_source_images = len(self.source_images)
        self._length = self.num_source_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # Fetch the concatenated image from pixel_values
        source_image = self.source_pixel_values[index % self.num_source_images]
        target_image = self.target_pixel_values[index % self.num_source_images]
        mask = self.mask_pixel_values[index % self.num_source_images]
        caption = self.custom_instance_prompts[index % self.num_source_images]

        example["source_image"] = source_image
        example["target_image"] = target_image
        example["mask"] = mask
        example["caption"] = caption

        return example