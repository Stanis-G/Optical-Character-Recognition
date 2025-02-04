import os
import random
from PIL import Image
import numpy as np
from torch.utils.data import IterableDataset


class CustomDataset(IterableDataset):

    def __init__(self, batch_size, indeces=None):
        self.batch_size = batch_size
        self.data_dir = os.path.join('custom_dataset', 'data')
        self.images_path = os.path.join(self.data_dir, 'images')
        self.texts_path = os.path.join(self.data_dir, 'texts')
        self.indeces = indeces # Indeces are needed to divide data into subsets


    def __len__(self):
        length = len(self.indeces) if self.indeces else len(os.listdir(self.texts_path))
        return length
    

    def __getitem__(self, idx):
        image_name = sorted(os.listdir(self.images_path))[idx]
        text_name = sorted(os.listdir(self.texts_path))[idx]
        img_path = os.path.join(self.images_path, image_name)
        text_path = os.path.join(self.texts_path, text_name)
        img = Image.open(img_path, mode='r')
        with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return img, text


    def __iter__(self):
        """Read data in chunks of size self.batch_size"""

        if self.indeces:
            image_names = np.array(os.listdir(self.images_path))[self.indeces]
            text_names = np.array(os.listdir(self.texts_path))[self.indeces]
        else:
            image_names = os.listdir(self.images_path)
            text_names = os.listdir(self.texts_path)

        images_batch, text_batch = [], []
        # Form one chunk, loaded into memory (not the actual batch)
        for image_name, text_name in zip(image_names, text_names):
            image_path = os.path.join(self.images_path, image_name)
            text_path = os.path.join(self.texts_path, text_name)
            img = Image.open(image_path, mode='r')
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()

            images_batch.append(np.array(img))
            text_batch.append(text)
            
            # Output chunk when it is loaded into memory
            if len(images_batch) == self.batch_size:
                yield images_batch, text_batch
                images_batch, text_batch = [], []

        # Unload the last chunk
        if images_batch:
            yield images_batch, text_batch
            images_batch, text_batch = [], []


class MappedDataset(IterableDataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor


    def __iter__(self):
        """Process chunks from dataset using processor and yielding samples by one"""

        for images_batch, text_batch in self.dataset:
            # Process the whole chunk at once, but return samples by one
            pixel_values = self.processor(images_batch, return_tensors='pt').pixel_values
            tokens = self.processor.tokenizer(text_batch, padding='max_length', truncation=True, max_length=100)
            for image, text in zip(pixel_values, tokens['input_ids']):
                yield {
                    'pixel_values': image,
                    'labels': text,
                }


class CustomDataProcessor:

    def __init__(self, processor):
        self.processor = processor


    def split(self, train_frac, val_frac, dataset_batch_size):
        dataset = CustomDataset(batch_size=dataset_batch_size)
        dataset_size = len(dataset)

        random.seed(0)
        all_indeces = list(range(dataset_size))
        random.shuffle(all_indeces)

        train_dataset_size = int(dataset_size * train_frac)
        val_dataset_size = int(dataset_size * val_frac)
        train_indeces = all_indeces[:train_dataset_size]
        val_indeces = all_indeces[train_dataset_size:train_dataset_size + val_dataset_size]
        test_indeces = all_indeces[train_dataset_size + val_dataset_size:]

        # Make iterable datasets, which are iterated over chunks of data
        train_dataset = CustomDataset(batch_size=dataset_batch_size, indeces=train_indeces)
        val_dataset = CustomDataset(batch_size=dataset_batch_size, indeces=val_indeces)
        test_dataset = CustomDataset(batch_size=dataset_batch_size, indeces=test_indeces)

        return train_dataset, val_dataset, test_dataset, train_dataset_size


    def __call__(
            self,
            dataset_batch_size=1000, # size of chunks loaded from IterableDataset
            train_frac=0.85,
            val_frac=0.1,
    ):

        train_dataset, val_dataset, test_dataset, train_dataset_size = self.split(train_frac, val_frac, dataset_batch_size)

        # Process the data, returned by dataset, and output data samples by one
        train_dataset = MappedDataset(train_dataset, self.processor)
        val_dataset = MappedDataset(val_dataset, self.processor)
        test_dataset = MappedDataset(test_dataset, self.processor)
        
        return train_dataset, val_dataset, test_dataset, train_dataset_size
