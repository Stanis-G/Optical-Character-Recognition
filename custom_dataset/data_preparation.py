import boto3
from botocore.config import Config
from dotenv import load_dotenv
from io import BytesIO
import time
import os
import random
import shutil
from PIL import Image
import yaml
import numpy as np
from torch.utils.data import IterableDataset

load_dotenv()


def profile(func):
    def wrapper(self, *args, **kwargs):
        DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        if DEBUG:
            with open('profile_logs.txt', 'a') as f:
                f.write(f'CALL {func.__name__}n')
                start = time.time()
                result = func(self, *args, **kwargs)
                end = time.time()
                f.write(f'END {func.__name__}: {end - start}\n')
            return result
        return func(self, *args, **kwargs)
    return wrapper


class TrOCRDatasetBase(IterableDataset):

    def __init__(self, batch_size, indeces=None):
        self.batch_size = batch_size
        self.indeces = indeces # Indeces are needed to divide data into subsets
        self.image_names = []
        self.text_names = []


    @profile
    def __len__(self):
        length = len(self.indeces) if self.indeces else len(self.text_names)
        return length
    

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        text_name = self.text_names[idx]
        return image_name, text_name


    @profile
    def __iter__(self):
        """Read data in chunks of size self.batch_size"""

        images_batch, text_batch = [], []
        # Form one chunk, loaded into memory (not the actual batch)
        for idx in range(len(self)):
            img, text = self.__getitem__(idx)

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


class TrOCRDatasetLocal(TrOCRDatasetBase):

    @profile
    def __init__(self, batch_size, dataset_name, indeces=None):
        super().__init__(batch_size=batch_size, indeces=indeces)
        self.data_dir = dataset_name
        self.images_path = os.path.join(self.data_dir, 'images')
        self.texts_path = os.path.join(self.data_dir, 'texts')
        self.image_names = sorted(self._read_all(self.images_path))
        self.text_names = sorted(self._read_all(self.texts_path))


    @profile
    def _read_all(self, prefix):
        full_path = os.path.join(self.data_dir, prefix)
        file_names = os.listdir(full_path)
        if self.indeces:
            file_names = [file_names[i] for i in self.indeces]

        return file_names

    
    @profile
    def __getitem__(self, idx):
        image_name, text_name = super().__getitem__(idx)
        img_path = os.path.join(self.images_path, image_name)
        text_path = os.path.join(self.texts_path, text_name)
        img = Image.open(img_path, mode='r')
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return img, text


class TrOCRDatasetS3(TrOCRDatasetBase):

    @profile
    def __init__(self, batch_size, dataset_name, indeces=None):
        super().__init__(batch_size=batch_size, indeces=indeces)
        MINIO_URL = f'http://localhost:{os.getenv("MINIO_API_PORT")}'
        self.MINIO_URL = MINIO_URL
        self.MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
        self.MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
        self.bucket = dataset_name
        self._get_s3_client()
        self.images_prefix = 'images'
        self.texts_prefix = 'texts'
        self.image_names = sorted(self._read_all('images'))
        self.text_names = sorted(self._read_all('texts'))


    @profile
    def _get_s3_client(self, retries=10):
        config = Config(retries={'max_attempts': retries, 'mode': 'standard'})
        session = boto3.session.Session()
        s3 = session.client(
            service_name='s3',
            endpoint_url=self.MINIO_URL,
            aws_access_key_id=self.MINIO_ROOT_USER,
            aws_secret_access_key=self.MINIO_ROOT_PASSWORD,
            config=config,
        )
        self.s3 = s3
    

    @profile
    def _read_file_from_s3(self, file_name):
        response = self.s3.get_object(Bucket=self.bucket, Key=file_name)

        # Read data
        file_data = response['Body'].read()

        if file_name.startswith('texts') or file_name.startswith('pages'):
            # Convert bytes to string (if text file)
            return file_data.decode("utf-8")
        elif file_name.startswith('images'):
            # Open image using PIL
            file_data = BytesIO(file_data)
            return Image.open(file_data)


    @profile
    def _read_all(self, prefix, page_size=1000):
        objects = []
        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, PaginationConfig={'PageSize': page_size}):
            objects.extend(obj["Key"] for obj in page.get("Contents", []))

        if self.indeces:
            objects = [objects[i] for i in self.indeces]

        return objects


    @profile
    def __getitem__(self, idx):
        image_name, text_name = super().__getitem__(idx)
        text = self._read_file_from_s3(text_name)
        img = self._read_file_from_s3(image_name)
        return img, text


class MappedDataset(IterableDataset):

    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        return self.dataset[idx]


    def __iter__(self):
        """Process chunks from dataset using processor and yielding samples by one"""

        if not self.processor:
            raise ValueError("You can't iterate over dataset while processor is not specified")

        for images_batch, text_batch in self.dataset:
            # Process the whole chunk at once, but return samples by one
            pixel_values = self.processor(images_batch, return_tensors='pt').pixel_values
            tokens = self.processor.tokenizer(text_batch, padding='max_length', truncation=True, max_length=100)
            for image, text in zip(pixel_values, tokens['input_ids']):
                yield {
                    'pixel_values': image,
                    'labels': text,
                }


class DatasetFactory:

    @classmethod
    def create_dataset(self, dataset_type):
        classes = {'local': TrOCRDatasetLocal, 'S3': TrOCRDatasetS3}
        return classes[dataset_type]
    

class DataProcessor:

    def split(self, dataset_size, train_frac, val_frac):
        random.seed(0)
        all_indeces = list(range(dataset_size))
        random.shuffle(all_indeces)

        self.train_dataset_size = int(dataset_size * train_frac)
        self.val_dataset_size = int(dataset_size * val_frac)
        self.train_indeces = all_indeces[:self.train_dataset_size]
        self.val_indeces = all_indeces[self.train_dataset_size:self.train_dataset_size + self.val_dataset_size]
        self.test_indeces = all_indeces[self.train_dataset_size + self.val_dataset_size:]
    

class TrOCRDataProcessor(DataProcessor):

    def __init__(self, trocr_processor=None):
        self.processor = trocr_processor
        

    def split(self, train_frac, val_frac, dataset_type, **dataset_params):

        dataset_class = DatasetFactory().create_dataset(dataset_type)

        dataset = dataset_class(**dataset_params)
        dataset_size = len(dataset)

        super().split(dataset_size, train_frac, val_frac)

        # Make iterable datasets, which are iterated over chunks of data
        train_dataset = dataset_class(indeces=self.train_indeces, **dataset_params)
        val_dataset = dataset_class(indeces=self.val_indeces, **dataset_params)
        test_dataset = dataset_class(indeces=self.test_indeces, **dataset_params)

        return train_dataset, val_dataset, test_dataset, self.train_dataset_size


    def __call__(
            self,
            dataset_type,
            train_frac=0.85,
            val_frac=0.1,
            **dataset_params,
    ):

        train_dataset, val_dataset, test_dataset, train_dataset_size = self.split(
            train_frac,
            val_frac,
            dataset_type,
            **dataset_params,
        )

        # Process the data, returned by dataset, and output data samples by one
        train_dataset = MappedDataset(train_dataset, self.processor)
        val_dataset = MappedDataset(val_dataset, self.processor)
        test_dataset = MappedDataset(test_dataset, self.processor)
        
        return train_dataset, val_dataset, test_dataset, train_dataset_size


class YOLODataProcessor(DataProcessor):

    def __init__(self, dataset_name):
        self.data_dir = dataset_name
        self.images_path = os.path.join(self.data_dir, 'images')
        self.labels_path = os.path.join(self.data_dir, 'labels')


    def split(self, train_frac=0.85, val_frac=0.1):
        dataset_size = len(os.listdir(self.images_path))
        super().split(dataset_size, train_frac, val_frac)
    

    def restructure_dataset(self):
        """Copy dataset and organize it with train, val and test subfolders"""
        data_dir_abs = os.path.abspath(self.data_dir)
        data_dir_new = f'{data_dir_abs}_yolo'
        subfolders = ['images/train', 'images/val', 'images/test', 'labels/train', 'labels/val', 'labels/test']

        # Create directory structure
        for subfolder in subfolders:
            os.makedirs(os.path.join(data_dir_new, subfolder), exist_ok=True)

        for split, indeces in zip(
            ('train', 'val', 'test'),
            (self.train_indeces, self.val_indeces, self.test_indeces),
        ):
            # Get list of images for current split
            img_lst = os.listdir(self.images_path)
            img_lst = sorted([img_lst[i] for i in indeces])

            # Get list of labels for current split
            label_lst = os.listdir(self.labels_path)
            label_lst = sorted([label_lst[i] for i in indeces])

            for img, label in zip(img_lst, label_lst):
                # Get abspath for image and label location and destination
                img_path_old = os.path.abspath(os.path.join(self.images_path, img))
                img_path_new = os.path.abspath(os.path.join(data_dir_new, 'images', split))
                label_path_old = os.path.abspath(os.path.join(self.labels_path, label))
                label_path_new = os.path.abspath(os.path.join(data_dir_new, 'labels', split))

                # Copy files
                shutil.copy(img_path_old, img_path_new)
                shutil.copy(label_path_old, label_path_new)
        return data_dir_new


    def generate_yolo_config(self, dataset_path, config_path):
        """Generate yaml file with dataset configuration"""
        if hasattr(self, 'train_indeces'):
            config = {
                'path': dataset_path,
                'train': 'images/train',
                'val': 'images/val',
                'test': 'images/test',
                'nc': 1,
                'names': ['text'],
            }
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
        else:
            raise Exception('There are no splits yet. Use "split" first')


    def __call__(self, config_path, train_frac=0.85, val_frac=0.1, restructure=False):
        self.split(train_frac, val_frac)
        data_dir = self.restructure_dataset() if restructure else self.data_dir
        self.generate_yolo_config(data_dir, config_path)
