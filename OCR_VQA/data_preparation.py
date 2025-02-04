from datasets import load_dataset

class VQAProcessor:

    def __init__(self, processor):
        self.processor = processor


    def load_data(self):
        dataset = load_dataset(
            "howard-hou/OCR-VQA", # https://huggingface.co/datasets/howard-hou/OCR-VQA
            streaming=True, # load data by small chunks (default 1000 samples in a chunk)
        )
        # 166k + 20.7k + 20.8k = 207.549 samples
        train_size = 166000
        return dataset['train'], dataset['validation'], dataset['test'], train_size


    def process_data(self, examples):

        # Preprocess input images
        input_imgs = examples['image']
        input_imgs = [img.convert('RGB') for img in input_imgs]
        pixel_values = self.processor(input_imgs, return_tensors='pt').pixel_values

        # Tokenize words in a batch
        label_words = []
        for i in examples['ocr_info']:
            label_words.append(' '.join([j['word'] for j in i]))
        tokens = self.processor.tokenizer(label_words, padding='max_length', truncation=True, max_length=100)
        return {
            'pixel_values': pixel_values,
            'labels': tokens['input_ids'],
        }


    def __call__(
            self,
            dataset_batch_size=1000, # size of chunks loaded from IterableDataset
    ):
        train_dataset, val_dataset, test_dataset, train_size = self.load_data()

        remove_columns = [i for i in train_dataset.column_names if i not in ('pixel_values', 'labels')]
        train_dataset = train_dataset.map(self.process_data, batched=True, batch_size=dataset_batch_size, remove_columns=remove_columns)
        val_dataset = val_dataset.map(self.process_data, batched=True, batch_size=dataset_batch_size, remove_columns=remove_columns)
        test_dataset = test_dataset.map(self.process_data, batched=True, batch_size=dataset_batch_size, remove_columns=remove_columns)
        
        return train_dataset, val_dataset, test_dataset, train_size
        

