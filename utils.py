import json
import logging
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import evaluate

# ----------------------------------------------------------------------------------------
# Common functions
# ----------------------------------------------------------------------------------------

def plot_img(img, figsize=(9, 3)):
    """Plot image"""

    fig, axes = plt.subplots(1, 1, figsize=figsize)

    axes.imshow(img)
    axes.axis('off')

    plt.tight_layout()
    return fig


def plot_history(epochs, history, run_name, figsize=(15, 10)):
    plt.figure(figsize=figsize)
    legend = []
    for metric_name, metric_values in history.items():
        plt.plot(epochs, metric_values)
        legend.append(metric_name)
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    if run_name:
        model_path = os.path.join(*['models', run_name, 'history_plot.png'])
        plt.savefig(model_path)


# ----------------------------------------------------------------------------------------
# TrOCR functions
# ----------------------------------------------------------------------------------------


# CER (Character Error Rate) metric = (S + D + I) / N = (S + D + I) / (S + D + C)
# S - num of substitutions (means how much characters in a word were replaced)
# D - num of deletions
# I - num of insertions
# C - num of correct characters
# N - num of characters in the reference
cer_score = evaluate.load(
    "cer", # https://huggingface.co/spaces/evaluate-metric/cer
    module_type='metric', # "metric" stands for evaluating a model
)


def inference(img, model, processor, max_new_tokens=100):
    
    if type(img) == str:
        img = Image.open(img).convert('RGB')
    model = model.eval()

    pixel_values = processor(img, return_tensors='pt').pixel_values.to(device=model.device)
    generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return img, generated_text


def evaluate_model(
        model, processor, indeces,
        score_fn=cer_score, max_new_tokens=100,
        data_path='custom_dataset/data',
    ):
    """Generate text using image and compare it to the original text. Calc metric"""

    texts_lst, generated_texts_lst = [], []
    for sample_index in indeces:
        
        # Read data, generate text
        with open(f'{data_path}/texts/title_{sample_index}.txt', 'r') as f:
            text = f.read()
        _, generated_text = inference(f'{data_path}/images/image_{sample_index}.png', model, processor, max_new_tokens)
        if text:
            texts_lst.append(text)
            generated_texts_lst.append(generated_text)

    metric_value = score_fn.compute(predictions=generated_texts_lst, references=texts_lst)

    return texts_lst, generated_texts_lst, metric_value


def save_history(history, file_path):
    if not file_path.endswith('.json'):
        file_path = f'{file_path}.json'
    with open(file_path, 'w') as f:
        json.dump(history, f)


def save_model_and_history(run_name, trainer):
    model_path = os.path.join(*['models', run_name, 'model'])
    if os.path.exists(model_path):
        logging.warning(f"Run '{run_name}' already exists. Specify another name")
        return None
    trainer.save_model(model_path)

    history_path = ['models', run_name, 'history.json']
    save_history(trainer.state.log_history, os.path.join(*history_path))

# ----------------------------------------------------------------------------------------
# YOLO functions
# ----------------------------------------------------------------------------------------

def plot_boxes(sample, to_plot=False, fig_width=5):
    """Add label boxes to image. Works with 'howard-hou/OCR-VQA' dataset"""

    img = np.array(sample['image'])
    boxes = [i['bounding_box'] for i in sample['ocr_info']]

    img_height, img_width = sample['image_height'], sample['image_width']
    for box in boxes:
        top_left_x = box['top_left_x'] * img_width
        top_left_y = box['top_left_y'] * img_height
        box_width = box['width'] * img_width
        box_height = box['height'] * img_height
        upper_left_corner = (
            int(top_left_x),
            int(top_left_y),
        )
        lower_right_corner = (
            int(top_left_x + box_width),
            int(top_left_y + box_height),
        )
        img = cv2.rectangle(img, upper_left_corner, lower_right_corner, (0, 255, 0), 2)  # Green color, 2-pixel thickness
    
    if to_plot:
        aspect_ratio = img_height / img_width
        plot_img(img, figsize=(fig_width, int(fig_width * aspect_ratio)))
    
    return img
