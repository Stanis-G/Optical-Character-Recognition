import json
import logging
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(project_root)

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.utils import CER_SCORE

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


def plot_history(epochs, history, run_name=None, figsize=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize)

    for metric_name, metric_values in history.items():
        ax.plot(epochs, metric_values, label=metric_name)
    
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric')

    if run_name:
        os.makedirs(os.path.join("models", run_name), exist_ok=True)
        model_path = os.path.join("models", run_name, "history_plot.png")
        fig.savefig(model_path)

    return fig


# ----------------------------------------------------------------------------------------
# TrOCR functions
# ----------------------------------------------------------------------------------------


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


def preprocess_logits_for_metrics(logits, labels):
    output_ids = torch.argmax(logits[0], dim=-1)
    return output_ids, labels


def get_compute_metrics(processor):
    def compute_metrics(eval_pred):
        output_ids, labels_ids = eval_pred
        words_predicted = processor.tokenizer.batch_decode(output_ids[0], skip_special_tokens=False)
        words_labels = processor.tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
        return {'cer': CER_SCORE.compute(predictions=words_predicted, references=words_labels)}
    return compute_metrics

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
