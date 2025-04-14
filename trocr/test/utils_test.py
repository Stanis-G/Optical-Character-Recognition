from trocr.utils.utils import CER_SCORE, inference


def evaluate_model(
        model,
        dataset,
        score_fn=CER_SCORE,
        max_new_tokens=100,
    ):
    """Generate text using image and compare it to the original text. Calc metric"""

    texts_lst, generated_texts_lst = [], []
    for sample_index in dataset.dataset.indeces:
        
        # Read data, generate text
        img, text = dataset.dataset[sample_index]
        _, generated_text = inference(img, model, dataset.processor, max_new_tokens)

        texts_lst.append(text)
        generated_texts_lst.append(generated_text)

    metric_value = score_fn.compute(predictions=generated_texts_lst, references=texts_lst)

    return texts_lst, generated_texts_lst, metric_value
