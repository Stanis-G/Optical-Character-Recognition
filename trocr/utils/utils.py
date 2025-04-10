from PIL import Image
import evaluate


# CER (Character Error Rate) metric = (S + D + I) / N = (S + D + I) / (S + D + C)
# S - num of substitutions (means how much characters in a word were replaced)
# D - num of deletions
# I - num of insertions
# C - num of correct characters
# N - num of characters in the reference
CER_SCORE = evaluate.load(
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
