from PIL import Image


def inference(img, model, processor, max_new_tokens=100):
    
    if type(img) == str:
        img = Image.open(img).convert('RGB')
    model = model.eval()

    pixel_values = processor(img, return_tensors='pt').pixel_values.to(device=model.device)
    generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return img, generated_text
