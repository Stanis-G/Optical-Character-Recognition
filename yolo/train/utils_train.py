from PIL import ImageDraw


def draw_yolo_bbox(image, yolo_bbox, outline_color="red", width=2):

    # Parse the string into floats
    parts = list(map(float, yolo_bbox.strip().split()))
    
    _, x_center, y_center, w, h = parts
    img_w, img_h = image.size

    # Convert relative coords to absolute pixel coords
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x0 = int(x_center - w / 2)
    y0 = int(y_center - h / 2)
    x1 = int(x_center + w / 2)
    y1 = int(y_center + h / 2)

    # Draw the rectangle
    image_with_box = image.copy()
    draw = ImageDraw.Draw(image_with_box)
    draw.rectangle([x0, y0, x1, y1], outline=outline_color, width=width)
    
    return image_with_box
