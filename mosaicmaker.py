import os
from trackdata import TrackData
from PIL import Image

from typing import List

def create(tracks: List[TrackData], images_path: str, out_path: str):
    images = []
    for track in tracks:
        image_path = f"{images_path}{track.name}.png"
        if os.path.exists(image_path):
            images.append(Image.open(image_path))

    # Assuming all images are the same size
    img_width, img_height = images[0].size
    mosaic_width = img_width * 4
    mosaic_height = img_height * 4

    mosaic_image = Image.new('RGB', (mosaic_width, mosaic_height))

    for i, img in enumerate(images):
        x = (i % 4) * img_width
        y = (i // 4) * img_height
        mosaic_image.paste(img, (x, y))

    mosaic_image.save(out_path)