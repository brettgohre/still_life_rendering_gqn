from PIL import Image
import os

path = "."
directory = os.listdir(path)

def resize_and_crop():
    new_path = os.path.join(path, 'fruit_stills_small')
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".jpg")]
    for file in files:
        if os.path.isfile(os.path.join(path, file)):
            image = Image.open(os.path.join(path, file))
            image_resize = image.resize((96, 72), Image.ANTIALIAS) # (4032, 3024) --> (96,72)
            image_resize_crop = image_resize.crop((16, 4, 80, 68)) # (96, 72) --> (64, 64)
            image_resize_crop.save(os.path.join(new_path, os.path.basename(file)), 'jpeg', quality=100)

resize_and_crop()
