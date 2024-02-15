import os
from PIL import Image
from numpy import array
import shutil
import random

dir = "C:/DeepL/"
extension = 'Zelda3LightOverworld.png'
result_dir = "C:/DeepL/additional_pics"
files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(extension)]

if os.path.exists(result_dir):
    shutil.rmtree(result_dir, ignore_errors=True)

os.makedirs(result_dir)
files.sort()
for file in files:
    size = 256  # Size of the chunk
    num_chunks = 50  # Number of random chunks you want to extract
    path = os.path.join(dir, file)
    im = array(Image.open(path).convert("RGB"))

    # Calculate how many chunks can fit in each dimension
    max_x = im.shape[0] // size
    max_y = im.shape[1] // size

    result_file_dir = os.path.join(result_dir, file).split(".")[0]
    os.makedirs(result_file_dir)

    used_starts = set()
    while len(used_starts) < num_chunks:
        # Randomly choose a start point for the chunk
        start_x = random.randint(0, max_x - 1) * size
        start_y = random.randint(0, max_y - 1) * size
        start_point = (start_x, start_y)

        if start_point in used_starts:
            continue  # Skip if this start point has already been used
        used_starts.add(start_point)

        tile = im[start_x:start_x + size, start_y:start_y + size]
        result_image = Image.fromarray(tile)
        result_path = os.path.join(result_file_dir, f"{len(used_starts) - 1}.png")
        result_image.save(result_path)
