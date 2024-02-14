import os
from PIL import Image
from numpy import array, uint8
import shutil

dir = "/home/kevin/Projects/ZeldaALinkToThePast/"
extension = 'BG.png'
result_dir = "/home/kevin/Projects/ZeldaALinkToThePast/results"
files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and f.endswith(extension)]

if os.path.exists(result_dir):
    shutil.rmtree(result_dir, ignore_errors=True)

os.makedirs(result_dir)
files.sort()
for file in files:
    size = 256
    path = os.path.join(dir, file)
    im = array(Image.open(path).convert("RGB"))
    tiles = [im[x:x+size, y:y+size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
    
    result_file_dir = os.path.join(result_dir, file).split(".")[0]
    os.makedirs(result_file_dir)

    for i in range(0, len(tiles)):
        tile = tiles[i]
        result_image = Image.fromarray(tile)
        result_path = os.path.join(result_file_dir, str(i) + ".png")
        result_image.save(result_path)