import cv2
import numpy as np
import os
import shutil

crop_x = 256
crop_y = 256


def main():
    # src = "../Zelda3LightOverworld.png"
    src = "../source_images/Zelda_Central_Hyrule.png"
    img_crops = random_cutout(src, 100)
    save_img("cave_exit", img_crops)


def random_cutout(img_path, amount):
    src = cv2.imread(img_path)
    
    max_x = src.shape[1] - crop_x
    max_y = src.shape[0] - crop_y

    crops = list()

    for i in range(amount):
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crops.append(src[y: y + crop_y, x: x + crop_x])

    return crops


def save_img(dest_name, img_list):
    dest = ".\\results\\"+dest_name+"\\"

    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    os.makedirs(dest)
    
    counter = 0
    for img in img_list:
        img_name = dest+str(counter)+"_gen2.png"
        cv2.imwrite(img_name, img)

        counter += 1


    

    

        



if __name__ == "__main__":
    main()
