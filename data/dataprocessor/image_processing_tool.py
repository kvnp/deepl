import cv2
import numpy as np
import os
import shutil
import csv

crop_x = 256
crop_y = 224

label_dict = {"ruins": 0,
              "waterfall": 1,
              "desert": 2,
              "village": 3,
              "woods": 4,
              "skyisland": 5
            }

def main():
    src = "source_images\Zelda_Desert.png"
    img_crops = random_cutout(src, 100)
    save_img("desert", img_crops)
    create_labels("results")

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
    dest = ".\\results\\"+dest_name+"_"

    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    counter = 0
    for img in img_list:
        img_name = dest+str(counter)+".png"
        cv2.imwrite(img_name, img)

        counter += 1

def create_labels(src_dir):
    annot_list = list()
    
    for file in os.listdir(src_dir):
        filename = os.fsdecode(file)
        label = filename.split('.')[0]
        label = label.split('_')[0]
        label = label_dict[label]
        annot_list.append([filename,label])
    annot_list = np.array(annot_list)
    
    # die annotationsdatei
    csv_name = os.path.join(src_dir, "labels.csv")    

    np.savetxt(csv_name,annot_list, delimiter=';',fmt= '% s')
        
if __name__ == "__main__":
    main()
