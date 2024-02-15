import cv2
import numpy as np
import os
import shutil
import csv
import json

crop_x = 256
crop_y = 224

# example json
# {
#     "ruins": 150,
#     "waterfall": 300,
#     "desert": 150,
#     "village": 200,
#     "woods": 300,
#     "skyisland": 100,
#     "mountains": 400,
#     "central": 200
# }

label_dict = {"ruins": 0,
              "waterfall": 1,
              "desert": 2,
              "village": 3,
              "woods": 4,
              "skyisland": 5,
              "mountains": 6,
              "central" : 7
            }

def create_dataset(config_path, dest_dir):
    image_data_location = "utils\\dataprocessor\\source_images\\class_image_data.json"
    clss_img_data = json.load(open(image_data_location)) 
    
    configs = json.load(open(config_path))
    for img_label in configs.keys():
        for img_data in clss_img_data[img_label]:
            img_path = img_data["path"]
            amount = configs[img_label] * img_data["portion"]
            _create_random_cutout(label_name=img_label, 
                                 destination=dest_dir, 
                                 img_path= img_path, 
                                 amount=amount)
    _create_labels(dest_dir)
        
        

def _create_random_cutout(label_name, destination, img_path, amount):
    src = cv2.imread(img_path)
    
    max_x = src.shape[1] - crop_x
    max_y = src.shape[0] - crop_y

    crops = list()

    for i in range(int(amount)):
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crops.append(src[y: y + crop_y, x: x + crop_x])

    _save_img(label_name, destination, crops)

def _save_img(label_name, destination, img_list):
    dest = destination+"\\"+label_name+"_"

    if os.path.exists(dest):
        shutil.rmtree(dest, ignore_errors=True)

    counter = 0
    for img in img_list:
        img_name = dest+str(counter)+".png"
        cv2.imwrite(img_name, img)

        counter += 1

def _create_labels(src_dir):
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

    np.savetxt(csv_name,annot_list, delimiter=',',fmt= '% s')
        