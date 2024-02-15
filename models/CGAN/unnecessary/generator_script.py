import cv2
import numpy as np


def look_at_samplebatch(img_batch, use_cuda= False):
    counter = 0
    if use_cuda:
        img_batch = img_batch.cpu()
    for test in img_batch:
        test = test.detach().numpy()
        print("########Bild", counter, ": ", test.shape)
        # print("imdata: ", test)
        cv2_image = np.transpose(test, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("bilder", cv2_image)
        cv2.waitKey(0)
    quit()

