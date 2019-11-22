import numpy as np
import json
from cv2 import imread
import sys
import os

# reference https://blog.csdn.net/weixin_41765699/article/details/100118660
def main():
    if len(sys.argv)<3:
        print("Usage: python %s [annotation json file] [images folder]" % sys.argv[0])
        return
    ann_json = sys.argv[1]
    img_folder = sys.argv[2]

    coco_data = json.load(open(ann_json, "r"))

    R_channel = 0
    G_channel = 0
    B_channel = 0

    num = 0
    count = 0
    for image in coco_data["images"]:
        filename = os.path.join(img_folder, image["file_name"])
        img = imread(filename) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])
        num = num + np.shape(img)[0]*np.shape(img)[1]
        count = count + 1
        print("Progress: ", count, "/", len(coco_data["images"]))
 
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num
 
    R_channel = 0
    G_channel = 0
    B_channel = 0
    count = 0
    for image in coco_data["images"]:
        filename = os.path.join(img_folder, image["file_name"])
        img = imread(filename) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)
        count = count + 1
        print("Progress: ", count, "/", len(coco_data["images"]))
    
    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
    print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))

if __name__ == "__main__":
    main()

