
import os
import cv2

for folder in os.listdir('./data/masked'):
    print folder
    folder_path = os.path.join('./data/masked', folder)
    dst_path = os.path.join('./data/processed', folder)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, 0)
        
        if 'left' in folder:
            mod_image = cv2.flip(image, 1)
        else:
            mod_image = image
        mod_image = cv2.resize(mod_image, (300, 300), interpolation = cv2.INTER_AREA)
        dst_image_path = os.path.join(dst_path, image_name)
        cv2.imwrite(dst_image_path, mod_image)

