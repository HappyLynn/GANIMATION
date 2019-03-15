import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2


def load_data(face_img_path, aus_pkl_path):
    '''
    :param:
        face_img_path: folder path of face images
        aus_pkl_path: path of 'aus.pkl'
    :return:
        imgs: RGB face np.array, shape [n, 128, 128, 3]
        aus: Action Unit np.array, shape [n, 17]
    '''
    imgs_names = os.listdir(face_img_path)
    imgs_names.sort()
    with open(aus_pkl_path, 'rb') as f:
        aus_dict = pickle.load(f)
    imgs = np.zeros((len(imgs_names), 128, 128, 3), dtype=np.float32)
    aus = np.zeros((len(imgs_names), 17), dtype=np.float32)
    i = 0
    for img_name in tqdm(imgs_names):
        #break
        img = cv2.imread(os.path.join(face_img_path, img_name))[:, :, ::-1]  # BGR -> RGB
        #img = cv2.resize(img, (128, 128))
        img = img / 127.5 - 1  # rescale within [-1,1]
        #print(img_name)
        imgs[i] = img
        aus[i] = aus_dict[img_name.strip('.jpg')] / 5
        i += 1

    return imgs, aus
