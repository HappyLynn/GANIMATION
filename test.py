import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import numpy as np
import cv2
from models import generator, discriminator
from tqdm import tqdm


real_img = tf.placeholder(tf.float32, [1, 128, 128, 3], name='real_img')
style_img = tf.placeholder(tf.float32, [1, 128, 128, 3], name='style_img')

_, desired_au = discriminator(style_img, reuse=False)
desired_au2 = tf.placeholder(tf.float32, [1, 17], name='feather')
fake_img, fake_mask = generator(real_img, desired_au2, reuse=False)
fake_img_masked = fake_mask * real_img + (1 - fake_mask) * fake_img

# imgs_names = os.listdir('jiaqi_face')
# real_src = face_recognition.load_image_file('obama.jpeg')  # RGB image
# face_loc = face_recognition.face_locations(real_src)
# if len(face_loc) == 1:
#     top, right, bottom, left = face_loc[0]

real_face = np.zeros((1,128,128,3), dtype=np.float32)
style_face = np.zeros((1,128,128,3), dtype=np.float32)
#real_face[0] = cv2.resize(real_src[top:bottom, left:right], (128,128)) / 127.5 - 1


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'weights/ganimation_epoch20.ckpt')
    total = []
    for i in range(9):
        save = []
        for img_name in tqdm(range(15)):
            style_face[0] = cv2.imread('/data/wangzhe/SematicSeg/GANimation-tf/tset_img/2.jpg')[:, :, ::-1] / 127.5 - 1
            real_face[0] = cv2.imread('/data/wangzhe/SematicSeg/GANimation-tf/tset_img/' + str(i + 1) + '.jpg')[:, :, ::-1] / 127.5 - 1
            expression = sess.run(desired_au, feed_dict={style_img: style_face}) * float(img_name) / 10.0
            output = sess.run(fake_img_masked, feed_dict={real_img: real_face, desired_au2: expression})
            real_src = (output[0] + 1) * 127.5
            save.append(real_src)
            cv2.imwrite(str(img_name) + '1.jpg', real_src[:, :, ::-1])
        save = np.concatenate(save, axis=1)
        total.append(save)
        cv2.imwrite('fina' + str(i) + '.jpg', save[:, :, ::-1])
    total = np.concatenate(total, axis=0)
    cv2.imwrite('final.jpg', total[:, :, ::-1])