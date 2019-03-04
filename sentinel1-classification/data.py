from __future__ import print_function
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans


def trainGenerator(train_path,num_image = 100,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = np.load(train_path + 'image/' + str(i) + '.npy')
        mask = np.load(os.path.join(train_path + 'label/',"%d.npy"%i))
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        mask = np.reshape(mask,mask.shape+(1,)) if (not flag_multi_class) else mask
        img = np.reshape(img,(1,)+img.shape)
        mask = np.reshape(mask,(1,)+mask.shape+(1,))
        yield (img,mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = np.load(os.path.join(test_path + 'image/',"%d.npy"%i))
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        np.save(os.path.join(save_path,"%d_predict"%i), img)