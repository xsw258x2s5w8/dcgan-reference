# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from PIL import Image


def read_imgs(img_path):
    with Image.open(img_path) as img:
        arr_img = np.asarray(img, dtype='uint8')
        return arr_img

def write_binary(imgpaths,save_dir='data/t.tfrecord'):  #用于制造tfrecord文件
    writer=tf.python_io.TFRecordWriter(save_dir)   #
    
    for i,imgpath in enumerate(imgpaths):
        img=read_imgs(imgpath)
        img_raw=img.tostring()  #将图片数据转为二进制的数据
        example=tf.train.Example(
             features=tf.train.Features(
                  feature={
                        'img':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])) 
                  }  
             )      
        )
    #serialized
        serialized=example.SerializeToString()  #序列文件
    #writer
        writer.write(serialized)             #写入
    writer.close()


if __name__=='__main__':

   train_savedir='data/train.tfrecord' 
   
   data_dir='data/crop_images_DB/'
    
    #sample
   train_imgpaths=[os.path.join(data_dir,img_i) 
                for img_i in os.listdir(data_dir)]
   print('read train txt ok!1/6')
   write_binary(train_imgpaths,save_dir=train_savedir)
   print('write train.tfrecord ok 2/6')
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   