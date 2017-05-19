#! /usr/bin/python
import os
from PIL import Image

def crop_img_by_half_center(src_file_path, dest_file_path):
    im = Image.open(src_file_path)
    x_size, y_size = im.size
    start_point_x = x_size / 4
    end_point_x   = x_size / 4 + x_size / 2
    start_point_y = y_size / 3
    end_point_y   = y_size / 3 + y_size / 2
    box = (start_point_x, start_point_y, end_point_x, end_point_y)
    new_im = im.crop(box)
    new_new_im = new_im.resize((64,64))
    new_new_im.save(dest_file_path)

def walk_through_the_folder_for_crop(aligned_db_folder, result_folder):
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    
    i = 0
    img_count = 0
    for img_i in os.listdir(aligned_db_folder):
        src_img_path=os.path.join(aligned_db_folder,img_i)
        dest_img_path=os.path.join(result_folder,img_i)
        crop_img_by_half_center(src_img_path, dest_img_path)
        img_count+=1
    print("%d is finish" %img_count)
        
if __name__ == '__main__':
    aligned_db_folder = "img_align_celeba"
    result_folder = "crop_images_DB"
    if not aligned_db_folder.endswith('/'):
        aligned_db_folder += '/'
    if not result_folder.endswith('/'):
        result_folder += '/'
    walk_through_the_folder_for_crop(aligned_db_folder, result_folder)
    

   # src_file_path='data/aligned_images_DB/Aaron_Eckhart/0/aligned_detect_0.593.jpg'