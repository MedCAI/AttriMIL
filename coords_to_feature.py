import openslide
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py

if __name__ == "__main__":
    # 需要改保存路径，和patch_size！！
    orgin_path = '/data2/clh/NSCLC/resnet18_simclr/h5_files/'
    coord_path = '/data2/clh/NSCLC/coords/'
    save_path = '/data2/clh/NSCLC/resnet18_simclr/h5_coords_files/'
    patch_size = (256, 256)
    start = 0
    for step, name in enumerate(os.listdir(orgin_path)):
        if step < start:
            continue
        if name in os.listdir(save_path):
            print("exist:", name)
            continue
        
        if name.endswith('h5'):
            # 读取文件
            print("Loading:", name)
            h5 = h5py.File(orgin_path + name)
            coords = np.array(h5['coords'])
            features = np.array(h5['features'])
            h5.close()
            h5 = h5py.File(coord_path + name)
            nearest = np.array(h5['nearest'])
            h5.close()
            h5 = h5py.File(save_path + name, 'w')  #写入文件
            h5['coords'] = coords
            h5['features'] = features
            h5['nearest'] = nearest #名称为image
            h5.close()  #关闭文件
            print("coords:{}, features:{}, nearest:{}".format(coords.shape, features.shape, nearest.shape))
