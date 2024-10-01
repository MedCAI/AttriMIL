import openslide
import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
from joblib import Parallel, delayed


def find_nearest(input_path,
                 output_path, 
                 patch_size=(256, 256)):
    print("Loading:", name)
    h5 = h5py.File(input_path)
    coords = np.array(h5['coords'])
    # features = np.array(h5['features'])
    h5.close()
    
    nearest = []
    # left, right, up, down, left_up, left_down, right_up, right_down
    for step, p in enumerate(coords):
        exists = [np.array(step)]
        left = np.array([p[0], p[1] - patch_size[1]])
        loc = np.where(np.sum(coords == left, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        right = np.array([p[0], p[1] + patch_size[1]])
        loc = np.where(np.sum(coords == right, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        up = np.array([p[0] - patch_size[0], p[1]])
        loc = np.where(np.sum(coords == up, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        down = np.array([p[0] + patch_size[0], p[1]])
        loc = np.where(np.sum(coords == down, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        left_up = np.array([p[0] - patch_size[0], p[1] - patch_size[1]])
        loc = np.where(np.sum(coords == left_up, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        left_down = np.array([p[0] + patch_size[0], p[1] - patch_size[1]])
        loc = np.where(np.sum(coords == left_down, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        right_up = np.array([p[0] - patch_size[0], p[1] + patch_size[1]])
        loc = np.where(np.sum(coords == right_up, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))

        right_down = np.array([p[0] + patch_size[0], p[1] + patch_size[1]])
        loc = np.where(np.sum(coords == right_down, axis=1) == 2)[0]
        if len(loc) != 0:
            exists.append(loc[0])
        else:
            exists.append(np.array(step))
        nearest.append(exists)
        
    h5 = h5py.File(output_path, 'w')  # 写入文件
    h5['coords'] = coords
    # h5['features'] = features
    h5['nearest'] = nearest # 名称为 image
    h5.close()  #关闭文件
    return


if __name__ == "__main__":
    # 需要改保存路径，和patch_size！！
    orgin_path = '/data2/clh/NSCLC/LUAD/20X/patches/'
    save_path = '/data2/clh/NSCLC/coords/'
    patch_size = (256, 256)
    start = 0
    name_list = []
    for step, name in enumerate(os.listdir(orgin_path)):
        if step < start:
            continue
        if name.endswith('h5'):
            name_list.append(name)
    Parallel(n_jobs=32)(delayed(find_nearest)(os.path.join(orgin_path, slide), os.path.join(save_path, slide)) for slide in name_list)