import glob
import shutil
from tqdm import tqdm
import os
if __name__ == '__main__':

    command = '/data/wangzhe/OpenFace/build/bin/FeatureExtraction '
    files = glob.glob('/data/wangzhe/SematicSeg/cityscapesScripts/imdb_imagess/*.jpg')
    files = files[377241:]
    for each_file in tqdm(files):
        command = 'nohup /data/wangzhe/OpenFace/build/bin/FeatureExtraction -aus '
        command += '-f \"' + each_file + "\" "
        command += '-out_dir /data/wangzhe/SematicSeg/cityscapesScripts/Aus_data -au_static'
        os.system(command)
    # #command += '-out_dir /data/wangzhe/SematicSeg/cityscapesScripts/Aus_data'
    # os.system(command)
    # files_name = '/data/wangzhe/SematicSeg/cityscapesScripts/imdb_crop/*/*jpg'
    # files = glob.glob(files_name)
    # for each_file in tqdm(files):
    #     shutil.copy(each_file, '/data/wangzhe/SematicSeg/cityscapesScripts/imdb_imagess/')