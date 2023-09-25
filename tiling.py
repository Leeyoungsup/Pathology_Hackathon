import sys
import numpy as np
import slideio
import os
import warnings
import pandas as pd
from PIL import Image
from glob import glob
from tqdm.notebook import tqdm
import warnings
import math
# 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore")

svs_files_path = sys.argv[1]
maske_files_path = sys.argv[2]
tile_path = sys.argv[3]
csv_path = sys.argv[4]

svs_files = glob(svs_files_path)

maske_files = glob(maske_files_path)

csv = pd.read_csv(csv_path, encoding='cp949')


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


slide_tile_size = 2048

for i in tqdm(range(len(svs_files))):
    count = 0
    fileName = os.path.basename(os.path.splitext(svs_files[i])[0])
    slide = slideio.open_slide(svs_files[i], "GDAL")
    scene = slide.get_scene(0)
    svsWidth = scene.rect[2]
    svsHeight = scene.rect[3]
    mask_file = [s for s in maske_files if fileName in s]
    mask_image = np.array(Image.open(mask_file[0]))
    ratio_index = os.path.basename(os.path.splitext(mask_file[0])[0]).find('_')
    ratio = float(os.path.basename(
        os.path.splitext(mask_file[0])[0])[ratio_index+1:])
    inverse_ratio = math.floor(1/ratio*10000)/10000
    data_frame = csv[csv['데이터톤번호'].str.contains(fileName)]
    data_frame = data_frame.reset_index()
    createDirectory(tile_path+data_frame.loc[0]['폴더']+'/5x/'+fileName)
    createDirectory(tile_path+data_frame.loc[0]['폴더']+'/10x/'+fileName)
    createDirectory(tile_path+data_frame.loc[0]['폴더']+'/20x/'+fileName)
    for widthCount in range(0, int(svsWidth // slide_tile_size)):
        for heightCount in range(0, int(svsHeight // slide_tile_size)):
            point_x = np.linspace(widthCount*slide_tile_size, widthCount *
                                  slide_tile_size+slide_tile_size-1, slide_tile_size, dtype=np.int32)
            point_y = np.linspace(heightCount*slide_tile_size, heightCount *
                                  slide_tile_size+slide_tile_size-1, slide_tile_size, dtype=np.int32)
            point = np.meshgrid(point_x, point_y)
            mask_point = np.copy(point)
            mask_point[0] = (mask_point[0]*inverse_ratio).astype(np.int64)
            mask_point[1] = (mask_point[1]*inverse_ratio).astype(np.int64)
            if mask_point[0].max() == mask_image.shape[1]:
                mask_point[0] -= 1
            if mask_point[1].max() == mask_image.shape[0]:
                mask_point[1] -= 1
            try:
                tile_mask_image = mask_image[mask_point[1], mask_point[0]]/255
                if tile_mask_image.mean() >= 1/2:
                    count += 1
                    image = scene.read_block((widthCount * slide_tile_size, heightCount *
                                             slide_tile_size, slide_tile_size, slide_tile_size), size=(1024, 1024))
                    img = Image.fromarray(image).resize((256, 256))
                    img.save(
                        tile_path+data_frame.loc[0]['폴더']+'/5x/'+fileName+'/'+fileName+'_'+str(count)+'.jpg')
                    img_count = 0
                    k = np.random.randint(2, size=1)[0]
                    j = np.random.randint(2, size=1)[0]
                    img = image[512*k:512*k+512, 512*j:512*j+512]
                    while_count = 0
                    while len(np.where(img[:, :, 1] >= 220)[0]) < (512*512)/2:
                        k = np.random.randint(2, size=1)[0]
                        j = np.random.randint(2, size=1)[0]
                        img = image[512*k:512*k+512, 512*j:512*j+512]
                        while_count += 1
                        if while_count == 10:
                            break
                    Image.fromarray(img).resize((256, 256)).save(
                        tile_path+data_frame.loc[0]['폴더']+'/10x/'+fileName+'/'+fileName+'_'+str(count)+'.jpg')
                    k = np.random.randint(4, size=1)[0]
                    j = np.random.randint(4, size=1)[0]
                    img = image[256*k:256*k+256, 256*j:256*j+256]
                    while_count = 0
                    while len(np.where(img[:, :, 1] >= 220)[0]) < (256*256)/2:
                        k = np.random.randint(2, size=1)[0]
                        j = np.random.randint(2, size=1)[0]
                        img = image[256*k:256*k+256, 256*j:256*j+256]
                        while_count += 1
                        if while_count == 10:
                            break
                    Image.fromarray(img).resize((256, 256)).save(
                        tile_path+data_frame.loc[0]['폴더']+'/20x/'+fileName+'/'+fileName+'_'+str(count)+'.jpg')
                    img_count += 1
            except:
                print(fileName+'_'+str(count))
