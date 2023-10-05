from Preprocessing.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
from Preprocessing.tissue_detection.tissue_detector import TissueDetector  # import dependent packages
from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # ERROR 수준 이상의 로그만 출력
import argparse
import glob

parser = argparse.ArgumentParser('argument for patch extracting')
parser.add_argument('--data_root', type=str, default="./Dataset/train/")
parser.add_argument('--num_processors', type=int, default=15)

def delete_directory(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(path)

def unfinished_filter(prefix, data_list):
    filterd_list = []
    for path in data_list:

        data_name = path.split('/')[-1][:-5]
        if data_name in 'CODIPAI-THCB-0450.tiff': #손상파일
            continue

        output_folder_dir = f"{prefix}/{data_name}"
        txt_file_dir = f"{prefix}/{data_name}.tiff_case_finished.txt"
        log_file_dir = f"{prefix.replace('/imgs/','/Logs/')}/{data_name}_extraction_grid_1008.png"

        patch_num, file_count = None, None
        # 중복 검사
        ## Txt 파일이 존재하면 생성완료 되었다고 적혀있는 패치 수를 읽어옴
        if os.path.isfile(txt_file_dir):
            with open(txt_file_dir, 'r') as file:
                content = file.read()
                try:
                    patch_num = int(content.split("Patch Num: ")[1].split()[0])
                except ValueError:
                    patch_num = None

        ## 출력 폴더가 존재하면 내부에 생성된 패치 수를 읽어옴
        if os.path.isdir(output_folder_dir):
            try:
                file_count = len(
                    [f for f in os.listdir(output_folder_dir) if os.path.isfile(os.path.join(output_folder_dir, f))])
            except:
                file_count = None

        ##둘 중 하나라도 불완전하다면, 삭제하고 생성 리스트에 append
        if patch_num is None or file_count is None:
            if os.path.exists(output_folder_dir):
                delete_directory(output_folder_dir)
            if os.path.isfile(txt_file_dir):
                os.remove(txt_file_dir)  # 파일 삭제
            if os.path.isfile(log_file_dir):
                os.remove(log_file_dir)  # 파일 삭제
                os.remove(log_file_dir.repalce('_grid_', '_grid_filtered_'))  # 파일 삭제
            filterd_list.append(path)

    return filterd_list

def _run(wsi_dir, output_dir, log_dir):
    tissue_detector = TissueDetector("LAB_Threshold", threshold=85)
    parameters = ExtractorParameters(save_dir=output_dir,  # Where the patches should be extracted to
                                     log_dir=log_dir,
                                     save_format='.png',  # Can be '.jpg', '.png', or '.tfrecord'
                                     patch_size=512,
                                     stride=512,
                                     sample_cnt=-1,
                                     rescale_rate=32,
                                     patch_filter_by_area=0.3,  # Amount of tissue that should be present in a patch
                                     with_anno=False,  # If true, you need to supply an additional XML file
                                     extract_layer=0)  # OpenSlide Level

    patch_extractor = PatchExtractor(tissue_detector, parameters=parameters, feature_map=None, annotations=None)
    patch_num = patch_extractor.extract(wsi_dir)
    print("%d Patches have been save to %s" % (patch_num, output_dir))

if __name__ == '__main__':
    args = parser.parse_args()

    for label in ['Lymph_node_metasis_present', 'Lymph_node_metasis_absent']:
        data_root_dir = f"{args.data_root}/raw_data/{label}/*.tiff"
        output_dir = f"{args.data_root}/patch_data/imgs/{label}/"
        log_dir = f"{args.data_root}/patch_data/logs/{label}/"
        data_list = unfinished_filter(prefix=output_dir, data_list=glob.glob(data_root_dir))
        if len(data_list) == 0:
            print(f'[{label}] is already completed')
            continue
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        results = Parallel(n_jobs=args.num_processors)(delayed(_run)(wsi_dir, output_dir, log_dir) for wsi_dir in tqdm(data_list))