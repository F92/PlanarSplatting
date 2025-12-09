import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_metric3d import extract_mono_geo_demo
from utils_demo.run_vggt import run_vggt
from utils_demo.misc import is_video_file, save_frames_from_video
from utils_demo.run_planarSplatting import run_planarSplatting
from npy import load_my_data   # ✅ 用你写的 load_my_data() 直接获取 data
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument("-d", "--data_path", type=str, default='examples/living_room/images', help='path of input data')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/demo', help='path of output dir')
    # parser.add_argument("-s", "--frame_step", type=int, default=10, help='sampling step of video frames')
    # parser.add_argument("--depth_conf", type=float, default=2.0, help='depth confidence threshold of vggt')
    parser.add_argument("--conf_path", type=str, default='utils_demo/demo.conf', help='path of configure file')
    # parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    args = parser.parse_args()

    # ========== 直接用你已有的 data ==========
    data = load_my_data()
    _, normal_maps_list = extract_mono_geo_demo(data['color'], data['intrinsics'])
    data['normal'] = normal_maps_list
    print("✅ 已加载本地 data")

    for i in range(len(data['intrinsics'])):
        data['intrinsics'][i] = data['intrinsics'][i].astype(np.float32)
    for i in range(len(data['extrinsics'])):
        data['extrinsics'][i] = data['extrinsics'][i].astype(np.float32)
    for i in range(len(data['depth'])):
        data['depth'][i] = data['depth'][i].astype(np.float32)
    for i in range(len(data['normal'])):
        data['normal'][i] = data['normal'][i].astype(np.float32)
    
    # ========== 输出目录 ==========
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # ========== 加载配置 ==========
    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)

    conf.put('train.exps_folder_name', out_path)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)

    # ========== 跑 PlanarSplatting ==========
    planar_rec = run_planarSplatting(data=data, conf=conf)
