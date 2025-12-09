import os
import numpy as np
import cv2
import torch

def load_my_data():
    base_dir = "/media/ubiboy/02AA-0ED1/DATASET/dataset_myLab/planeset/scene_0003"
    rgb_dir = os.path.join(base_dir, "rgb")
    depth_dir = os.path.join(base_dir, "depth")

    # ========== 读取 RGB ==========
    color_images_list = []
    image_paths_list = []
    for i in range(20,21):
        fname = f"{i:04d}.png"
        fpath = os.path.join(rgb_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)[:, :, ::-1].copy()  # BGR→RGB
        color_images_list.append(img)          # numpy.uint8 (H,W,3)
        image_paths_list.append(fpath)

    # ========== 读取 Depth ==========
    depth_maps_list = []
    for i in range(20,21):
        fname = f"{i:04d}.png"
        fpath = os.path.join(depth_dir, fname)
        if not os.path.exists(fpath):
            continue
        depth = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if depth.max() > 50:  # 假设是 mm
            depth = depth / 1000.0
        depth_maps_list.append(depth)

    # ========== 读取相机位姿 ==========
    extrinsics_all = np.load(os.path.join(base_dir, "camera_poses.npy"))  # (N,4,4)
    c2ws_list = [extrinsics_all[i] for i in range(extrinsics_all.shape[0])]

    # ========== 读取相机内参 ==========
    intrinsics_all = np.load(os.path.join(base_dir, "camK.npy"))  # (3,3)
    intrinsics_list = [intrinsics_all for _ in range(len(color_images_list))]

    # ========== 打包 ==========
    data = {
        "color": color_images_list,       # list of np.uint8(H,W,3)
        "depth": depth_maps_list,         # list of np.float32(H,W)
        "image_paths": image_paths_list,  # list of str
        "extrinsics": c2ws_list,          # list of np.ndarray(4,4)
        "intrinsics": intrinsics_list,    # list of np.ndarray(3,3)
    }
    return data
def check_data_structure(data):
    """
    检查 data 是否符合 PlanarSplatting 需要的格式
    """

    assert isinstance(data, dict), "data 必须是 dict"

    required_keys = ["color", "depth", "image_paths", "extrinsics", "intrinsics"]
    for k in required_keys:
        assert k in data, f"缺少字段: {k}"

    N = len(data["color"])
    print(f"共 {N} 帧")

    # color
    assert isinstance(data["color"], list), "color 必须是 list"
    assert isinstance(data["color"][0], np.ndarray), "color 元素必须是 numpy array"
    assert data["color"][0].dtype == np.uint8, f"color dtype 应为 uint8, 现在是 {data['color'][0].dtype}"
    assert data["color"][0].ndim == 3 and data["color"][0].shape[2] == 3, "color 必须是 H×W×3"

    # depth
    assert isinstance(data["depth"], list), "depth 必须是 list"
    assert isinstance(data["depth"][0], np.ndarray), "depth 元素必须是 numpy array"
    assert data["depth"][0].dtype in [np.float32, np.float64], f"depth dtype 应为 float32/64, 现在是 {data['depth'][0].dtype}"
    assert data["depth"][0].ndim == 2, "depth 必须是 H×W"

    # image_paths
    assert isinstance(data["image_paths"], list), "image_paths 必须是 list"
    assert isinstance(data["image_paths"][0], str), "image_paths 元素必须是 str"

    # extrinsics
    assert isinstance(data["extrinsics"], list), "extrinsics 必须是 list"
    assert data["extrinsics"][0].shape == (4,4), f"extrinsics 每个矩阵应为 4×4, 现在是 {data['extrinsics'][0].shape}"

    # intrinsics
    assert isinstance(data["intrinsics"], list), "intrinsics 必须是 list"
    assert data["intrinsics"][0].shape == (3,3), f"intrinsics 每个矩阵应为 3×3, 现在是 {data['intrinsics'][0].shape}"

    print("✅ data 结构检查通过，和 run_vggt 输出一致！")

if __name__ == "__main__":
    data = load_my_data()
    check_data_structure(data)