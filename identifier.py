import pickle
import torch
import numpy as np
from torch.nn import DataParallel

from models import resnet_face18
from config import Config
import cv2
import os


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image

def get_feature(model, image):
    data = torch.from_numpy(image).to(torch.device("cpu"))
    with torch.no_grad():
        output = model(data).data.cpu().numpy()
        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))
    return feature  # 返回一维特征

def cos_sim(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def identify(img_path, top_k=3):
    # 加载配置和模型
    opt = Config()
    model = resnet_face18(opt.use_se)
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path, map_location=torch.device("cpu")))
    model.to(torch.device("cpu"))
    model.eval()

    # 加载特征库
    with open("face_feature_dict.pkl", "rb") as f:
        face_db = pickle.load(f)

    # 加载输入图像
    image = load_image(img_path)
    if image is None:
        return None

    feat = get_feature(model, image)

    # 计算相似度
    results = {}
    for path, db_feat in face_db.items():
        person_name = path.split("/")[0]
        sim = cos_sim(feat, db_feat)
        if person_name not in results:
            results[person_name] = 0
        if sim > results[person_name] and sim >= 0.255:
            results[person_name] = sim

    sorted_persons = sorted(results.items(), key=lambda x: x[1], reverse=True)

    if sorted_persons:
        print(f"Matched {len(sorted_persons)} results above threshold {0.255:.3f}")
        return sorted_persons[:top_k]  # 返回所有超过阈值的
    else:
        print(f"No result above threshold, returning top-{top_k}")
        return sorted_persons[:top_k]  # 返回最相似的几个


# 示例用法
if __name__ == "__main__":
    result = identify("Abel_Pacheco_0001_副本.jpg")
    for name, sim in result:
        print(f"{name}: similarity={sim.item():.4f}")

