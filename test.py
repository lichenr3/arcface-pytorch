# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
from config import Config
from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list

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
# def load_image(img_path):
#     # 检查文件是否存在
#     if not os.path.exists(img_path):
#         print(f"File does not exist: {img_path}")
#         return None
#
#     # 尝试读取图像
#     image = cv2.imread(img_path)
#     if image is None:
#         # 尝试不同的读取方式
#         image = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         if image is None:
#             print(f"Failed to read image: {img_path}")
#             return None
#
#     # 确保图像是彩色
#     if len(image.shape) < 3:
#         image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#
#     # 转换为RGB并调整大小
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (128, 128))
#     image = image.astype(np.float32)
#     image -= 127.5
#     image /= 127.5
#     image = np.expand_dims(image, axis=0)
#     return image
# python3 test.py

def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    print(f"Total images to process: {len(test_list)}")
    for i, img_path in enumerate(test_list):
        if i % 100 == 0:  # 每100张打印一次进度
            print(f"Processing image {i+1}/{len(test_list)}")
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cpu"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt
# def get_featurs(model, test_list, batch_size=10):
#     all_features = []
#     cnt = 0
#     batch_images = []
#     success_count = 0
#     error_count = 0
#
#     print(f"Total images to process: {len(test_list)}")
#
#     for i, img_path in enumerate(test_list):
#         if i % 100 == 0:  # 每100张打印一次进度
#             print(f"Processing image {i+1}/{len(test_list)}")
#
#         image = load_image(img_path)
#         if image is None:
#             print(f'read {img_path} error')
#             error_count += 1
#             continue
#
#         success_count += 1
#         batch_images.append(image)
#
#         if len(batch_images) == batch_size or i == len(test_list)-1:
#             cnt += 1
#             data = np.stack(batch_images, axis=0)
#             data = torch.from_numpy(data)
#             data = data.to(torch.device("cpu"))
#             output = model(data)
#             output = output.data.cpu().numpy()
#
#             all_features.append(output)
#             batch_images = []
#
#     features = np.vstack(all_features) if len(all_features) > 0 else np.array([])
#     print(f"Successfully processed {success_count} images, failed {error_count}")
#     return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    # 只处理成功读取的特征
    min_length = min(len(test_list), len(features))
    print(f"Creating feature dict for {min_length} items")
    for i in range(min_length):
        fe_dict[test_list[i]] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        try:
            fe_1 = fe_dict[splits[0]]
            fe_2 = fe_dict[splits[1]]
            # ... 计算相似度 ...
        except KeyError as e:
            print(f"跳过缺失的图像: {e}")
            continue  # 继续处理下一对
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':
    opt = Config()

    # 检查LFW根目录是否存在
    if not os.path.exists(opt.lfw_root):
        print(f"Error: LFW root directory does not exist: {opt.lfw_root}")
        exit(1)

    # 检查测试列表文件是否存在
    if not os.path.exists(opt.lfw_test_list):
        print(f"Error: Test list file does not exist: {opt.lfw_test_list}")
        exit(1)

    # 打印路径信息
    print(f"LFW root: {opt.lfw_root}")
    print(f"Test list: {opt.lfw_test_list}")

    # 使用项目自定义模型
    if opt.backbone == 'resnet18':
        model = resnet_face18(opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()
    else:
        raise ValueError(f"Unsupported backbone: {opt.backbone}")

    print(f"Using {opt.backbone} model")
    model = DataParallel(model)
    model.load_state_dict(torch.load(opt.test_model_path, map_location=torch.device('cpu')))
    model.to(torch.device("cpu"))
    model.eval()

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    # 测试第一个图像路径
    test_path = img_paths[0] if len(img_paths) > 0 else ""
    print(f"Testing first image path: {test_path}")
    if os.path.exists(test_path):
        print("First image exists")
    else:
        print("First image does not exist")

    lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
