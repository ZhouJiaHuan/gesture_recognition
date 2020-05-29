# Description: model training
# Author: ZhouJH
# Data: 2020/4/8
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from mmcv import Config
import time
import argparse
import sys
sys.path.append(".")
from gesture_lib.utils import make_dirs
from gesture_lib.model import build_model
from gesture_lib.dataset import build_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_correct(predicts, labels, cls_names):
    correct_dict = {}

    for cls_id, cls_name in enumerate(cls_names):
        label_ids = labels == cls_id
        label_num = len(labels[label_ids])
        predict = predicts[label_ids]
        correct_num = len(predict[predict == cls_id])
        correct_dict[cls_name] = [label_num, correct_num]

    return correct_dict


def merge_batch_correct(dict1, dict2):
    dict3 = dict1.copy()
    for key, value in dict2.items():
        dict3.setdefault(key, [0, 0])
        dict3[key][0] += value[0]
        dict3[key][1] += value[1]

    return dict3


def train_pipeline(model, data_loader, loss_func, optimizer):
    model.train()

    for i, batch_data in enumerate(data_loader):
        keypoints = batch_data['keypoints'].to(device)
        labels = batch_data['label'].to(device)
        keypoints = torch.autograd.Variable(keypoints)
        labels = torch.autograd.Variable(labels)
        out = model(keypoints)
        loss = loss_func(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss


def test_pipeline(model, data_loader, cls_names):
    model.eval()
    correct_dict = {}

    for batch_data in tqdm(data_loader):
        keypoints = batch_data['keypoints'].to(device)
        labels = batch_data['label'].to(device)
        keypoints = torch.autograd.Variable(keypoints)
        labels = torch.autograd.Variable(labels)
        out = model(keypoints)
        _, predict = torch.max(out.data, 1)
        temp_correct = count_correct(predict, labels, cls_names)
        correct_dict = merge_batch_correct(correct_dict, temp_correct)

    return correct_dict


def parse_args():
    parser = argparse.ArgumentParser(description="gesture model training")
    parser.add_argument('config',
                        help='train config file path')
    parser.add_argument('--work_dir',
                        help='the dirs to save logs and models')
    parser.add_argument('--gpus', type=int, default=1,
                        help='gpu number, default: 1')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    print("\n------- configure info -------:")
    for key, value in dict(cfg).items():
        print("{}: {}".format(key, value))

    make_dirs(os.path.abspath(cfg.work_dir))
    lstm_model = build_model(cfg.model).to(device)
    print("\n------- model info -------:")
    print(lstm_model)

    data_cfg = cfg.dataset
    cls_names = data_cfg.train.cls_names
    train_dataset = build_dataset(data_cfg.train)
    train_loader = DataLoader(train_dataset, **data_cfg.train_loader)

    test_dataset = build_dataset(data_cfg.test)
    test_loader = DataLoader(test_dataset, **data_cfg.test_loader)

    work_dir = cfg.work_dir
    model_dir = os.path.join(work_dir, 'checkpoints')
    log_dir = os.path.join(work_dir, 'runs')
    make_dirs(model_dir)
    make_dirs(log_dir)

    lr_cfg = cfg.lr_config
    epochs = lr_cfg.epochs
    optimizer = torch.optim.Adam(lstm_model.parameters(),
                                 lr=lr_cfg.lr,
                                 weight_decay=lr_cfg.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, lr_cfg.step, 0.1)
    writer = SummaryWriter(log_dir=log_dir)
    print("\nstart training ...")

    for epoch in range(epochs):
        # train pipeline
        lr = scheduler.get_lr()[0]
        loss = train_pipeline(lstm_model, train_loader, lstm_model.loss, optimizer)
        writer.add_scalar('scalar/train_loss', loss, epoch+1)

        if (epoch+1) % 10 == 0:
            log_info = time.strftime("%Y-%m-%d %H:%M:%S".format(time.localtime))
            log_info += " Epoch = {}/{}, lr = {:.6f}, train loss = {:.5f}".format(epoch+1, epochs, lr, loss.data)
            print(log_info)

        if (epoch+1) % 100 == 0:
            # test pipeline
            print("run test process ...")

            correct_dict = test_pipeline(lstm_model, test_loader, cls_names)

            for cls_name, correct_num in correct_dict.items():
                tag = "acc_per_class/{}".format(cls_name)
                temp_accuracy = correct_num[1] / correct_num[0]
                writer.add_scalar(tag, temp_accuracy, epoch+1)

            total_correct = np.sum(np.array(list(correct_dict.values())), axis=0)
            accuracy = total_correct[1] / total_correct[0]
            log_info = time.strftime("%Y-%m-%d %H:%M:%S".format(time.localtime))
            log_info += " Epoch = {}/{}, test accuracy = {:.5f}".format(epoch+1, epochs, accuracy)
            writer.add_scalar('scalar/test_accuracy', accuracy, epoch+1)
            print(log_info)

            # save model
            model_file = os.path.join(model_dir, 'model_'+str(epoch+1)+'.pth')
            torch.save(lstm_model.state_dict(), model_file)
            print("save checkpoint to {}".format(model_file))

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    main()
