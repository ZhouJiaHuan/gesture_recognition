import torch
from tqdm import tqdm


def _count_correct(predicts, labels, cls_names):
    correct_dict = {}

    for cls_id, cls_name in enumerate(cls_names):
        label_ids = labels == cls_id
        label_num = len(labels[label_ids])
        predict = predicts[label_ids]
        correct_num = len(predict[predict == cls_id])
        correct_dict[cls_name] = [label_num, correct_num]

    return correct_dict


def _merge_batch_correct(dict1, dict2):
    dict3 = dict1.copy()
    for key, value in dict2.items():
        dict3.setdefault(key, [0, 0])
        dict3[key][0] += value[0]
        dict3[key][1] += value[1]

    return dict3


def train_pipeline(model, data_loader, loss_func, optimizer):
    model.train()

    for i, batch_data in enumerate(data_loader):
        keypoints = batch_data['keypoints'].cuda()
        labels = batch_data['label'].cuda()
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
        keypoints = batch_data['keypoints'].cuda()
        labels = batch_data['label'].cuda()
        keypoints = torch.autograd.Variable(keypoints)
        labels = torch.autograd.Variable(labels)
        out = model(keypoints)
        _, predict = torch.max(out.data, 1)
        temp_correct = _count_correct(predict, labels, cls_names)
        correct_dict = _merge_batch_correct(correct_dict, temp_correct)

    return correct_dict