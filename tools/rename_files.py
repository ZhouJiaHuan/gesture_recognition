from gesture_lib.ops.io import get_file_path
import os


def rename_bag(bag_dir, pre, start_idx=1):
    bag_list = get_file_path(bag_dir, filter='.bag')
    for i, bag_path in enumerate(sorted(bag_list)):
        dst_name = pre + str(start_idx+i) + '.bag'
        dst_path = os.path.join(bag_dir, dst_name)
        assert not os.path.exists(dst_path)
        os.rename(bag_path, dst_path)

        txt_path = bag_path.replace('.bag', '.txt')
        if os.path.exists(txt_path):
            dst_txt_path = dst_path.replace('.bag', '.txt')
            os.rename(txt_path, dst_txt_path)

def rename_txt(txt_dir, pre, start_idx=1):
    txt_list = get_file_path(txt_dir, filter='.txt')
    for i, txt_path in enumerate(sorted(txt_list)):
        dst_name = pre + str(start_idx+i) + '.txt'
        dst_path = os.path.join(txt_dir, dst_name)
        assert not os.path.exists(dst_path)
        os.rename(txt_path, dst_path)


txt_dir = "/home/zhoujh/Data/gesture_recognition/bag_files/_history_data/test_aug/come"

rename_txt(txt_dir, pre='come_his_')