import numpy as np
import os
import glob
from tqdm import tqdm
from gesture_lib.ops import make_dirs


class BodyGenerator(object):
    '''
    generate the source body keypoints info for train and test

    src_dir:
        - gesture1
            - xxx.txt
            - xxx.txt
            - ...
        - gesture2
            - xxx.txt
            - xxx.txt
            - ...
        - ...

    out_dir:
        - gesture1:
            - gesture1_1.txt
            - gesture1_2.txt
            - ...
        - gesture2:
            - gesture2_1.txt
            - gesture2_2.txt
            - ...
        - ...

    process pipeline:
        - extract keypoints with specified body name, (N, :) -> (M, :) (M <= N)
        - drop the keypoints with low score
        - save body keypoints extracted to txt
    '''

    def __init__(self, cls_names):
        '''
        Args:
            cls_names: [str list], eg. [wave, stop, other]

        '''
        self.cls_names = cls_names

    def generate(self, src_dir, out_dir, ext, seq_len=30, gap=10, ignore=5):
        '''

        src_dir:
            - gesture_1
                - xxx.txt
                - xxx.txt
                - ...
            - gesture_2
                - xxx.txt
                - xxx.txt
                - ...
            - ...
        out_dir:
            - gesture_1:
                - 1.txt
                - 2.txt
                - ...
            - gesture_2:
                - 1.txt
                - 2.txt
                - ...
            - ...
                
        Args:
            out_dir: [str], the dir of the extracted keypoints saved
            seq_len: [int], sequence length for one sample (eg. for LSTM)

        Return:
            None
        '''
        assert os.path.exists(src_dir) and src_dir != out_dir

        for cls_name in self.cls_names:
            print("process class: {}".format(cls_name))
            current_dir = os.path.join(src_dir, cls_name)
            if not os.path.isdir(current_dir):
                continue
            dst_dir = os.path.join(out_dir, cls_name)
            make_dirs(dst_dir)
            
            idx_s = 1
            path_list = glob.glob(os.path.join(current_dir, '*.txt'))
            for txt_path in tqdm(path_list):
                try:
                    keypoints_array = np.loadtxt(txt_path)
                except Exception as e:
                    print("extract failed for {}".format(txt_path))
                    print(e)
                    continue
                
                T = keypoints_array.shape[0]
                if T < seq_len + 2 * ignore:
                    print("{} is too short.".format(txt_path))
                    continue
                keypoints_array = keypoints_array[ignore:-ignore]
                T = keypoints_array.shape[0]
                crop_num = int(np.floor((T-seq_len)/gap)+1)
                for i in range(crop_num):
                    temp_keypoints = keypoints_array[i*gap:i*gap+seq_len, :]
                    save_txt = os.path.join(dst_dir, cls_name + '_' + str(idx_s+i) + ext + '.txt')
                    try:
                        np.savetxt(save_txt, temp_keypoints, fmt='%.5f')
                    except Exception as e:
                        print("save failed for {} at times {}".format(txt_path, i))
                        print(e)
                        continue

                idx_s += crop_num
