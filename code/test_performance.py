import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/LA/', help='Name of Experiment')
parser.add_argument('--dataset', type=str, default='LA', help='Name of dataset')
parser.add_argument('--exp', type=str, default='LA/POST', help='experiment_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--index', type=int, default=1, help='')
parser.add_argument('--labeled_num', type=int, default=4, help='labeled data')
parser.add_argument('--save_result',type=int, default=0, help='0 or 1')
parser.add_argument('--strict_load_mode',type=int, default=1, help='0 or 1')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

assert FLAGS.index in [0, 1, 2, 3]

ckpt_name = f"best_model{FLAGS.index}" if FLAGS.index > 0 else "best_model"
snapshot_path = "model/{}/{}_{}_labeled/{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
test_save_path = "model/{}/{}_{}_labeled/predictions_{}".format(FLAGS.dataset, FLAGS.exp, FLAGS.labeled_num, ckpt_name)

num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

if FLAGS.dataset == "LA":
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [f"{FLAGS.root_path}/2018LA_Seg_Training/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
elif FLAGS.dataset == "Pancreas_CT":
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [item.replace('\n', '') for item in image_list]
    image_list = [os.path.join(FLAGS.root_path, "Pancreas_h5", f"{item}_norm.h5") for item in image_list]
elif FLAGS.dataset == "BRATS":
    with open(FLAGS.root_path + '/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [f"{FLAGS.root_path}/data/" + item.replace('\n', '') + ".h5" for item in image_list]
else:
    raise NotImplementedError


def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_model_path = os.path.join(snapshot_path, '{}_{}.pth'.format(FLAGS.model, ckpt_name))

    FLAGS.strict_load_mode = True if FLAGS.strict_load_mode == 1 else False
    model.load_state_dict(torch.load(save_model_path), strict=FLAGS.strict_load_mode)
    print("init weight from {}".format(save_model_path))
    model.eval()
    
    save_result = True if FLAGS.save_result == 1 else False

    if FLAGS.dataset == "LA":
        avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                            patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                            save_result=save_result, test_save_path=test_save_path,
                            metric_detail=FLAGS.detail, nms=True, index=FLAGS.index)
    else:
        avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                           save_result=save_result, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=True, index=FLAGS.index)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)