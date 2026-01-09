import argparse
import logging
import os, math
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils import losses, test_3d_patch, ramps
# from einops import rearrange

from copy import deepcopy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="LA", help="dataset for experiments")
parser.add_argument('--root_path', type=str, default='../', help='')
parser.add_argument('--exp', type=str, default='LA/', help='experiment_name')
parser.add_argument('--model', type=str, default='unet_3D', help='model_name')

parser.add_argument('--patch_size', nargs='+', type=int)
parser.add_argument('--num_classes', type=int, default=2, help='')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float,  default=1e-4, help='')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=25, help='labeled data')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
# parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float, default=1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40, help='consistency_rampup')

parser.add_argument('--step_normgrad', type=int, default=0, help='')
parser.add_argument('--softpl_mask_thd', type=float, default=0.0, help='')
parser.add_argument('--faster_factor', type=float, default=1.0, help='')

parser.add_argument('--flag_rotflip', type=int, default=0, help='')
parser.add_argument('--cutmix_prob', type=float, default=0.5, help='probability of applying cutmix')
parser.add_argument('--cutmix_beta', type=float, nargs='+', default=[4,4], help='')


args = parser.parse_args()

dice_loss = losses.mask_DiceLoss(nclass=args.num_classes)

class BackupModel():
    def __init__(self, model, norm_grad=False):
        self.model = model
        self.backup = {}
        self.norm_grad = norm_grad

    def backup_param(self):
        for name, param in self.model.named_parameters():
            self.backup[name] = param.data.clone()

    def step(self, epsilon=1.0):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data
                if self.norm_grad:
                    norm = torch.norm(param.grad)
                    if norm.item() != 0:
                        grad.div_(norm)
                param.data.add_(grad, alpha=-epsilon)
                 
    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name]
            if param.grad is not None:
                param.grad.data = torch.zeros_like(param.data)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# bcp 
def cutmix_mask(img, mask_ratio, p=0.5):
    mask_ratio = np.sqrt(1. - mask_ratio)
    
    _, _, img_x, img_y, img_z = img.shape
    mask = torch.zeros_like(img[:, 0], requires_grad=False)
    if np.random.rand() > p:
        return mask.long()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[:, w : w + patch_pixel_x, 
            h : h + patch_pixel_y, 
            z : z + patch_pixel_z] = 1
    return mask.long()

def criterion(logits, label, mask=None):
    logits = logits.flatten(2)
    label = label.flatten(1)
    if mask is not None:
        mask = mask.flatten(1)
    loss_dc = dice_loss(logits, label, mask)
    loss_ce = F.cross_entropy(logits, label, reduction="none")
    if mask is not None:
        loss_ce = torch.mean(torch.sum(loss_ce*mask, dim=-1) / (torch.sum(mask, dim=-1)+1e-12))
    else:
        loss_ce = torch.mean(torch.mean(loss_ce, dim=-1))
    loss = 0.5 * (loss_ce + loss_dc)
    return loss

def cutmix_loss(logits, label, criterion, cm_mask=None, uda_mask=None):
    if cm_mask is not None:
        logits[cm_mask.unsqueeze(1).expand_as(logits) == 1] = \
            logits.flip(0)[cm_mask.unsqueeze(1).expand_as(logits) == 1]
    loss = criterion(logits, label, uda_mask)
    return loss

def train(args, snapshot_path):
    base_lr = args.base_lr
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    lbs = args.labeled_bs
    num_classes = args.num_classes
    
    flag_rotflip = True if args.flag_rotflip == 1 else False
    step_normgrad = True if args.step_normgrad == 1 else False

    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=args.root_path,
                           split='train', num=None,
                           transform=transforms.Compose([
                                DoubleStrongAugment(args.patch_size,flag_rot=flag_rotflip) 
                        ]))
    elif args.dataset_name == "Pancreas_CT":
         db_train = Pancreas(base_dir=args.root_path,
                        split='train', num=None, 
                        transform=transforms.Compose([
                            DoubleStrongAugment(args.patch_size,flag_rot=flag_rotflip)
                        ]))
    elif args.dataset_name == "BRATS":
        db_train = BraTS2019(base_dir=args.root_path,
                        split='train', num=None, 
                        transform=transforms.Compose([
                            DoubleStrongAugment(args.patch_size,flag_rot=flag_rotflip)
                        ]))
    else:
        raise NotImplementedError
    
    # dataloader
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    labelnum = args.labeled_num
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-lbs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    
    teacher_math = net_factory(net_type=args.model, mode="train", in_chns=1, class_num=args.num_classes)
    teacher_ling = net_factory(net_type=args.model, mode="train", in_chns=1, class_num=args.num_classes)
    student = net_factory(net_type=args.model, mode="train", in_chns=1, class_num=args.num_classes)

    teacher_math.train()
    teacher_ling.train()
    student.train()

    back_student_util = BackupModel(student, norm_grad=step_normgrad)

    t_optimizer = optim.SGD([{"params": teacher_math.parameters()}, {"params": teacher_ling.parameters()}], 
                            lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    s_optimizer = optim.SGD(student.parameters(), lr=base_lr, momentum=0.9, weight_decay=args.weight_decay)
    
    t_scheduler = optim.lr_scheduler.LambdaLR(t_optimizer, lr_lambda=lambda x: (1. - x/max_iterations)**0.9)
    s_scheduler = optim.lr_scheduler.LambdaLR(s_optimizer, lr_lambda=lambda x: (1. - x/max_iterations)**0.9)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice_s = 0.0 
    best_iter_s = 0
    
    threshold = args.softpl_mask_thd
    
    tracing_count = 25
    loss_tracing = dict(t_loss=0., t_label=0., t_uda=0., t_fb=0., s_loss=0.)

    max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=50)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'],  sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_w_l, img_w_u = volume_batch[:lbs], volume_batch[lbs:]
            img_s1_u, img_s2_u = sampled_batch["s1_image"][lbs:].cuda(), sampled_batch["s2_image"][lbs:].cuda()
            
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            # Pseudo-labels
            math_logits_u = teacher_math(img_w_u)
            ling_logits_u = teacher_ling(img_w_u)
            math_prob, math_pl = torch.max(F.softmax(math_logits_u, 1), 1)
            ling_prob, ling_pl = torch.max(F.softmax(ling_logits_u, 1), 1)
            
            back_student_util.backup_param()
            s_optimizer.zero_grad()

            with torch.no_grad():
                # Agreement of low-confidence
                math_mask_lowconf = torch.logical_and(math_pl==ling_pl, math_prob<ling_prob).type(math_prob.dtype).detach()
                ling_mask_lowconf = torch.logical_and(math_pl==ling_pl, math_prob>=ling_prob).type(math_prob.dtype).detach()
                
                # Disagreement of high-confidence 
                math_mask_highconf = torch.logical_and(math_pl!=ling_pl, math_prob>ling_prob).type(math_prob.dtype).detach()
                ling_mask_highconf = torch.logical_and(math_pl!=ling_pl, math_prob<=ling_prob).type(math_prob.dtype).detach()
                
                math_mask = torch.logical_or(math_mask_lowconf, math_mask_highconf).type(math_prob.dtype)
                u_pl = torch.where(math_mask==1, math_pl, ling_pl)
                
                agreement = torch.logical_or(math_mask_lowconf, ling_mask_lowconf) 
                disagreement = torch.logical_or(math_mask_highconf, ling_mask_highconf)

                math_reliable = math_prob.ge(threshold).float().detach() if threshold > 0 else None
                ling_reliable = ling_prob.ge(threshold).float().detach() if threshold > 0 else None

                mix_img_s1 = deepcopy(img_s1_u)
                mix_img_s2 = deepcopy(img_s2_u)
                mix_u1_mask = cutmix_mask(mix_img_s1, np.random.beta(*args.cutmix_beta), p=args.cutmix_prob).cuda()
                # UniMatch 
                mix_img_s1[mix_u1_mask.unsqueeze(1).expand_as(mix_img_s1)==1] = \
                    mix_img_s1.flip(0)[mix_u1_mask.unsqueeze(1).expand_as(mix_img_s1)==1]
                    
                mix_u2_mask = cutmix_mask(mix_img_s2, np.random.beta(*args.cutmix_beta), p=args.cutmix_prob).cuda()
                mix_img_s2[mix_u2_mask.unsqueeze(1).expand_as(mix_img_s2)==1] = \
                    mix_img_s2.flip(0)[mix_u2_mask.unsqueeze(1).expand_as(mix_img_s2)==1]
            
            # Feedback evaluation
            eta_step = base_lr*args.faster_factor
            def forward_to_feedback(loss_mask, start_ce, epsilon):
                back_student_util.restore()
                s_logits_u = student(img_w_u)
                tmp_loss = criterion(s_logits_u, u_pl.detach(), loss_mask)
                tmp_loss.backward()
                back_student_util.step(epsilon=epsilon)
                
                with torch.no_grad():
                    s_logits_l = student(img_w_l)
                    updated_ce = F.cross_entropy(s_logits_l, label_batch[:lbs], reduction="none")
                    feedback_ce = start_ce - updated_ce
                    fed = torch.mean(feedback_ce)
                return tmp_loss, fed
            
            with torch.no_grad():
                s_logits_l = student(img_w_l)
                fed_ce_start = F.cross_entropy(s_logits_l, label_batch[:lbs], reduction="none")
            
            # Agreement
            s_loss_lconf, fed_agree_lconf = forward_to_feedback(agreement.detach(), fed_ce_start, eta_step)
            # Disagreement
            s_loss_hconf, fed_disag_hconf = forward_to_feedback(disagreement.detach(), fed_ce_start, eta_step)       
                
            # Student supervised by pseudo-labels
            back_student_util.restore()
            s_logits_u = student(img_w_u)
            s_loss_u = criterion(s_logits_u, u_pl.detach())
            s_optimizer.zero_grad()
            s_loss_u.backward()
            s_optimizer.step()
                
            # feedback loss 
            math_logits_fb = teacher_math(img_s1_u)
            math_fed_pl = torch.argmax(math_logits_fb, 1)
            math_likeli = F.cross_entropy(math_logits_fb, math_fed_pl.detach(), reduction="none")
            
            ling_logits_fb = teacher_ling(img_s2_u)
            ling_fed_pl = torch.argmax(ling_logits_fb, 1)
            ling_likeli = F.cross_entropy(ling_logits_fb, ling_fed_pl.detach(), reduction="none")
                
            fed_to_lmath = torch.sum(math_likeli*math_mask_lowconf)/(torch.sum(math_mask_lowconf)+1e-12)
            fed_to_hmath = torch.sum(math_likeli*math_mask_highconf)/(torch.sum(math_mask_highconf)+1e-12)
            fed_to_lling = torch.sum(ling_likeli*ling_mask_lowconf)/(torch.sum(ling_mask_lowconf)+1e-12)
            fed_to_hling = torch.sum(ling_likeli*ling_mask_highconf)/(torch.sum(ling_mask_highconf)+1e-12)
                
            feedback_math = fed_agree_lconf*fed_to_lmath + fed_disag_hconf*fed_to_hmath
            feedback_ling = fed_agree_lconf*fed_to_lling + fed_disag_hconf*fed_to_hling
            fb_loss = feedback_math + feedback_ling
            
            writer.add_scalars('student/s_loss', {"loss_lconf": s_loss_lconf, "loss_hconf": s_loss_hconf, "merge": s_loss_u}, iter_num+1)
            writer.add_scalars('student/fb_loss', {"math": feedback_math, "ling": feedback_ling, "merge": fb_loss}, iter_num+1)
        
            # TEACHER sup-loss
            math_logits_l = teacher_math(img_w_l)
            ling_logits_l = teacher_ling(img_w_l)
            tl_loss = criterion(math_logits_l, label_batch[:lbs]) + criterion(ling_logits_l, label_batch[:lbs])
            
            # TEACHER Weak-to-Strong consistency 
            math_logits_uda = teacher_math(mix_img_s1)
            ling_logits_uda = teacher_ling(mix_img_s2)  
            math_uda_loss = cutmix_loss(math_logits_uda, ling_pl.detach(), criterion, mix_u1_mask, ling_reliable)
            ling_uda_loss = cutmix_loss(ling_logits_uda, math_pl.detach(), criterion, mix_u2_mask, math_reliable)
            uda_loss = math_uda_loss + ling_uda_loss
            
            t_loss = tl_loss + fb_loss + consistency_weight * uda_loss
            
            t_optimizer.zero_grad()
            t_loss.backward()
            t_optimizer.step()
        
            s_scheduler.step()
            t_scheduler.step()
            
            t_lr, s_lr = t_optimizer.param_groups[0]["lr"], s_optimizer.param_groups[0]["lr"]

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_uda', uda_loss, iter_num)
            writer.add_scalar('Self/loss_teacher', t_loss, iter_num)
            writer.add_scalar('Self/loss_fb', fb_loss, iter_num)
            writer.add_scalar('Self/loss_student', s_loss_u, iter_num)

            loss_tracing['t_fb'] += fb_loss.item()
            loss_tracing['s_loss'] += s_loss_u.item()
            loss_tracing['t_loss'] += t_loss.item()
            loss_tracing['t_label'] += tl_loss.item()
            loss_tracing['t_uda'] += uda_loss.item()

            if iter_num % tracing_count == 0:
                loss_tracing = {k: v / tracing_count for k, v in loss_tracing.items()}          
                logging.info("Iter %d - T: total %.4f, label %.4f, UDA %.4f, FB %.6f (argee %.4f, disagree %.4f) | S: %.4f | consW, %.4f t_lr %.4f, s_lr %.4f, eps %.4f" % 
                             (iter_num, loss_tracing['t_loss'], loss_tracing['t_label'], loss_tracing['t_uda'], 
                              loss_tracing['t_fb'], fed_agree_lconf, fed_disag_hconf, loss_tracing['s_loss'], 
                              consistency_weight, t_lr*(1/base_lr), s_lr*(1/base_lr), eta_step*(1/base_lr)))
                loss_tracing = {k: 0. for k, v in loss_tracing.items()}
    
            if iter_num % 200 == 0:
                student.eval()
                if args.dataset_name =="LA":
                    dice_sample1, per_case_time = test_3d_patch.var_all_case_LA(student, args.root_path, num_classes=num_classes, patch_size=args.patch_size, 
                                                                                stride_xy=18, stride_z=4, flag_nms=True, time_verbose=True)
                elif args.dataset_name == "Pancreas_CT":
                    dice_sample1, per_case_time = test_3d_patch.var_all_case_Pancrease(student, args.root_path, num_classes=num_classes, patch_size=args.patch_size, 
                                                                                       stride_xy=16, stride_z=16, flag_nms=True)
                elif args.dataset_name == "BRATS":
                    dice_sample1, per_case_time = test_3d_patch.var_all_case_BraTS19(student, args.root_path, num_classes=num_classes, patch_size=args.patch_size, 
                                                                                     stride_xy=16, stride_z=16, flag_nms=True)
                    
                if dice_sample1 > best_dice_s:
                    best_dice_s = round(dice_sample1, 4)
                    best_iter_s = iter_num
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice_s))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model1.pth'.format(args.model))
                    # torch.save(student.state_dict(), save_mode_path)
                    torch.save(student.state_dict(), save_best_path)
                    logging.info("save best model1 to {}".format(save_mode_path))
                logging.info("Valid student iter %d : dice %.2f avg time %.4f s/case | best %.2f @ %d" % (iter_num, dice_sample1*100, per_case_time, best_dice_s*100, best_iter_s))
                writer.add_scalar('4_Var_dice/Model1_dice', dice_sample1, iter_num)
                writer.add_scalar('4_Var_dice/Model1_best_dice', best_dice_s, iter_num)
                student.train()
                
            if iter_num == max_iterations:
                logging.info("save last model.")
                save_model_path = os.path.join(snapshot_path, '{}_last_model1.pth'.format(args.model))
                torch.save(student.state_dict(), save_model_path)
                # save_model_path = os.path.join(snapshot_path, '{}_last_model2.pth'.format(args.model))
                # torch.save(teacher_math.state_dict(), save_model_path)
                # save_model_path = os.path.join(snapshot_path, '{}_last_model3.pth'.format(args.model))
                # torch.save(teacher_ling.state_dict(), save_model_path)

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "model/{}/{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('code', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(__file__)
    logging.info(str(args))
    
    train(args, snapshot_path)


