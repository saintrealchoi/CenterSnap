import argparse
import pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
import pytorch_lightning as pl
import _pickle as cPickle
import sys
sys.path.append("simnet")
from simnet.lib.net import common
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
from simnet.lib.net.models.auto_encoder import PointCloudAE
from utils.nocs_utils import load_img_NOCS, create_input_norm
from utils.viz_utils import depth2inv, viz_inv_depth
from utils.transform_utils import get_gt_pointclouds, transform_coordinates_3d, calculate_2d_projections
from utils.transform_utils import project
from utils.viz_utils import save_projected_points, draw_bboxes, line_set_mesh, draw_gt_bboxes
from utils.nocs_eval_utils import draw_detections
import time
import glob

from utils.get_ids import get_ids_from_seg
from utils.nocs_eval_utils_od import load_depth, get_bbox, compute_mAP, plot_mAP
from tqdm import tqdm

def get_auto_encoder(model_path):
    emb_dim = 128
    n_pts = 2048
    ae = PointCloudAE(emb_dim, n_pts)
    ae.cuda()
    ae.load_state_dict(torch.load(model_path))
    ae.eval()
    return ae

def detect(
    hparams,
    data_dir, 
    output_path,
    min_confidence=0.1,
    use_gpu=True,
):
    model = PanopticModel(hparams, 0, None, None)
    model.eval()
    if use_gpu:
        model.cuda()
        
    data_path = open(os.path.join(data_dir, 'Real', 'test_list.txt')).read().splitlines()
    _CAMERA = camera.NOCS_Real()
    min_confidence = 0.50

    img_count = 0
    inst_count = 0
    
    torch.cuda.synchronize()
    t_start = time.time()
    t_inference = 0.0
  
    for i, img_path in enumerate(tqdm(data_path)):
        img_full_path = os.path.join(data_dir, 'Real', img_path)
        color_path = img_full_path + '_color.png' 
        if not os.path.exists(color_path):
            continue
        depth_full_path = img_full_path + '_depth.png'
        img_vis = cv2.imread(color_path)
        
        label_full_path = img_full_path + '_label.pkl'
        with open (label_full_path,'rb') as f:
            gts = cPickle.load(f)
        
        left_linear, depth, actual_depth = load_img_NOCS(color_path, depth_full_path)
        # input = color_img
        input = create_input_norm(left_linear, depth)
        input = input[None, :, :, :]
        if use_gpu:
            input = input.to(torch.device('cuda:0'))
            
        torch.cuda.synchronize()
        t_now = time.time()
        with torch.no_grad():
            seg_output, depth_output, small_depth_output, pose_output = model.forward(input)
            latent_emb_outputs, abs_pose_outputs, img_output, scores, indices= pose_output.compute_pointclouds_and_poses(min_confidence,is_target = False)
        torch.cuda.synchronize()
        t_inference += (time.time() - t_now)
        
        pred_cls_ids = get_ids_from_seg(seg_output,indices)

        auto_encoder_path = os.path.join(data_dir, 'ae_checkpoints', 'model_50_nocs.pth')
        ae = get_auto_encoder(auto_encoder_path)
        
        pred_RTs = []
        pred_sizes = []
        
        for j in range(len(latent_emb_outputs)):

            emb = latent_emb_outputs[j]
            emb = torch.FloatTensor(emb).unsqueeze(0)
            emb = emb.cuda()
            _, shape_out = ae(None, emb)
            shape_out = shape_out.cpu().detach().numpy()[0]
            rotated_pc, rotated_box, pred_size = get_gt_pointclouds(abs_pose_outputs[j], shape_out, camera_model = _CAMERA)
            sRT = abs_pose_outputs[j].camera_T_object @ abs_pose_outputs[j].scale_matrix
            #RT output
            pred_RTs.append(sRT)
            pred_sizes.append(pred_size)
            
        img_count += 1
        inst_count += len(latent_emb_outputs)
        
        result = {}
        result['pred_RTs'] = np.array(pred_RTs)
        result['pred_scales'] = np.array(pred_sizes)
        result['pred_class_ids'] = np.array(pred_cls_ids)
        result['pred_scores'] = np.array(scores)

        result['gt_bboxes'] = gts['bboxes']
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_class_ids'] = gts['class_ids']
        result['gt_handle_visibility'] = gts['handle_visibility']
        
        img_short_path = '_'.join(img_path.split('/')[-2:])
        save_path = os.path.join(output_path,'results_{}.pkl'.format(img_short_path))
        with open(save_path,'wb') as f:
            cPickle.dump(result,f)
    
    # write statistics
    fw = open('{0}/eval_logs.txt'.format(output_path), 'w')
    messages = []
    messages.append("Total images: {}".format(len(data_path)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    messages.append("Inference time: {:06f}  Average: {:06f}/image".format(t_inference, t_inference/img_count))
    messages.append("Total time: {:06f}".format(time.time() - t_start))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    
def evaluate(result_dir):
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False
    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_10_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_10_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_10_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    # load NOCS results
    # pkl_path = os.path.join('results/nocs_results', opt.data, 'mAP_Acc.pkl')
    # with open(pkl_path, 'rb') as f:
    #     nocs_results = cPickle.load(f)
    # nocs_iou_aps = nocs_results['iou_aps'][-1, :]
    # nocs_pose_aps = nocs_results['pose_aps'][-1, :, :]
    # iou_aps = np.concatenate((iou_aps, nocs_iou_aps[None, :]), axis=0)
    # pose_aps = np.concatenate((pose_aps, nocs_pose_aps[None, :, :]), axis=0)
    # # plot
    # plot_mAP(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  app_group = parser.add_argument_group('app')
  app_group.add_argument('--app_output', default='real', type=str)
  app_group.add_argument('--result_name', default='evaluation', type=str)
  app_group.add_argument('--data_dir', default='data', type=str)

  hparams = parser.parse_args()
  print(hparams)
  result_name = hparams.result_name
  path = 'results/'+result_name
  output_path = pathlib.Path(path) / hparams.app_output
  output_path.mkdir(parents=True, exist_ok=True)
  
  detect(hparams, hparams.data_dir, output_path)
  
  evaluate(output_path)