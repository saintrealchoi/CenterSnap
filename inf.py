import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from simnet.lib.net.models.auto_encoder import PointCloudAE
from external.shape_pretraining.dataset.shape_dataset import ShapeDataset
from utils.tsne import tsne

def visualize_shape(name, shape_list, result_dir):
    """ Visualization and save image.
    Args:
        name: window name
        shape: list of geoemtries
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=512, height=512, left=50, top=25)
    for shape in shape_list:
        vis.add_geometry(shape)
    ctr = vis.get_view_control()
    ctr.rotate(-300.0, 150.0)
    if name == 'camera':
        ctr.translate(20.0, -20.0)     # (horizontal right +, vertical down +)
    if name == 'laptop':
        ctr.translate(25.0, -60.0)
    vis.run()
    vis.capture_screen_image(os.path.join(result_dir, name+'.png'), False)
    vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', type=str, default='data/obj_models/ShapeNetCore_2048.h5', help='h5py file')
parser.add_argument('--model', type=str, default='data/ae_checkpoints/model_50_nocs.pth',  help='resume model')
parser.add_argument('--result_dir', type=str, default='results/ae_points', help='directory to save mean shapes')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
opt = parser.parse_args()
opt.emb_dim = 128
opt.n_cat = 6
opt.n_pts = 2048
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

estimator = PointCloudAE(opt.emb_dim, opt.n_pts)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()
train_dataset = ShapeDataset(opt.h5_file, mode='val', augment=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

obj_models = []
embedding = []
catId = []  # zero-indexed
for i, data in enumerate(train_dataloader):
    batch_xyz, batch_label = data
    batch_xyz = batch_xyz[:, :, :3].cuda()
    batch_label = batch_label.cuda()
    emb, pred_points = estimator(batch_xyz)
    emb = emb.cpu().detach().numpy()
    inst_shape = batch_xyz.cpu().numpy()
    label = batch_label.cpu().numpy()
    embedding.append(emb)
    obj_models.append(inst_shape)
    catId.append(label)
    
#  mean embedding and mean shape
mean_emb = np.empty((opt.n_cat, opt.emb_dim), dtype=np.float)
catId_to_name = {0: 'bottle', 1: 'bowl', 2: 'camera', 3: 'can', 4: 'laptop', 5: 'mug'}
mean_points = np.empty((opt.n_cat, opt.n_pts, 3), dtype=np.float)

for i in range(len(embedding)):
    cat = int(catId[i])
    if cat != 2:
        continue
    emb = embedding[i]
    assigned_emb = torch.cuda.FloatTensor(emb[None, :])
    _, shape = estimator(None, assigned_emb)
    shape = shape.cpu().detach().numpy()[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shape)
    visualize_shape(catId_to_name[cat], [pcd], opt.result_dir)
    
# for i in range(opt.n_cat):
#     mean = np.mean(embedding[np.where(catId==i)[0], :], axis=0, keepdims=False)
#     mean_emb[i] = mean
#     assigned_emb = torch.cuda.FloatTensor(mean[None, :])
#     _, mean_shape = estimator(None, assigned_emb)
#     mean_shape = mean_shape.cpu().detach().numpy()[0]
#     mean_points[i] = mean_shape
#     # save point cloud and visualize
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(mean_shape)
#     visualize_shape(catId_to_name[i], [pcd], opt.result_dir)