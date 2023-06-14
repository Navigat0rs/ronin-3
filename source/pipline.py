# import json
# import random
# from os import path as osp
#
# import h5py
import numpy as np
# import quaternion
# from scipy.ndimage import gaussian_filter1d
# from torch.utils.data import Dataset
#
# from data_utils import CompiledSequence, select_orientation_source, load_cached_sequences
#
#
# class GlobSpeedSequence(CompiledSequence):
#     """
#     Dataset :- RoNIN (can be downloaded from http://ronin.cs.sfu.ca/)
#     Features :- raw angular rate and acceleration (includes gravity).
#     """
#     feature_dim = 6
#     target_dim = 2
#     aux_dim = 8
#
#     def __init__(self, data_path=None, **kwargs):
#         super().__init__(**kwargs)
#         self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
#         self.info = {}
#
#         self.grv_only = kwargs.get('grv_only', False)
#         self.max_ori_error = kwargs.get('max_ori_error', 20.0)
#         self.w = kwargs.get('interval', 1)
#         if data_path is not None:
#             self.load(data_path)
#
#     def load(self, data_path):
#         if data_path[-1] == '/':
#             data_path = data_path[:-1]
#         with open(osp.join(data_path, 'info.json')) as f:
#             self.info = json.load(f)
#
#         self.info['path'] = osp.split(data_path)[-1]
#
#         self.info['ori_source'], ori, self.info['source_ori_error'] = select_orientation_source(
#             data_path, self.max_ori_error, self.grv_only)
#
#         with h5py.File(osp.join(data_path, 'data.hdf5')) as f:
#             gyro_uncalib = f['synced/gyro_uncalib']
#             acce_uncalib = f['synced/acce']
#             gyro = gyro_uncalib - np.array(self.info['imu_init_gyro_bias'])
#             acce = np.array(self.info['imu_acce_scale']) * (acce_uncalib - np.array(self.info['imu_acce_bias']))
#             ts = np.copy(f['synced/time'])
#             tango_pos = np.copy(f['pose/tango_pos'])
#             init_tango_ori = quaternion.quaternion(*f['pose/tango_ori'][0])
#
#         # Compute the IMU orientation in the Tango coordinate frame.
#         ori_q = quaternion.from_float_array(ori)
#         rot_imu_to_tango = quaternion.quaternion(*self.info['start_calibration'])
#         init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
#         ori_q = init_rotor * ori_q
#
#         dt = (ts[self.w:] - ts[:-self.w])[:, None]
#         glob_v = (tango_pos[self.w:] - tango_pos[:-self.w]) / dt
#
#         gyro_q = quaternion.from_float_array(np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1))
#         acce_q = quaternion.from_float_array(np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1))
#         glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
#         glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]
#
#         start_frame = self.info.get('start_frame', 0)
#         self.ts = ts[start_frame:]
#         self.features = np.concatenate([glob_gyro, glob_acce], axis=1)[start_frame:]
#         self.targets = glob_v[start_frame:, :2]
#         self.orientations = quaternion.as_float_array(ori_q)[start_frame:]
#         self.gt_pos = tango_pos[start_frame:]
#
#     def get_feature(self):
#         return self.features
#
#     def get_target(self):
#         return self.targets
#
#     def get_aux(self):
#         return np.concatenate([self.ts[:, None], self.orientations, self.gt_pos], axis=1)
#
#     def get_meta(self):
#         return '{}: device: {}, ori_error ({}): {:.3f}'.format(
#             self.info['path'], self.info['device'], self.info['ori_source'], self.info['source_ori_error'])

import numpy as np
import torch
from model_resnet1d import *

def preprocess(imu):
    #input numpy shape (200,6)
    imu=imu[0]
    gyro_uncalib=imu[:,:3]
    acce_uncalib=imu[:,3:]
    gyro_bias=0
    acce_bias=0
    gyro = gyro_uncalib - gyro_bias
    acce = acce_uncalib-acce_bias

    z=np.concatenate([gyro,acce],axis=1)
    zz=torch.tensor(z,device="cuda:0",dtype=torch.float32)
    zz=torch.unsqueeze(zz, 0)
    zz=torch.transpose(zz,1,2)

    return (zz)

def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


# a=np.random.uniform(0.1, 0.8, size=(1, 200, 6))
# _fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
# network = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2],
#                            base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
# checkpoint = torch.load("D:\\000_Mora\\FYP\\RONiN\\Pre_trained models\\ronin_resnet\\ronin_resnet\\checkpoint_gsn_latest.pt")
# network.load_state_dict(checkpoint['model_state_dict'])
# device = torch.device('cuda:0')
# network.eval().to(device)
#
# a=preprocess(a)
# import pdb
# pdb.set_trace()
# z=network(a)
# print(z)




# def test_sequence(args):
#     if args.test_path is not None:
#         if args.test_path[-1] == '/':
#             args.test_path = args.test_path[:-1]
#         root_dir = osp.split(args.test_path)[0]
#         test_data_list = [osp.split(args.test_path)[1]]
#     elif args.test_list is not None:
#         root_dir = args.root_dir
#         with open(args.test_list) as f:
#             test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
#     else:
#         raise ValueError('Either test_path or test_list must be specified.')
#
#     if args.out_dir is not None and not osp.isdir(args.out_dir):
#         os.makedirs(args.out_dir)
#
#     if not torch.cuda.is_available() or args.cpu:
#         device = torch.device('cpu')
#         checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
#     else:
#         device = torch.device('cuda:0')
#         checkpoint = torch.load(args.model_path)
#
#     # Load the first sequence to update the input and output size
#     _ = get_dataset(root_dir, [test_data_list[0]], args)
#
#     global _fc_config
#     _fc_config['in_dim'] = args.window_size // 32 + 1
#
#     network = get_model(args.arch)
#
#     network.load_state_dict(checkpoint['model_state_dict'])
#     network.eval().to(device)
#     print('Model {} loaded to device {}.'.format(args.model_path, device))
#
#     preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
#     traj_lens = []
#
#     pred_per_min = 200 * 60
#
#     for data in test_data_list:
#         seq_dataset = get_dataset(root_dir, [data], args, mode='test')
#         seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
#         ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int)
#
#         targets, preds = run_test(network, seq_loader, device, True)
#         losses = np.mean((targets - preds) ** 2, axis=0)
#         preds_seq.append(preds)
#         targets_seq.append(targets)
#         losses_seq.append(losses)
#
#         pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
#         pos_gt = seq_dataset.gt_pos[0][:, :2]
#
#         traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
#         ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
#         ate_all.append(ate)
#         rte_all.append(rte)
#         pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
#
#         print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))
#
#         # Plot figures
#         kp = preds.shape[1]
#         if kp == 2:
#             targ_names = ['vx', 'vy']
#         elif kp == 3:
#             targ_names = ['vx', 'vy', 'vz']
#
#         plt.figure('{}'.format(data), figsize=(16, 9))
#         plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
#         plt.plot(pos_pred[:, 0], pos_pred[:, 1])
#         plt.plot(pos_gt[:, 0], pos_gt[:, 1])
#         plt.title(data)
#         plt.axis('equal')
#         plt.legend(['Predicted', 'Ground truth'])
#         plt.subplot2grid((kp, 2), (kp - 1, 0))
#         plt.plot(pos_cum_error)
#         plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
#         for i in range(kp):
#             plt.subplot2grid((kp, 2), (i, 1))
#             plt.plot(ind, preds[:, i])
#             plt.plot(ind, targets[:, i])
#             plt.legend(['Predicted', 'Ground truth'])
#             plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
#         plt.tight_layout()
#
#         if args.show_plot:
#             plt.show()
#
#         if args.out_dir is not None and osp.isdir(args.out_dir):
#             np.save(osp.join(args.out_dir, data + '_gsn.npy'),
#                     np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
#             plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))
#
#         plt.close('all')
#
#     losses_seq = np.stack(losses_seq, axis=0)
#     losses_avg = np.mean(losses_seq, axis=1)
#     # Export a csv file
#     if args.out_dir is not None and osp.isdir(args.out_dir):
#         with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
#             if losses_seq.shape[1] == 2:
#                 f.write('seq,vx,vy,avg,ate,rte\n')
#             else:
#                 f.write('seq,vx,vy,vz,avg,ate,rte\n')
#             for i in range(losses_seq.shape[0]):
#                 f.write('{},'.format(test_data_list[i]))
#                 for j in range(losses_seq.shape[1]):
#                     f.write('{:.6f},'.format(losses_seq[i][j]))
#                 f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))
#
#     print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
#         np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
#     return losses_avg



