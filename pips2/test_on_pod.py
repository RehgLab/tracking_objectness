import time
import numpy as np
import saverloader
#from nets.pips2 import Pips, data_per_itr_motion, data_per_itr_motion_4
from nets.pips2 import Pips, Pips2#, delta_coord_itr1, delta_coord_itr2
import utils.improc
import utils.misc
import random, os, cv2
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
from torch.utils.data import Dataset, DataLoader
from datasets.pointodysseydataset_fullseq import PointOdysseyDataset
import imageio


delta_1=[]
delta_2=[]

def create_pools(n_pool=1000):
    pools = {}
    pool_names = [
        'd_1',
        'd_2',
        'd_4',
        'd_8',
        'd_16',
        'd_avg',
        'median_l2',
        'survival',
        'ate_all',
        'ate_vis',
        'ate_occ',
        'total_loss',
    ]
    for pool_name in pool_names:
        pools[pool_name] = utils.misc.SimplePool(n_pool, version='np')
    return pools
def make_video(seq, rgbs, pred=True):

    print(rgbs.shape)
    rgbs =rgbs[0]

    S, C, H, W = rgbs.shape
    path = "PIPS_DIST_VIz"
    seq_path = os.path.join(path, seq)
    print(seq_path)
    if not os.path.isdir(seq_path):
        os.makedirs(seq_path)

    #video = cv2.VideoWriter('{}.mp4'.format(seq), -1, 10, (H,W))
    for frame in range(S):
        print(frame)
        img = cv2.cvtColor(cv2.rotate(rgbs[frame].permute(2,1,0).numpy(), cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
        if pred:
            cv2.imwrite(os.path.join(seq_path, "frame_{}.jpg".format(frame)),img)
        else:
            cv2.imwrite(os.path.join(seq_path, "frameg_{}.jpg".format(frame)),img)
        cv2.imshow("Frame",cv2.rotate(rgbs[frame].permute(2,1,0).numpy(), cv2.ROTATE_90_CLOCKWISE))

        if cv2.waitKey(1) & 0xFF == ord('s'): 
            break
        #video.write(rgbs[frame].permute(1,2,0).numpy())

    cv2.destroyAllWindows()
    

def test_on_fullseq(model, d, sw, iters=8, S_max=8, image_size=(384,512)):
    metrics = {}
    window_metrics = {}
    window_metrics['d_avg'] = []

    seq = str(d['seq'][0])
    print('seq', seq.split("/"))
    trajs_g = d['trajs'].cuda().float()[:,:,:,:] # B,S,N,2
    #trajs_g = trajs_g[motion]
    #print("GT Traj", trajs_g[:,10,:5])
    visibs = d['visibs'].cuda().float()[:,:,:] # B,S,N
    valids = d['valids'].cuda().float()[:,:,:] # B,S,N
    #print(valids[:,::10,::4])
    #valids = valids*visibs

    B, S, N, D = trajs_g.shape
    assert(D==2)
    assert(B==1)
    print('this video is %d frames long' % S)

    # load one to check H,W
    rgb_path0 = os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (0))
    rgb0_bak = cv2.imread(rgb_path0)
    H_bak, W_bak = rgb0_bak.shape[:2]
    H, W = image_size
    sy = H/H_bak
    sx = W/W_bak
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    rgb0_bak = cv2.resize(rgb0_bak, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb0_bak = torch.from_numpy(rgb0_bak[:,:,::-1].copy()).permute(2,0,1) # 3,H,W
    rgb0_bak = rgb0_bak.unsqueeze(0).to(trajs_g.device) # 1,3,H,W

    if sw is not None and sw.save_this:
        prep_rgb0 = utils.improc.preprocess_color(rgb0_bak)
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb0', trajs_g[0:1], prep_rgb0, valids=valids[0:1], cmap='winter', linewidth=1)
        
    # zero-vel init
    trajs_e = trajs_g[:,0].repeat(1,S,1,1)
    
    cur_frame = 0
    done = False
    feat_init = None
    #motion_feat = None
    while not done:
        end_frame = cur_frame + S_max
        

        if end_frame > S:
            diff = end_frame-S
            end_frame = end_frame-diff
            cur_frame = max(cur_frame-diff,0)
        #print('working on subseq %d:%d' % (cur_frame, end_frame))

        traj_seq = trajs_e[:, cur_frame:end_frame]

        idx_seq = np.arange(cur_frame, end_frame)
        rgb_paths_seq = [os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (idx)) for idx in idx_seq]
        rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths_seq]
        rgbs = [rgb[:,:,::-1] for rgb in rgbs] # BGR->RGB
        H_load, W_load = rgbs[0].shape[:2]
        assert(H_load==H_bak and W_load==W_bak)
        rgbs = [cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        rgb_seq = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
        rgb_seq = rgb_seq.unsqueeze(0).to(traj_seq.device) # 1,S,3,H,W
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
        #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        preds, preds_anim, feat_init, _, delta_coord_itr1, delta_coord_itr2 = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init)
        #print(delta_coord_itr1)
        delta_1.append(delta_coord_itr1)
        delta_2.append(delta_coord_itr2)

        trajs_e[:, cur_frame:end_frame] = preds[-1][:, :S_local]
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame] # update the future with new zero-vel
        

        #trajs_e[:, end_frame:] = trajs_g[:,end_frame-1:end_frame] #only to check window performance, change later

        # if sw is not None and sw.save_this:
        #     traj_seq_e = preds[-1]
        #     traj_seq_g = trajs_g[:,cur_frame:end_frame]
        #     valid_seq = valids[:,cur_frame:end_frame]
        #     prep_rgbs = utils.improc.preprocess_color(rgb_seq)
        #     gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        #     gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', traj_seq_g, gray_rgbs[0:1].mean(dim=1), valids=valid_seq, cmap='winter', only_return=True))
        #     rgb_vis = []
        #     for tre in preds_anim:
        #         ate = torch.norm(tre - traj_seq_g, dim=-1) # B,S,N
        #         ate_all = utils.basic.reduce_masked_mean(ate, valid_seq, dim=[1,2]) # B
        #         rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valid_seq, only_return=True, cmap='spring', frame_id=ate_all[0]))
        #     sw.summ_rgbs('3_test/animated_trajs_on_rgb_cur%02d' % cur_frame, rgb_vis)
        #print(H,W, H_bak, W_bak)


        #Compute Windowed Metric
        #d_sum = 0.0
        #thrs = [1,2,4,8,16]
        #sx_ = W / 256.0
        #sy_ = H / 256.0
        #sc_py = np.array([sx_, sy_]).reshape([1,1,2])
        #sc_pt = torch.from_numpy(sc_py).float().cuda()
        #for thr in thrs:
        #    # note we exclude timestep0 from this eval
        #    d_ = (torch.norm(trajs_e[:,35:64]/sc_pt - trajs_g[:,35:64]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        #    d_ = utils.basic.reduce_masked_mean(d_, valids[:,35:64]).item()*100.0
        #    d_sum += d_
        #    window_metrics['d_%d' % thr] = d_
        #d_avg = d_sum / len(thrs)
        #if cur_frame==9*(S_max-1):
        #    print(window_metrics)
        #break
        #if end_frame==71:
        #    window_metrics['d_avg'].append(d_avg)

        
        if end_frame >= S:
            done = True
        else:
            cur_frame = cur_frame + S_max - 1

        #if end_frame==71:
        #    done=True
    #k = trajs_e[:,:,:,0]<0
    ##print(k.shape)
    #trajs_e[trajs_e[:,:,:,0]<0]=0
    #trajs_e[trajs_e[:,:,:,0]>W-1]=W+5
    #trajs_e[trajs_e[:,:,:,1]<0]=0
    #trajs_e[trajs_e[:,:,:,1]>H-1]=H+5
    

    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    
    metrics['d_avg'] = d_avg
    #print(trajs_e[:,::10,::4])
    #print(trajs_g[:,::10,::4])
    sur_thr = 50
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0
    
    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True)
    


    #idx_seq = np.arange(0, S)
    #rgb_paths_seq = [os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (idx)) for idx in idx_seq]
    #rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths_seq]
    #rgbs = [rgb[:,:,::-1] for rgb in rgbs]
    #rgbs = [cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    #rgb_seq = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
    #rgb_seq = rgb_seq.unsqueeze(0)
    #print(rgb_seq.shape)
    #
    #prep_rgb0 = utils.improc.preprocess_color(rgb_seq[0:1])
    ##print(prep_rgb0.shape)
    #print("start giff generation")
    #giff_pred = sw.summ_traj2ds_on_rgbs('0_inputs/trajs_g_on_rgbs2', trajs_e[0:1][::10], prep_rgb0[::10],valids=valids[0:1][::10], frame_ids=list(range(0,S,10)), only_return=False)
    ##giff_gt = sw.summ_traj2ds_on_rgbs('0_inputs/trajs_g_on_rgbs2', trajs_g[0:1], prep_rgb0, valids=valids[0:1], frame_ids=list(range(0,S)), only_return=False)
    #print(giff_pred.shape)
    #make_video(seq.split('/')[-2],giff_pred, pred=True)
    ##make_video(seq.split('/')[-2],giff_gt, pred=False)

    #rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgb0, valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
    #giff = sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg,  only_return=True)
    
    metrics['median_l2'] = median_l2.mean().item()

    if sw is not None and sw.save_this:
        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgb0, valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg)
        
    return metrics, 0#sum(window_metrics['d_avg'])/len(window_metrics['d_avg'])


def test_on_fullseq_overlap(model, d, sw, iters=8, S_max=8, image_size=(384,512)):
    metrics = {}
    window_metrics = {}
    window_metrics['d_avg'] = []

    seq = str(d['seq'][0])
    print('seq', seq.split("/"))
    trajs_g = d['trajs'].cuda().float()[:,:,:,:] # B,S,N,2
    #trajs_g = trajs_g[motion]
    #print("GT Traj", trajs_g[:,10,:5])
    visibs = d['visibs'].cuda().float()[:,:,:] # B,S,N
    valids = d['valids'].cuda().float()[:,:,:] # B,S,N
    #valids = valids*visibs

    B, S, N, D = trajs_g.shape
    assert(D==2)
    assert(B==1)
    print('this video is %d frames long' % S)

    # load one to check H,W
    rgb_path0 = os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (0))
    rgb0_bak = cv2.imread(rgb_path0)
    H_bak, W_bak = rgb0_bak.shape[:2]
    H, W = image_size
    sy = H/H_bak
    sx = W/W_bak
    trajs_g[:,:,:,0] *= sx
    trajs_g[:,:,:,1] *= sy
    rgb0_bak = cv2.resize(rgb0_bak, (W, H), interpolation=cv2.INTER_LINEAR)
    rgb0_bak = torch.from_numpy(rgb0_bak[:,:,::-1].copy()).permute(2,0,1) # 3,H,W
    rgb0_bak = rgb0_bak.unsqueeze(0).to(trajs_g.device) # 1,3,H,W

    if sw is not None and sw.save_this:
        prep_rgb0 = utils.improc.preprocess_color(rgb0_bak)
        sw.summ_traj2ds_on_rgb('0_inputs/trajs_g_on_rgb0', trajs_g[0:1], prep_rgb0, valids=valids[0:1], cmap='winter', linewidth=1)
        
    # zero-vel init
    trajs_e = trajs_g[:,0].repeat(1,S,1,1)
    
    cur_frame = 0
    done = False
    feat_init = None
    motion_feat=None
    while not done:
        end_frame = cur_frame + S_max
        

        if end_frame > S:
            diff = end_frame-S
            end_frame = end_frame-diff
            cur_frame = max(cur_frame-diff,0)
        print('working on subseq %d:%d' % (cur_frame, end_frame))

        traj_seq = trajs_e[:, cur_frame:end_frame]
        traj_seq[:,:] = traj_seq[:,0]

        idx_seq = np.arange(cur_frame, end_frame)
        rgb_paths_seq = [os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (idx)) for idx in idx_seq]
        rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths_seq]
        rgbs = [rgb[:,:,::-1] for rgb in rgbs] # BGR->RGB
        H_load, W_load = rgbs[0].shape[:2]
        assert(H_load==H_bak and W_load==W_bak)
        rgbs = [cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        rgb_seq = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
        rgb_seq = rgb_seq.unsqueeze(0).to(traj_seq.device) # 1,S,3,H,W
        S_local = rgb_seq.shape[1]

        if feat_init is not None:
            feat_init = [fi[:,:S_local] for fi in feat_init]
        #with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        preds, preds_anim, feat_init, _, delta_coord_itr1, delta_coord_itr2, motion_feat = model(traj_seq, rgb_seq, iters=iters, feat_init=feat_init, motion_feat = motion_feat)
        #print(delta_coord_itr1)
        delta_1.append(delta_coord_itr1)
        delta_2.append(delta_coord_itr2)
        if cur_frame==0:
            trajs_e[:, cur_frame:end_frame] = preds[-1][:,:S_local]
        else:
            trajs_e[:, cur_frame+S_max//2:end_frame] = preds[-1][:,S_max//2:S_local]
        trajs_e[:, end_frame:] = trajs_e[:, end_frame-1:end_frame] # update the future with new zero-vel
        #trajs_e[:, cur_frame+S_max//2 - 1:] = trajs_e[:,curr_frame+S_max//2-1 : curr_frame+S_max]

        #trajs_e[:, end_frame:] = trajs_g[:,end_frame-1:end_frame] #only to check window performance, change later

        # if sw is not None and sw.save_this:
        #     traj_seq_e = preds[-1]
        #     traj_seq_g = trajs_g[:,cur_frame:end_frame]
        #     valid_seq = valids[:,cur_frame:end_frame]
        #     prep_rgbs = utils.improc.preprocess_color(rgb_seq)
        #     gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
        #     gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', traj_seq_g, gray_rgbs[0:1].mean(dim=1), valids=valid_seq, cmap='winter', only_return=True))
        #     rgb_vis = []
        #     for tre in preds_anim:
        #         ate = torch.norm(tre - traj_seq_g, dim=-1) # B,S,N
        #         ate_all = utils.basic.reduce_masked_mean(ate, valid_seq, dim=[1,2]) # B
        #         rgb_vis.append(sw.summ_traj2ds_on_rgb('', tre[0:1], gt_rgb, valids=valid_seq, only_return=True, cmap='spring', frame_id=ate_all[0]))
        #     sw.summ_rgbs('3_test/animated_trajs_on_rgb_cur%02d' % cur_frame, rgb_vis)



        #Compute Windowed Metric
        d_sum = 0.0
        thrs = [1,2,4,8,16]
        sx_ = W / 256.0
        sy_ = H / 256.0
        sc_py = np.array([sx_, sy_]).reshape([1,1,2])
        sc_pt = torch.from_numpy(sc_py).float().cuda()
        for thr in thrs:
            # note we exclude timestep0 from this eval
            d_ = (torch.norm(trajs_e[:,cur_frame+S_max//2:end_frame]/sc_pt - trajs_g[:,cur_frame+S_max//2:end_frame]/sc_pt, dim=-1) < thr).float() # B,S-1,N
            d_ = utils.basic.reduce_masked_mean(d_, valids[:,cur_frame+S_max//2:end_frame]).item()*100.0
            d_sum += d_
            window_metrics['d_%d' % thr] = d_
        d_avg = d_sum / len(thrs)
        #if cur_frame==9*(S_max-1):
        #    print(window_metrics)
        #
        if end_frame==150:
            window_metrics['d_avg'].append(d_avg)

        
        if end_frame >= S:
            done = True
        else:
            cur_frame = cur_frame + S_max//2
        if end_frame==150:
            done=True

    d_sum = 0.0
    thrs = [1,2,4,8,16]
    sx_ = W / 256.0
    sy_ = H / 256.0
    sc_py = np.array([sx_, sy_]).reshape([1,1,2])
    sc_pt = torch.from_numpy(sc_py).float().cuda()
    for thr in thrs:
        # note we exclude timestep0 from this eval
        d_ = (torch.norm(trajs_e[:,1:]/sc_pt - trajs_g[:,1:]/sc_pt, dim=-1) < thr).float() # B,S-1,N
        d_ = utils.basic.reduce_masked_mean(d_, valids[:,1:]).item()*100.0
        d_sum += d_
        metrics['d_%d' % thr] = d_
    d_avg = d_sum / len(thrs)
    
    metrics['d_avg'] = d_avg

    sur_thr = 16
    dists = torch.norm(trajs_e/sc_pt - trajs_g/sc_pt, dim=-1) # B,S,N
    dist_ok = 1 - (dists > sur_thr).float() * valids # B,S,N
    survival = torch.cumprod(dist_ok, dim=1) # B,S,N
    metrics['survival'] = torch.mean(survival).item()*100.0
    
    # get the median l2 error for each trajectory
    dists_ = dists.permute(0,2,1).reshape(B*N,S)
    valids_ = valids.permute(0,2,1).reshape(B*N,S)
    median_l2 = utils.basic.reduce_masked_median(dists_, valids_, keep_batch=True)

    #print("length window", len(window_metrics['d_avg']))



    #idx_seq = np.arange(0, S)
    #rgb_paths_seq = [os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % (idx)) for idx in idx_seq]
    #rgbs = [cv2.imread(rgb_path) for rgb_path in rgb_paths_seq]
    #rgbs = [rgb[:,:,::-1] for rgb in rgbs]
    #rgbs = [cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
    #rgb_seq = torch.from_numpy(np.stack(rgbs, 0)).permute(0,3,1,2) # S,3,H,W
    #rgb_seq = rgb_seq.unsqueeze(0)
    #print(rgb_seq.shape)
    #
    #prep_rgb0 = utils.improc.preprocess_color(rgb_seq[0:1])
    ##print(prep_rgb0.shape)
    #print("start giff generation")
    #giff_pred = sw.summ_traj2ds_on_rgbs('0_inputs/trajs_g_on_rgbs2', trajs_e[0:1], prep_rgb0, valids=valids[0:1], frame_ids=list(range(0,S)), only_return=False)
    #giff_gt = sw.summ_traj2ds_on_rgbs('0_inputs/trajs_g_on_rgbs2', trajs_g[0:1], prep_rgb0, valids=valids[0:1], frame_ids=list(range(0,S)), only_return=False)
    #print(giff_pred.shape)
    #make_video(seq.split('/')[-2],giff_pred, pred=True)
    #make_video(seq.split('/')[-2],giff_gt, pred=False)

    #rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgb0, valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
    #giff = sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg,  only_return=True)
    
    metrics['median_l2'] = median_l2.mean().item()

    if sw is not None and sw.save_this:
        rgb0 = sw.summ_traj2ds_on_rgb('', trajs_g[0:1], prep_rgb0, valids=valids[0:1], cmap='winter', linewidth=2, only_return=True)
        sw.summ_traj2ds_on_rgb('2_outputs/trajs_e_on_rgb0', trajs_e[0:1], utils.improc.preprocess_color(rgb0), valids=valids[0:1], cmap='spring', linewidth=2, frame_id=d_avg)
        
    return metrics, sum(window_metrics['d_avg'])/len(window_metrics['d_avg'])

def main(
        dset='test', 
        B=1, # batchsize 
        S=36, # seqlen
        N=256, # number of points per clip
        stride=8, # spatial stride of the model
        iters=16, # inference steps of the model
        image_size=(512,896), # input resolution
        shuffle=False, # dataset shuffling
        log_freq=99, # how often to make image summaries
        max_iters=99, # how many samples to test
        log_dir='./logs_test_on_pod',
        dataset_location='/home/boote/PointOdyssey_v1/',
        init_dir='./pips_dist_scaled_best',
        device_ids=[0],
        n_pool=1000, # how long the running averages should be
):
    device = 'cuda:%d' % device_ids[0]
    #device='cpu'

    # the idea in this file is:
    # load a ckpt, and test it in pointodyssey,
    # tracking points from frame0 to the end.
    
    exp_name = 'pod00' # copy from dev repo
    exp_name = 'pod01' # pod test
    exp_name = 'pod02' # clean up
    exp_name = 'pod03' # survival thr 16
    exp_name = 'pod04' # refmodel repeat
    exp_name = 'pod05' # fix bug with d_ stats

    assert(B==1) # B>1 not implemented here
    assert(image_size[0] % 32 == 0)
    assert(image_size[1] % 32 == 0)
    
    
    # autogen a descriptive name
    model_name = "%d_%d" % (B,S)
    model_name += "_i%d" % (iters)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H%M%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    writer_x = SummaryWriter(log_dir + '/' + model_name + '/x', max_queue=10, flush_secs=60)

    dataset_x = PointOdysseyDataset(
        dataset_location=dataset_location,
        dset=dset,
        N=N,
        verbose=True,
    )
    dataloader_x = DataLoader(
        dataset_x,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True)
    iterloader_x = iter(dataloader_x)

    model = Pips(stride=stride).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    utils.misc.count_parameters(model)

    _ = saverloader.load(init_dir, model.module, step=000)
    model.eval()

    pools_x = create_pools(n_pool)
    window_d = []
    
    global_step = 0
    max_iters = min(max_iters, len(dataset_x))
    while global_step < max_iters:
        global_step += 1
        iter_start_time = time.time()
        with torch.no_grad():
            torch.cuda.empty_cache()
        sw_x = utils.improc.Summ_writer(
            writer=writer_x,
            global_step=global_step,
            log_freq=log_freq,
            fps=min(S,8),
            scalar_freq=1,
            just_gif=True)
        try:
            sample = next(iterloader_x)
        except StopIteration:
            iterloader_x = iter(dataloader_x)
            sample = next(iterloader_x)
        iter_rtime = time.time()-iter_start_time
        print(global_step)
        #if global_step<9:
        #    continue
        with torch.no_grad():
            metrics, window_avg= test_on_fullseq(model, sample, sw_x, iters=iters, S_max=S, image_size=image_size)
        for key in list(pools_x.keys()):
            if key in metrics:
                pools_x[key].update([metrics[key]])
                sw_x.summ_scalar('_/%s' % (key), pools_x[key].mean())
        iter_itime = time.time()-iter_start_time

        window_d.append(window_avg)
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; d_x %.1f; sur_x %.1f; med_x %.1f' % (
            model_name, global_step, max_iters, iter_rtime, iter_itime,
            pools_x['d_avg'].mean(), pools_x['survival'].mean(), pools_x['median_l2'].mean()))
    print(pools_x['d_1'].mean(), pools_x['d_2'].mean(), pools_x['d_4'].mean(), pools_x['d_8'].mean(), pools_x['d_16'].mean())
        
        #print("delta_1: {}, delta_2: {}".format(sum(delta_1)/len(delta_1), sum(delta_2)/len(delta_2)))
   # print("window len, window avg", len(window_d), sum(window_d)/len(window_d))
        #giff.save("{}.giff".format(global_step))
    #print("Feature diff vs motion for Frame 0 and Frame S-4") 
    #for key in data_per_itr_motion_4.keys():
    #    print("motion {}".format(key), np.nanmean(data_per_itr_motion_4[key]))
    #print("Feature diff vs motion for Frame 0 and Frame S-2")
    #for key in data_per_itr_motion.keys():
    #    print("motion {}".format(key), np.nanmean(data_per_itr_motion[key]))
    
    #print("Final delta_1: {}, delta_2: {}".format(sum(delta_1)/len(delta_1), sum(delta_2)/len(delta_2)))
    writer_x.close()

if __name__ == '__main__':
    Fire(main)
