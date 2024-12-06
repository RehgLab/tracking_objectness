from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from torch._C import dtype, set_flush_denormal
#import utils.geom
import glob
import cv2
from pathlib import Path
import sys
import albumentations as A
from functools import partial
from torch.utils.data import Dataset, DataLoader


def augment_video(augmenter, **kwargs):
    assert isinstance(augmenter, A.ReplayCompose)
    keys = kwargs.keys()
    for i in range(len(next(iter(kwargs.values())))):
        data = augmenter(**{
            key: kwargs[key][i] if key not in ['bboxes', 'keypoints'] else [kwargs[key][i]] for key in keys
        })
        if i == 0:
            augmenter = partial(A.ReplayCompose.replay, data['replay'])
        for key in keys:
            if key == 'bboxes':
                kwargs[key][i] = np.array(data[key]).reshape(4)
            elif key == 'keypoints':
                kwargs[key][i] = np.array(data[key]).reshape(2)
            else:
                kwargs[key][i] = data[key]

def read_mp4(fn):
    vidcap = cv2.VideoCapture(fn)
    frames = []
    while(vidcap.isOpened()):
        ret, frame = vidcap.read()
        if ret == False:
            break
        frames.append(frame)
    vidcap.release()
    return frames



class ExportDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='./pod_export',
                 dataset_version='aa',
                 S=36,
                 N=128,
                 crop_size=(384,512), 
                 use_augs=False,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.N = N
        self.H, self.W = crop_size
        self.crop_size = crop_size
        self.use_augs = use_augs
        
        self.dataset_location = Path('%s/%s' % (self.dataset_location, dataset_version))

        folder_names = self.dataset_location.glob('*/')
        folder_names = [fn for fn in folder_names]
        folder_names = sorted(list(folder_names))
        print('found %d folders in %s' % (len(folder_names), self.dataset_location))

        self.all_folder_names = []
        rgbs = read_mp4(str(folder_names[0] / 'rgb.mp4'))
        rgbs = np.stack(rgbs, axis=0)[:8] # S,H,W,3
        S_local,H,W,C = rgbs.shape
        assert(H==self.H)
        assert(W==self.W)
        assert(S_local==self.S)
        
        self.all_folder_names = folder_names

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.2),
            A.RGBShift(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        ], p=0.8)
        
    
    def __getitem__(self, index):
        folder = self.all_folder_names[index]
        # print('folder', folder)
        
        rgbs = read_mp4(str(folder / 'rgb.mp4'))

        if len(rgbs)==0:
            print('corrupted mp4 in %s; returning fake' % folder)
            fake_sample = {
                'rgbs': np.zeros((self.S,3,self.H,self.W), dtype=np.uint8), 
                'track_g': np.zeros((self.S,self.N,4), dtype=np.float32)
            }
            return fake_sample
        rgbs = np.stack(rgbs, axis=0)[:8] # S,H,W,3

        d = dict(np.load(folder / 'track.npz', allow_pickle=True))
        track = d['track_g'][:8]

        H,W,C = rgbs[0].shape
        S,N,D = track.shape

        assert(N >= self.N)
        assert(H==self.H)
        assert(W==self.W)
        assert(S==self.S)
        assert(D==4) # xy, vis, valid
        assert(C==3)
        
        if N > self.N:
            inds = np.random.choice(N, self.N, replace=False)
            track = track[:,inds]
        
        if self.use_augs:
            augment_video(self.color_augmenter, image=rgbs)

        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        return {
            'rgbs': rgbs,
            'track_g': track,
        }

    def __len__(self):
        return len(self.all_folder_names)
    

def read_frames(dir, type="rgb"):
    frames = []
    for file in sorted(os.listdir(dir)):
        frame = cv2.imread(os.path.join(dir, file))
        if type=="masks":
            frames.append(frame[:,:,0])
        else:
            frames.append(frame)

    return frames
def read_masks_npz(dir):
    masks = []
    #print("reading cropformer")
    for f in sorted(os.listdir(dir)):
        frame = dict(np.load(os.path.join(dir,f), allow_pickle=True))['mask']
        masks.append(frame)

    return masks
    
import time
class ExportDataset_Masks(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='./pod_export',
                 dataset_version='aa',
                 S=36,
                 N=128,
                 crop_size=(384,512), 
                 use_augs=False,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.N = N
        self.H, self.W = crop_size
        self.crop_size = crop_size
        self.use_augs = use_augs
        
        self.dataset_location = Path('%s/%s' % (self.dataset_location, dataset_version))

        folder_names = self.dataset_location.glob('*/')
        folder_names = [fn for fn in folder_names]#[::300]
        folder_names = sorted(list(folder_names))
        print('found %d folders in %s' % (len(folder_names), self.dataset_location))

        self.all_folder_names = []
        #rgbs = read_mp4(str(folder_names[0] / 'rgb.mp4'))
        rgbs = read_frames(os.path.join(folder_names[0], "rgbs"), type="rgbs")
        rgbs = np.stack(rgbs, axis=0)[:24] # S,H,W,3
        S_local,H,W,C = rgbs.shape
        assert(H==self.H)
        assert(W==self.W)
        assert(S_local==self.S)
        
        self.all_folder_names = folder_names

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.2),
            A.RGBShift(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        ], p=0.8)
        
    
    def __getitem__(self, index):
        start = time.time()
        folder = self.all_folder_names[index]


        #print('folder', folder)
        
        #rgbs = read_mp4(str(folder / 'rgb.mp4'))
        rgbs = read_frames(os.path.join(folder,"rgbs"), "rgbs")
        #masks = read_frames(os.path.join(folder, "masks"), "masks")
        masks = read_masks_npz(os.path.join(folder, "cropformer_masks"))


        if len(rgbs)==0 or len(masks)<24:
            print('corrupted mp4 in %s; returning fake' % folder)
            fake_sample = {
                'rgbs': np.zeros((self.S,3,self.H,self.W), dtype=np.uint8), 
                'track_g': np.zeros((self.S,self.N,4), dtype=np.float32),
                'masks': np.zeros((self.S,self.H,self.W), dtype=np.uint8)
            }
            return fake_sample
        rgbs = np.stack(rgbs, axis=0)[:24] # S,H,W,3
        masks = np.stack(masks, axis=0)[:24] #S,H,W

        d = dict(np.load(folder / 'track.npz', allow_pickle=True))
        track = d['track_g'][:24]

        H,W,C = rgbs[0].shape
        S,N,D = track.shape

        assert(N >= self.N)
        assert(H==self.H)
        assert(W==self.W)
        assert(S==self.S)
        assert(D==4) # xy, vis, valid
        assert(C==3)
        
        if N > self.N:
            inds = np.random.choice(N, self.N, replace=False)
            track = track[:,inds]
        
        if self.use_augs:
            augment_video(self.color_augmenter, image=rgbs)

        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        #print("time", time.time()-start)

        return {
            'rgbs': rgbs,
            'track_g': track,
            'masks': masks
        }

    def __len__(self):
        return len(self.all_folder_names)
    


class ExportDataset_Mask_Contours(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='./pod_export',
                 dataset_version='aa',
                 S=36,
                 N=128,
                 crop_size=(384,512), 
                 use_augs=False,
    ):
        print('loading export...')

        self.dataset_location = dataset_location
        self.S = S
        self.N = N
        self.H, self.W = crop_size
        self.crop_size = crop_size
        self.use_augs = use_augs
        self.contour_path = "/media/boote/ec537bcf-bb0a-4b9a-9c71-be6183baaccf/simple_contours_gt/"
        
        self.dataset_location = Path('%s/%s' % (self.dataset_location, dataset_version))

        folder_names = self.dataset_location.glob('*/')
        folder_names = [fn for fn in folder_names]#[::300]
        folder_names = sorted(list(folder_names))
        print('found %d folders in %s' % (len(folder_names), self.dataset_location))

        self.all_folder_names = []
        #rgbs = read_mp4(str(folder_names[0] / 'rgb.mp4'))
        rgbs = read_frames(os.path.join(folder_names[0], "rgbs"), type="rgbs")
        rgbs = np.stack(rgbs, axis=0)[:24] # S,H,W,3
        S_local,H,W,C = rgbs.shape
        assert(H==self.H)
        assert(W==self.W)
        assert(S_local==self.S)
        
        self.all_folder_names = folder_names
        self.all_contour_files = sorted(list([file for file in os.listdir(self.contour_path)]))

        self.color_augmenter = A.ReplayCompose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.2),
            A.RGBShift(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
        ], p=0.8)
        
    
    def __getitem__(self, index):
        start = time.time()
        folder = self.all_folder_names[index]


        # print('folder', folder)
        
        #rgbs = read_mp4(str(folder / 'rgb.mp4'))
        #print(folder)
        contour_file = self.all_contour_files[index]
        #print(folder, contour_file)
        
        rgbs = read_frames(os.path.join(folder,"rgbs"), "rgbs")
        #masks = read_frames(os.path.join(folder, "masks"), "masks")
        masks = read_masks_npz(os.path.join(folder, "cropformer_masks"))
        contours = np.load(os.path.join(self.contour_path, contour_file) ,allow_pickle=True)['arr_0'][:24]


        if len(rgbs)==0:
            print('corrupted mp4 in %s; returning fake' % folder)
            fake_sample = {
                'rgbs': np.zeros((self.S,3,self.H,self.W), dtype=np.uint8), 
                'track_g': np.zeros((self.S,self.N,4), dtype=np.float32),
                'masks': np.zeros((self.S,self.H,self.W), dtype=np.uint8)
            }
            return fake_sample
        rgbs = np.stack(rgbs, axis=0)[:24] # S,H,W,3
        masks = np.stack(masks, axis=0)[:24] #S,H,W

        d = dict(np.load(folder / 'track.npz', allow_pickle=True))
        track = d['track_g'][:24]

        H,W,C = rgbs[0].shape
        S,N,D = track.shape

        assert(N >= self.N)
        assert(H==self.H)
        assert(W==self.W)
        assert(S==self.S)
        assert(D==4) # xy, vis, valid
        assert(C==3)
        
        if N > self.N:
            inds = np.random.choice(N, self.N, replace=False)
            track = track[:,inds]
        
        if self.use_augs:
            augment_video(self.color_augmenter, image=rgbs)

        rgbs = rgbs.transpose(0,3,1,2)
        rgbs = rgbs[:,::-1].copy() # BGR->RGB

        #print("time", time.time()-start)
        #print(contours.shape)

        return {
            'rgbs': rgbs,
            'track_g': track,
            'masks': masks, 
            'contours': contours
        }

    def __len__(self):
        return len(self.all_folder_names)

def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


import cv2

if __name__=="__main__":



    dataset_location='/home/boote/TrackingTRI/pod_export/masked_clips' # where we exported the data
    dataset_version='ae_36_128_384x512'

    dataset_t = ExportDataset_Mask_Contours(
        dataset_location=dataset_location,
        dataset_version=dataset_version,
        S=24,
        crop_size=(384,512),
        N=20,
        use_augs=True)

    #finetune to full size
    #dataset_t = PointOdysseyDataset(
    #    dataset_location="/home/boote/PointOdyssey/",
    #    dset='train',
    #    use_augs=use_augs,
    #    crop_size=crop_size,
    #    verbose=True,
    #    S=S,
    #    N=N
    #)
    dataloader_t = DataLoader(
        dataset_t,
        batch_size=1,
        shuffle=True,
        num_workers=6,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    iterloader_t = iter(dataloader_t)
    sample = next(iterloader_t)
    rgbs = sample["rgbs"]
    print(rgbs[0,0].shape)
    track_g = sample['track_g']
    trajs_g = track_g[:,:,:,:2]
    B,S,N,_ = trajs_g.shape
    contours = sample['contours']
    print(contours.shape)
    for j in range(8):
        img = rgbs[0,j].permute(1,2,0).numpy()
        print(img.shape)
        img = img.astype(np.uint8).copy()
        for i in range(N):
            pt = trajs_g[0,j,i].numpy()
            #print(pt)
            #break
            cv2.circle(img, (int(pt[0]),int(pt[1])), radius=5, color=(0,0,255), thickness=-1)
            if i==20:
                break
        cv2.imshow("frame{}".format(j), img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

