import numpy as np
import torchvision
import torch
from sklearn.model_selection import train_test_split
from sparse_coding_torch.utils import MinMaxScaler
from sparse_coding_torch.onsd.video_loader import get_participants, ONSDGoodFramesLoader, ONSDAllFramesLoader, FrameLoader, RegressionLoader
from sparse_coding_torch.utils import VideoGrayScaler
from typing import Sequence, Iterator
import csv
from sklearn.model_selection import train_test_split, GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold, StratifiedKFold, KFold, ShuffleSplit
    
def load_onsd_videos(batch_size, crop_size, yolo_model=None, mode=None, n_splits=None, do_regression=False):   
    video_path = "/shared_data/bamc_onsd_data/revised_extended_onsd_data"
    
    transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(1),
     MinMaxScaler(0, 255)
    ])
#     augment_transforms = torchvision.transforms.Compose(
#     [torchvision.transforms.RandomRotation(45),
#      torchvision.transforms.RandomHorizontalFlip(0.5),
#      torchvision.transforms.RandomAdjustSharpness(0.05)
     
#     ])
    if do_regression:
        dataset = ONSDGoodFramesLoader(video_path, crop_size[1], crop_size[0], transform=transforms, yolo_model=yolo_model)
    else:
        dataset = ONSDAllFramesLoader(video_path, crop_size[1], crop_size[0], transform=transforms, yolo_model=yolo_model)
    
    targets = dataset.get_labels()
    
    if mode == 'leave_one_out':
        gss = LeaveOneGroupOut()

        groups = get_participants(dataset.get_filenames())
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    elif mode == 'all_train':
        train_idx = np.arange(len(targets))
        test_idx = None
        
        return [(train_idx, test_idx)], dataset
    elif mode == 'k_fold':
        gss = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)

        groups = get_participants(dataset.get_filenames())
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    elif mode == 'custom':
        splits = []
        splits.append(['64', '129', '250', '117'])
        splits.append(['163', '189', '3', '50', '106', '158'])
        splits.append(['108', '95', '75', '51', '36', '241', '168'])
        
        test_train_idxs = []
        
        for split in splits:
            test_idx = []
            train_idx = []

            for i, participant in enumerate(get_participants(dataset.get_filenames())):
                if participant in split:
                    test_idx.append(i)
                else:
                    train_idx.append(i)
            
            test_train_idxs.append((train_idx, test_idx))
                
        return test_train_idxs, dataset
    else:
#         gss = ShuffleSplit(n_splits=n_splits, test_size=0.2)
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)

        groups = get_participants(dataset.get_filenames())
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
#         return gss.split(np.arange(len(targets)), targets), dataset
    
def load_onsd_frames(batch_size, input_size, mode=None, yolo_model=None):   
    video_path = "/shared_data/bamc_onsd_data/revised_onsd_data"
    
    transforms = torchvision.transforms.Compose(
    [
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize(input_size[:2])
    ])

    dataset = FrameLoader(video_path, input_size[1], input_size[0], transform=transforms, yolo_model=yolo_model)
    
    targets = dataset.get_labels()
    
    if mode == 'all_train':
        train_idx = np.arange(len(targets))
        test_idx = None
        
        return [(train_idx, test_idx)], dataset
    else:
#         gss = ShuffleSplit(n_splits=n_splits, test_size=0.2)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2)

        groups = get_participants(dataset.get_filenames())
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    
def load_onsd_regression(batch_size, input_size, mode=None, yolo_model=None):   
    csv_path = "sparse_coding_torch/onsd/individual_frames_cleaned/onsd_labeled_widths.csv"
    
    transforms = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(1),
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize(input_size[:2])
    ])

    dataset = RegressionLoader(csv_path, input_size[1], input_size[0], transform=transforms, yolo_model=yolo_model)
    
    targets = dataset.get_labels()
    
    if mode == 'all_train':
        train_idx = np.arange(len(targets))
        test_idx = None
        
        return [(train_idx, test_idx)], dataset
    else:
#         gss = ShuffleSplit(test_size=0.2)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2)

        groups = dataset.get_filenames()
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
#         return gss.split(np.arange(len(targets)), targets), dataset