import numpy as np
import torchvision
import torch
from sklearn.model_selection import train_test_split
from sparse_coding_torch.utils import MinMaxScaler, VideoGrayScaler
from sparse_coding_torch.ptx.video_loader import YoloClipLoader, get_ptx_participants, COVID19Loader
import csv
from sklearn.model_selection import train_test_split, GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold, StratifiedKFold, KFold, ShuffleSplit

def load_yolo_clips(batch_size, mode, num_clips=1, num_positives=100, device=None, n_splits=None, sparse_model=None, whole_video=False, positive_videos=None):   
    video_path = "/shared_data/YOLO_Updated_PL_Model_Results/"

    video_to_participant = get_ptx_participants()
    
    transforms = torchvision.transforms.Compose(
    [VideoGrayScaler(),
#      MinMaxScaler(0, 255),
     torchvision.transforms.Normalize((0.2592,), (0.1251,)),
    ])
#     augment_transforms = torchvision.transforms.Compose(
#     [torchvision.transforms.RandomRotation(45),
#      torchvision.transforms.RandomHorizontalFlip(),
#      torchvision.transforms.CenterCrop((100, 200))
#     ])

    dataset = YoloClipLoader(video_path, num_clips=num_clips, num_positives=num_positives, positive_videos=positive_videos, transform=transforms, sparse_model=sparse_model, device=device)
    
    targets = dataset.get_labels()
    
    if mode == 'leave_one_out':
        gss = LeaveOneGroupOut()

#         groups = [v for v in dataset.get_filenames()]
        groups = [video_to_participant[v.lower().replace('_clean', '')] for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    elif mode == 'all_train':
        train_idx = np.arange(len(targets))
#         train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                sampler=train_sampler)
#         test_loader = None
        
        return [(train_idx, None)], dataset
    elif mode == 'k_fold':
        gss = StratifiedGroupKFold(n_splits=n_splits)

        groups = [video_to_participant[v.lower().replace('_clean', '')] for v in dataset.get_filenames()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    else:
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)

        groups = [video_to_participant[v.lower().replace('_clean', '')] for v in dataset.get_filenames()]
        
        return list(gss.split(np.arange(len(targets)), targets, groups)), dataset
    
def load_covid_clips(batch_size, mode, clip_width, clip_height, clip_depth, n_splits=None):   
    video_path = "/home/dwh48@drexel.edu/covid19_ultrasound/data/pocus_videos"
    
    transforms = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize((clip_height, clip_width))
    ])
#     augment_transforms = torchvision.transforms.Compose(
#     [torchvision.transforms.RandomRotation(45),
#      torchvision.transforms.RandomHorizontalFlip(),
#      torchvision.transforms.Resize((clip_height, clip_width))
#     ])

    dataset = COVID19Loader(video_path, clip_depth, transform=transforms)
    
    targets = dataset.get_labels()
    
    if mode == 'leave_one_out':
        gss = LeaveOneGroupOut()

        groups = [v for v in dataset.get_groups()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    elif mode == 'all_train':
        train_idx = np.arange(len(targets))
        
        return [(train_idx, None)], dataset
    elif mode == 'k_fold':
        gss = StratifiedGroupKFold(n_splits=n_splits)

        groups = [v for v in dataset.get_groups()]
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    else:
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)

        groups = [v for v in dataset.get_groups()]
        
        train_idx, test_idx = list(gss.split(np.arange(len(targets)), targets, groups))[0]
        
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
        
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=test_sampler)
        
        return train_loader, test_loader, dataset