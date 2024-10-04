import numpy as np
import torchvision
import torch
from sklearn.model_selection import train_test_split
from sparse_coding_torch.utils import MinMaxScaler
from sparse_coding_torch.pnb.video_loader import PNBLoader, get_participants, NeedleLoader
from sparse_coding_torch.utils import VideoGrayScaler
from typing import Sequence, Iterator
import csv
from sklearn.model_selection import train_test_split, GroupShuffleSplit, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold, StratifiedKFold, KFold, ShuffleSplit
    
def get_sample_weights(train_idx, dataset):
    dataset = list(dataset)

    num_positive = len([clip[0] for clip in dataset if clip[0] == 'Positives'])
    negative_weight = num_positive / len(dataset)
    positive_weight = 1.0 - negative_weight
    
    weights = []
    for idx in train_idx:
        label = dataset[idx][0]
        if label == 'Positives':
            weights.append(positive_weight)
        elif label == 'Negatives':
            weights.append(negative_weight)
        else:
            raise Exception('Sampler encountered invalid label')
    
    return weights

class SubsetWeightedRandomSampler(torch.utils.data.Sampler[int]):
    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], indicies: Sequence[int],
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.indicies = indicies
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, len(self.indicies), self.replacement, generator=self.generator)
        for i in rand_tensor:
            yield self.indicies[i]

    def __len__(self) -> int:
        return len(self.indicies)
    
def load_pnb_videos(yolo_model, batch_size, input_size, crop_size=None, mode=None, classify_mode=False, balance_classes=False, device=None, n_splits=None, sparse_model=None, frames_to_skip=1):   
    video_path = "/shared_data/bamc_pnb_data/revised_training_data"
#     video_path = '/home/dwh48@drexel.edu/pnb_videos_for_testing/train'
#     video_path = '/home/dwh48@drexel.edu/special_splits/train'

    if not crop_size:
        crop_size = input_size
    
    transforms = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize(input_size[:2])
    ])
#     augment_transforms = torchvision.transforms.Compose(
# #     [torchvision.transforms.Resize(input_size[:2]),
# #     [torchvision.transforms.RandomRotation(15),
# #      torchvision.transforms.RandomHorizontalFlip(),
# #      torchvision.transforms.RandomVerticalFlip(),
#      [torchvision.transforms.ColorJitter(brightness=0.02),     
#      torchvision.transforms.RandomAdjustSharpness(0, p=0.15),
#      torchvision.transforms.RandomAffine(degrees=0, translate=(0.01, 0))
# #      torchvision.transforms.CenterCrop((100, 200))
# #      torchvision.transforms.Resize(input_size[:2])
#     ])
    dataset = PNBLoader(yolo_model, video_path, crop_size[1], crop_size[0], crop_size[2], classify_mode, balance_classes=balance_classes, num_frames=5, transform=transforms, frames_to_skip=frames_to_skip)
    
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
    else:
#         gss = ShuffleSplit(n_splits=n_splits, test_size=0.2)
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2)

        groups = get_participants(dataset.get_filenames())
        
#         train_idx, test_idx = list(gss.split(np.arange(len(targets)), targets, groups))[0]
#         train_idx, test_idx = list(gss.split(np.arange(len(targets)), targets))[0]

        
#         train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         train_sampler = SubsetWeightedRandomSampler(get_sample_weights(train_idx, dataset), train_idx, replacement=True)
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                sampler=train_sampler)
        
#         test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
#         test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                sampler=test_sampler)
        
        return gss.split(np.arange(len(targets)), targets, groups), dataset
    
def load_needle_clips(batch_size, input_size):   
    video_path = "/shared_data/bamc_pnb_data/needle_data/non_needle"
    
    transforms = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize(input_size[:2])
    ])

    dataset = NeedleLoader(video_path, transform=transforms, augmentation=None)

    train_idx = np.arange(len(dataset))
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
    test_loader = None

    return train_loader, test_loader, dataset