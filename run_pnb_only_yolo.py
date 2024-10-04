import os
import time
import numpy as np
from yolov4.get_bounding_boxes import YoloModel
import argparse
import glob
from sklearn.linear_model import LogisticRegression
import torchvision as tv
from tqdm import tqdm
import random

def calc_yolo_dist(yolo_model, frame):
    bounding_boxes, classes = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())
    
    needle = None
    nerve = None

    for bb, class_pred in zip(bounding_boxes, classes):
        if class_pred == 0 and nerve is None:
            nerve = np.array(bb)
        elif class_pred == 2 and needle is None:
            needle = np.array(bb)
            
    if nerve is None or needle is None:
        return None

    return np.concatenate([nerve, needle])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/shared_data/bamc_pnb_data/revised_training_data', type=str)
    
    args = parser.parse_args()
        
    yolo_model = YoloModel('pnb')

    all_predictions = []

    all_files = glob.glob(pathname=os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)
    
    all_data = []
    
    print('Building dataset...')

    for f in tqdm(all_files):
        vc = tv.io.read_video(f)[0].permute(3, 0, 1, 2)
        
        x = None
        i = -1
        while x is None and i >= -5:
            x = calc_yolo_dist(yolo_model, vc[:, i, :, :])
            i -= 1
            
        if x is None:
            continue
            
        y = 1 if f.split('/')[-3] == 'Positives' else 0
        
        all_data.append((x, y))
        
    random.shuffle(all_data)
    
    split = int(len(all_data) * 0.8)
    
    train_data = all_data[:split]
    test_data = all_data[split:]
    
    print('Loaded {} train examples.'.format(len(train_data)))
    print('Loaded {} test examples.'.format(len(test_data)))
    
    train_x = [ex[0] for ex in train_data]
    train_y = [ex[1] for ex in train_data]
    
    test_x = [ex[0] for ex in test_data]
    test_y = [ex[1] for ex in test_data]
    
    print('Fitting to data...')
    
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(train_x, train_y)
    
    print('Evaluating model...')
    
    score = clf.score(test_x, test_y)
    
    print(score)