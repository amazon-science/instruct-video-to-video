import csv
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class LoveuTgveVideoDataset(Dataset):
    def __init__(self, root_dir, image_size=(480, 480)):
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1,1] for each channel
        ])

        csv_file = os.path.join(root_dir, 'LOVEU-TGVE-2023_Dataset.csv')
        self.data = {}
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader, None)  # skip the headers
            for row in reader:
                if len(row[0]) == 0:
                    continue
                if row[0].endswith('Videos:'):
                    dataset_type = row[0].split(' ')[0]
                    if dataset_type == 'DAVIS':
                        self.source_folder = dataset_type + '_480p/480p_videos'
                    else:
                        self.source_folder = dataset_type.lower() + '_480p/480p_videos'
                elif len(row) > 1:
                    video_name = row[0]
                    self.data[video_name] = {
                        'video_name': video_name,
                        'original': row[1],
                        'style': row[2],
                        'object': row[3],
                        'background': row[4],
                        'multiple': row[5],
                        'source_folder': self.source_folder,
                    }

    def load_frames(self, video_name, source_folder):
        video_path = os.path.join(self.root_dir, source_folder, f'{video_name}.mp4')
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.image_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)  # convert to PyTorch tensor and normalize
                frames.append(frame)
            else:
                break
        cap.release()
        return torch.stack(frames, dim=0)

    def load_fps(self, video_name, source_folder):
        video_path = os.path.join(self.root_dir, source_folder, f'{video_name}.mp4')
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            video_name = idx
        else:
            video_name = list(self.data.keys())[idx]
        
        # Load frames and fps when the dataset is called
        source_folder = self.data[video_name]['source_folder']
        frames = self.load_frames(video_name, source_folder)
        fps = self.load_fps(video_name, source_folder)
        item = self.data[video_name].copy()
        item['frames'] = frames
        item['fps'] = fps

        return item