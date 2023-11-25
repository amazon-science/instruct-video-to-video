import os
import random
from einops import rearrange
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import omegaconf

class SingleVideoDataset(Dataset):
    def __init__(
        self, 
        video_file, 
        video_description,
        sampling_fps=24, 
        frame_gap=0, 
        num_frames=2,
        output_size=(512, 512), 
        mode='train'
    ):
        self.video_file = video_file
        self.video_id = os.path.splitext(os.path.basename(video_file))[0]
        self.sampling_fps = sampling_fps
        self.frame_gap = frame_gap
        self.description = video_description
        self.output_size = output_size
        self.mode = mode
        self.num_frames = num_frames

        cap = cv2.VideoCapture(video_file)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if self.sampling_fps is not None:
            if isinstance(self.sampling_fps, int):
                sampling_fps = self.sampling_fps
            elif isinstance(self.sampling_fps, list) or isinstance(self.sampling_fps, omegaconf.listconfig.ListConfig):
                sampling_fps = random.choice(self.sampling_fps)
                assert isinstance(sampling_fps, int)
            else:
                raise ValueError(f'sampling_fps should be int or list of int, got {self.sampling_fps}')
            sampling_fps = int(min(sampling_fps, video_fps))
            frame_gap = max(0, int(video_fps / sampling_fps))
        else:
            sampling_fps = video_fps // (1 + self.frame_gap)
            frame_gap = self.frame_gap

        self.frame_gap = frame_gap
        self.sampling_fps = sampling_fps

        cap.release()

        # print(num_frames, frame_gap, self.num_frames)
        self.num_frames = min(self.num_frames, num_frames // frame_gap) # this is number of sampling frames
        self.total_possible_starting_frames = max(0, num_frames - (frame_gap * (self.num_frames - 1)))

    def __len__(self):
        return self.total_possible_starting_frames

    def __getitem__(self, index):
        video_file = self.video_file

        cap = cv2.VideoCapture(video_file)
        video_fps = round(cap.get(cv2.CAP_PROP_FPS))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
        sampling_fps = self.sampling_fps
        frame_gap = self.frame_gap
        
        frames = []
        if num_frames > 1 + frame_gap:
            first_frame_index = index
            for i in range(self.num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index + i * frame_gap)
                ret, frame = cap.read()

                # Convert BGR to RGB and resize frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame is not None else None

                if frame is not None:
                    height, width, _ = frame.shape
                    aspect_ratio = width / height
                    target_width = int(self.output_size[0] * aspect_ratio)
                    frame = rearrange(torch.from_numpy(frame), 'h w c -> c h w')
                    frame = F.resize(frame, (self.output_size[1], target_width), antialias=True)

                    if target_width > self.output_size[1]:
                        margin = (target_width - self.output_size[1]) // 2
                        frame = F.crop(frame, 0, margin, self.output_size[1], self.output_size[0])
                    else:
                        margin = (self.output_size[1] - target_width) // 2
                        frame = F.pad(frame, (margin, 0), 0, 'constant')
                    frame = (frame / 127.5) - 1.0

                frames.append(frame)
        else:
            for i in range(self.num_frames):
                frames.append(None)

        cap.release()

        # Stack frames
        frames = [frame for frame in frames if frame is not None]
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        frames = frames[:self.num_frames]
        frames = torch.stack(frames, dim=0)

        caption = self.description

        res = {
            'frames': frames,
            'video_id': self.video_id,
            'text': caption,
            'fps': torch.tensor(sampling_fps),
        }
        return res