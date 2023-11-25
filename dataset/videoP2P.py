import os
import json
import jsonlines
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class VideoPromptToPrompt(Dataset):
    def __init__(self, root_dirs, num_frames=8):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.image_folders = []
        for root_dir in self.root_dirs:
            image_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
            self.num_frames = num_frames

            for f in image_folders:
                if os.path.exists(os.path.join(root_dir, f, 'image')) and os.path.exists(os.path.join(root_dir, f, 'metadata.jsonl')) and os.path.exists(os.path.join(root_dir, f, 'prompt.json')):
                    self.image_folders.append(
                        os.path.join(root_dir, f)
                    )

    def __len__(self):
        return len(self.image_folders)

    def numpy_to_tensor(self, image):
        image = torch.from_numpy(image.transpose((0, 3, 1, 2))).to(torch.float32)
        return image * 2 - 1

    def __getitem__(self, idx):
        folder = self.image_folders[idx]
        with jsonlines.open(os.path.join(folder, 'metadata.jsonl')) as reader:
            seeds = [obj['seed'] for obj in reader if (obj['sim_dir'] > 0.2 and obj['sim_0'] > 0.2 and obj['sim_1'] > 0.2 and obj['sim_image'] > 0.5)]
        seed = np.random.choice(seeds)

        # Load prompt.json
        with open(os.path.join(folder, 'prompt.json'), 'r') as f:
            prompt = json.load(f)

        start_idx = np.random.randint(0, 16 - self.num_frames)
        end_idx = start_idx + self.num_frames
        
        input_images = np.array([self.load_image(os.path.join(folder, 'image', f'{seed}_0_{img_idx:04d}.jpg')) for img_idx in range(start_idx, end_idx)])
        edited_images = np.array([self.load_image(os.path.join(folder, 'image', f'{seed}_1_{img_idx:04d}.jpg')) for img_idx in range(start_idx, end_idx)])

        return {
            'input_video': self.numpy_to_tensor(input_images), 
            'edited_video': self.numpy_to_tensor(edited_images),
            'input_prompt': prompt['input'], 
            'output_prompt': prompt['output'], 
            'edit_prompt': prompt['edit']
        }

    @staticmethod
    def load_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalization
        return img


class VideoPromptToPromptMotionAug(VideoPromptToPrompt):
    def __init__(self, *args, zoom_ratio=0.2, max_zoom=1.2, translation_ratio=0.3, translation_range=(0, 0.2), **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_ratio = zoom_ratio
        self.max_zoom = max_zoom
        self.translation_ratio = translation_ratio
        self.translation_range = translation_range

    def translation_crop(self, delta_h, delta_w, images):
        def center_crop(img, center_x, center_y, h, w):
            x_start = int(center_x - w / 2)
            x_end = int(x_start + w)
            y_start = int(center_y - h / 2)
            y_end = int(y_start + h)
            img = img[y_start: y_end, x_start: x_end]
            return img

        H, W = images.shape[1:3]
        crop_H = H - abs(delta_h)
        crop_W = W - abs(delta_w)

        if delta_h > 0:
            h_start = (H - delta_h) // 2
            h_end = h_start + delta_h
        else:
            h_end = H - (H + delta_h) // 2
            h_start = h_end + delta_h

        if delta_w > 0:
            w_start = (W - delta_w) // 2
            w_end = w_start + delta_w
        else:
            w_end = W - (W + delta_w) // 2
            w_start = w_end + delta_w

        center_xs = np.linspace(w_start, w_end, self.num_frames)
        center_ys = np.linspace(h_start, h_end, self.num_frames)

        if delta_h < 0:
            center_ys = center_ys[::-1]
        if delta_w < 0:
            center_xs = center_xs[::-1]

        images = np.stack([center_crop(img, center_x, center_y, crop_H, crop_W) for img, center_x, center_y in zip(images, center_xs, center_ys)], axis=0)
        images = np.stack([cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC) for img in images], axis=0)
        return images

    def zoom_aug(self, images, final_scale=1., zoom_in_or_out='in'):
        def zoom_in_with_scale(img, scale):
            H, W = img.shape[:2]
            img = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_CUBIC)
            # center crop to H, W
            img = img[(img.shape[0] - H) // 2: (img.shape[0] - H) // 2 + H, (img.shape[1] - W) // 2: (img.shape[1] - W) // 2 + W]
            return img
        if final_scale <= 1.02 :
            return images
        if zoom_in_or_out == 'in':
            scales = np.linspace(1., final_scale, self.num_frames)
            images = np.array([zoom_in_with_scale(img, scale) for img, scale in zip(images, scales)])
        elif zoom_in_or_out == 'out':
            scales = np.linspace(final_scale, 1., self.num_frames)
            images = np.array([zoom_in_with_scale(img, scale) for img, scale in zip(images, scales)])
        return images

    def motion_augmentation(self, input_images, edited_images):
        H, W = input_images.shape[1:3]

        # translation augmentation
        if np.random.random() < self.translation_ratio:
            delta_h = np.random.uniform(self.translation_range[0], self.translation_range[1]) * H * np.random.choice([-1, 1])
            delta_w = np.random.uniform(self.translation_range[0], self.translation_range[1]) * W * np.random.choice([-1, 1])
            # print(delta_h, delta_w)
            input_images = self.translation_crop(delta_h, delta_w, input_images)
            edited_images = self.translation_crop(delta_h, delta_w, edited_images)

        # zoom augmentation
        if np.random.random() < self.zoom_ratio:
            final_scale = np.random.uniform(1., self.max_zoom)
            zoom_in_or_out = np.random.choice(['in', 'out'])
            # print(final_scale, zoom_in_or_out)
            input_images = self.zoom_aug(input_images, final_scale, zoom_in_or_out)
            edited_images = self.zoom_aug(edited_images, final_scale, zoom_in_or_out)
        
        return input_images, edited_images

    def __getitem__(self, idx):
        folder = self.image_folders[idx]
        with jsonlines.open(os.path.join(folder, 'metadata.jsonl')) as reader:
            seeds = [obj['seed'] for obj in reader if (obj['sim_dir'] > 0.2 and obj['sim_0'] > 0.2 and obj['sim_1'] > 0.2 and obj['sim_image'] > 0.5)]
        seed = np.random.choice(seeds)

        # Load prompt.json
        with open(os.path.join(folder, 'prompt.json'), 'r') as f:
            prompt = json.load(f)

        start_idx = np.random.randint(0, 16 - self.num_frames + 1)
        end_idx = start_idx + self.num_frames
        
        input_images = np.array([self.load_image(os.path.join(folder, 'image', f'{seed}_0_{img_idx:04d}.jpg')) for img_idx in range(start_idx, end_idx)])
        edited_images = np.array([self.load_image(os.path.join(folder, 'image', f'{seed}_1_{img_idx:04d}.jpg')) for img_idx in range(start_idx, end_idx)])

        input_images, edited_images = self.motion_augmentation(input_images, edited_images)

        return {
            'input_video': self.numpy_to_tensor(input_images), 
            'edited_video': self.numpy_to_tensor(edited_images),
            'input_prompt': prompt['input'], 
            'output_prompt': prompt['output'], 
            'edit_prompt': prompt['edit']
        }