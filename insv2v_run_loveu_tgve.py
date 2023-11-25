import os
import torch
from dataset.loveu_tgve_dataset import LoveuTgveVideoDataset
from matplotlib import pyplot as plt
from misc_utils.train_utils import unit_test_create_model
from itertools import product
from pl_trainer.inference.inference import InferenceIP2PVideo, InferenceIP2PVideoOpticalFlow
import json
import argparse
from misc_utils.image_utils import save_tensor_to_gif, save_tensor_to_images

def split_batch(cond, frames_in_batch=16, num_ref_frames=4):
    frames_in_following_batch = frames_in_batch - num_ref_frames
    conds = [cond[:, :frames_in_batch]]
    frame_ptr = frames_in_batch
    num_ref_frames_each_batch = []

    while frame_ptr < cond.shape[1]:
        remaining_frames = cond.shape[1] - frame_ptr
        if remaining_frames < frames_in_batch:
            frames_in_following_batch = remaining_frames
        else:
            frames_in_following_batch = frames_in_batch - num_ref_frames
        this_ref_frames = frames_in_batch - frames_in_following_batch
        conds.append(cond[:, frame_ptr:frame_ptr+frames_in_following_batch])
        frame_ptr += frames_in_following_batch
        num_ref_frames_each_batch.append(this_ref_frames)

    return conds, num_ref_frames_each_batch

parser = argparse.ArgumentParser(description='Your program description')

# Add arguments
parser.add_argument('--text-cfg', nargs='+', type=float, default=[7.5], help='Text configuration parameter')
parser.add_argument('--video-cfg', nargs='+', type=float, default=[1.8], help='Image configuration parameter')
parser.add_argument('--num-frames', nargs='+', type=int, default=[32], help='Number of frames')
parser.add_argument('--image-size', nargs='+', type=int, default=[384], help='Image size')
parser.add_argument('--prompt-source', type=str, default='edit', help='Prompt source')
parser.add_argument('--ckpt-path', type=str, help='Path to checkpoint')
parser.add_argument('--config-path', type=str, default='configs/instruct_v2v.yaml', help='Path to config file')
parser.add_argument('--data-dir', type=str, default='loveu-tgve-2023', help='Path to LOVEU dataset')
parser.add_argument('--with_optical_flow', action='store_true', help='Use motion compensation')

# Parse arguments
args = parser.parse_args()

TEXT_CFGS = args.text_cfg
VIDEO_CFGS = args.video_cfg
NUM_FRAMES = args.num_frames
IMAGE_SIZE = args.image_size

PROMPT_SOURCE = args.prompt_source
DATA_ROOT = args.data_dir

config_path = args.config_path
ckpt_path = args.ckpt_path

diffusion_model = unit_test_create_model(config_path)

ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt = {k.replace('_forward_module.', ''): v for k, v in ckpt.items()}
diffusion_model.load_state_dict(ckpt, strict=False)

if args.with_optical_flow:
    inf_pipe = InferenceIP2PVideoOpticalFlow(
        unet = diffusion_model.unet,
        num_ddim_steps=20,
        scheduler='ddpm'
    )
else:
    inf_pipe = InferenceIP2PVideo(
        unet = diffusion_model.unet,
        num_ddim_steps=20,
        scheduler='ddpm'
    )

frames_in_batch = 16
num_ref_frames = 4

edit_prompt_file = 'dataset/loveu_tgve_edit_prompt_dict.json'
edit_prompt_dict = json.load(open(edit_prompt_file, 'r'))

for VIDEO_ID, text_cfg, video_cfg, num_frames, image_size in product(range(len(edit_prompt_dict)), TEXT_CFGS, VIDEO_CFGS, NUM_FRAMES, IMAGE_SIZE):
    dataset = LoveuTgveVideoDataset(
        root_dir=DATA_ROOT,
        image_size=(image_size, image_size),
    )

    batch = dataset[VIDEO_ID]
    video_name = batch['video_name']
    num_video_frames = len(batch['frames'])
    if num_video_frames > num_frames:
        frame_skip = num_video_frames // num_frames
    else:
        frame_skip = 1
    batch = {k: v[::frame_skip].cuda()[None] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    cond = diffusion_model.encode_image_to_latent(batch['frames']) / 0.18215
    text_uncond = diffusion_model.encode_text([''])

    for prompt_key in ['style', 'object', 'background', 'multiple']:
        final_prompt = batch[prompt_key]
        if PROMPT_SOURCE == 'edit':
            prompt = edit_prompt_dict[video_name]['edit_' + prompt_key]
            out_folder = f'v2v_results/edit_prompt/loveu_tgve_{image_size}/gif/VID_{VIDEO_ID}/VIDEO_CFG_{video_cfg}_TEXT_CFG_{text_cfg}'
            image_output_dir = f'v2v_results/edit_prompt/loveu_tgve_{image_size}/images_{num_frames}/VIDEO_CFG_{video_cfg}_TEXT_CFG_{text_cfg}/{video_name}/{prompt_key}'
        elif PROMPT_SOURCE == 'original':
            prompt = batch[prompt_key]
            out_folder = f'v2v_results/original_prompt/loveu_tgve_{image_size}/gif/VID_{VIDEO_ID}/VIDEO_CFG_{video_cfg}_TEXT_CFG_{text_cfg}'
            image_output_dir = f'v2v_results/original_prompt/loveu_tgve_{image_size}/images_{num_frames}/VIDEO_CFG_{video_cfg}_TEXT_CFG_{text_cfg}/{video_name}/{prompt_key}'

        text = '_'.join(final_prompt.split(' '))
        output_path = f'{out_folder}/{prompt_key}_{num_frames}_{text}.gif'
        if os.path.exists(output_path):
            print(f'File {output_path} exists, skip')
            continue

        text_cond = diffusion_model.encode_text(prompt)
        conds, num_ref_frames_each_batch = split_batch(cond, frames_in_batch=frames_in_batch, num_ref_frames=num_ref_frames)
        splitted_frames, _ = split_batch(batch['frames'], frames_in_batch=frames_in_batch, num_ref_frames=num_ref_frames)

        # First video clip
        cond1 = conds[0]
        latent_pred_list = []
        init_latent = torch.randn_like(cond1)
        latent_pred = inf_pipe(
            latent = init_latent,
            text_cond = text_cond,
            text_uncond = text_uncond,
            img_cond = cond1,
            text_cfg = text_cfg,
            img_cfg = video_cfg,
        )['latent']
        latent_pred_list.append(latent_pred)


        # Subsequent video clips
        for prev_cond, cond_, prev_frame, curr_frame, num_ref_frames_ in zip(conds[:-1], conds[1:], splitted_frames[:-1], splitted_frames[1:], num_ref_frames_each_batch):
            init_latent = torch.cat([init_latent[:, -num_ref_frames_:], torch.randn_like(cond_)], dim=1)
            cond_ = torch.cat([prev_cond[:, -num_ref_frames_:], cond_], dim=1)
            if args.with_optical_flow:
                ref_images = prev_frame[:, -num_ref_frames_:]
                query_images = curr_frame
                additional_kwargs = {
                    'ref_images': ref_images,
                    'query_images': query_images,
                }
            else:
                additional_kwargs = {}
            latent_pred = inf_pipe.second_clip_forward(
                latent = init_latent, 
                text_cond = text_cond,
                text_uncond = text_uncond,
                img_cond = cond_,
                latent_ref = latent_pred[:, -num_ref_frames_:],
                noise_correct_step = 0.5,
                text_cfg = text_cfg,
                img_cfg = video_cfg,
                **additional_kwargs,
            )['latent']
            latent_pred_list.append(latent_pred[:, num_ref_frames_:])

        # Save GIF
        latent_pred = torch.cat(latent_pred_list, dim=1)
        image_pred = diffusion_model.decode_latent_to_image(latent_pred).clip(-1, 1)

        original_images = batch['frames'].cpu()
        transferred_images = image_pred.float().cpu()
        concat_images = torch.cat([original_images, transferred_images], dim=4)

        save_tensor_to_gif(concat_images, output_path, fps=5)
        save_tensor_to_images(transferred_images, image_output_dir)