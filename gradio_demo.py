import os
import gradio as gr
import numpy as np
from misc_utils.image_utils import save_tensor_to_gif
from misc_utils.train_utils import unit_test_create_model
from pl_trainer.inference.inference import InferenceIP2PVideoOpticalFlow
from dataset.single_video_dataset import SingleVideoDataset
import torch

NEGATIVE_PROMPT = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting'
# The order is: [video_path, edit_prompt, negative_prompt, text_cfg, video_cfg, resolution, sample_rate, num_frames, start_frame]
CARTURN = [ 'data/car-turn.mp4', 'Change the car to a red Porsche and make the background beach.', NEGATIVE_PROMPT, 10, 1.1, 512, 10, 28, 20, ]
AIRPLANE = ['data/airplane-and-contrail.mp4', "add Taj Mahal in the image", NEGATIVE_PROMPT, 10, 1.2, 512, 30, 28, 0]
AUDI = ['data/audi-snow-trail.mp4', "make the car drive in desert trail.", NEGATIVE_PROMPT, 10, 1.5, 512, 3, 28, 0]
CATINSUN_BKG = ['data/cat-in-the-sun.mp4', "change the background to a beach.", NEGATIVE_PROMPT, 7.5, 1.3, 512, 6, 28, 0]
DIRTROAD = ['data/dirt-road-driving.mp4', 'add dust cloud effect.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
EARTH = ['data/earth-full-view.mp4', 'add a fireworks display in the background..', NEGATIVE_PROMPT, 7.5, 1.2, 512, 30, 28, 0]
EIFFELTOWER = ['data/eiffel-flyover.mp4', 'add a large fireworks display.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
FERRIS = ['data/ferris-wheel-timelapse.mp4', 'Add a sunset in the background.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
GOLDFISH = ['data/gold-fish.mp4', 'make the style impressionist', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
ICEHOCKEY = ['data/ice-hockey.mp4', 'make the players to cartoon characters.', NEGATIVE_PROMPT, 10, 1.5, 512, 6, 28, 0]
MIAMISURF = ['data/miami-surf.mp4', 'change the background to wave pool.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
RAINDROP = ['data/raindrops.mp4', 'Make the style expressionism.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
REDROSE_BKG = ['data/red-roses-sunny-day.mp4', 'make background to moonlight.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
REDROSE_STY = ['data/red-roses-sunny-day.mp4', 'Make the style origami.', NEGATIVE_PROMPT, 10, 1.2, 512, 6, 28, 0]
SWAN_OBJ = ['data/swans.mp4', 'change swans to pink flamingos.', NEGATIVE_PROMPT, 7.5, 1.2, 512, 6, 28, 0]

class VideoTransfer:
    def __init__(self, config_path, model_ckpt, device='cuda'):
        self.config_path = config_path
        self.model_ckpt = model_ckpt
        self.device = device
        self.diffusion_model = None
        self.pipe = None

    def _init_pipe(self):
        diffusion_model = unit_test_create_model(self.config_path, device=self.device)
        ckpt = torch.load(self.model_ckpt, map_location='cpu')
        diffusion_model.load_state_dict(ckpt, strict=False)
        self.diffusion_model = diffusion_model
        self.pipe = InferenceIP2PVideoOpticalFlow(
            unet = diffusion_model.unet,
            num_ddim_steps=20,
            scheduler='ddpm'
        )

    def get_batch(self, video_path, video_sample_rate, num_frames, image_size, start_frame=0):
        dataset = SingleVideoDataset(
            video_file=video_path,
            video_description='',
            sampling_fps=video_sample_rate,
            num_frames=num_frames,
            output_size=(image_size, image_size)
        )
        batch = dataset[start_frame]
        batch = {k: v.to(self.device)[None] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    @staticmethod
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

    def get_splitted_batch_and_conds(self, batch, edit_promt, negative_prompt):
        diffusion_model = self.diffusion_model
        cond = [diffusion_model.encode_image_to_latent(frames) / 0.18215 for frames in batch['frames'].chunk(16, dim=1)] # when encoding, chunk the frames to avoid oom in vae, you can reduce the 16 if you have a smaller gpu
        cond = torch.cat(cond, dim=1)
        text_cond = diffusion_model.encode_text([edit_promt])
        text_uncond = diffusion_model.encode_text([negative_prompt])
        conds, num_ref_frames_each_batch = self.split_batch(cond, frames_in_batch=16, num_ref_frames=4)
        splitted_frames, _ = self.split_batch(batch['frames'], frames_in_batch=16, num_ref_frames=4)
        return conds, num_ref_frames_each_batch, text_cond, text_uncond, splitted_frames

    def transfer_video(self, video_path, edit_prompt, negative_prompt, text_cfg, video_cfg, resolution, video_sample_rate, num_frames, start_frame):
        # TODO, support seed
        video_name = os.path.basename(video_path).split('.')[0]
        output_file_id = f'{video_name}_{text_cfg}_{video_cfg}_{resolution}_{video_sample_rate}_{num_frames}'
        if self.pipe is None:
            self._init_pipe()

        batch = self.get_batch(video_path, video_sample_rate, num_frames, resolution, start_frame)
        conds, num_ref_frames_each_batch, text_cond, text_uncond, splitted_frames = self.get_splitted_batch_and_conds(batch, edit_prompt, negative_prompt)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            # First video clip
            cond1 = conds[0]
            latent_pred_list = []
            init_latent = torch.randn_like(cond1)
            latent_pred = self.pipe(
                latent = init_latent,
                text_cond = text_cond,
                text_uncond = text_uncond,
                img_cond = cond1,
                text_cfg = text_cfg,
                img_cfg = video_cfg,
            )['latent']
            latent_pred_list.append(latent_pred)


            # Subsequent video clips
            for prev_cond, cond_, prev_frame, curr_frame, num_ref_frames_ in zip(
                conds[:-1], conds[1:], splitted_frames[:-1], splitted_frames[1:], num_ref_frames_each_batch
            ):
                init_latent = torch.cat([init_latent[:, -num_ref_frames_:], torch.randn_like(cond_)], dim=1)
                cond_ = torch.cat([prev_cond[:, -num_ref_frames_:], cond_], dim=1)

                # additional kwargs for using motion compensation
                ref_images = prev_frame[:, -num_ref_frames_:]
                query_images = curr_frame
                additional_kwargs = {
                    'ref_images': ref_images,
                    'query_images': query_images,
                }

                latent_pred = self.pipe.second_clip_forward(
                    latent = init_latent, 
                    text_cond = text_cond,
                    text_uncond = text_uncond,
                    img_cond = cond_,
                    latent_ref = latent_pred[:, -num_ref_frames_:],
                    noise_correct_step = 0.6,
                    text_cfg = text_cfg,
                    img_cfg = video_cfg,
                    **additional_kwargs,
                )['latent']
                latent_pred_list.append(latent_pred[:, num_ref_frames_:])

            # Save GIF
            original_images = batch['frames'].cpu()
            latent_pred = torch.cat(latent_pred_list, dim=1)
            image_pred = self.diffusion_model.decode_latent_to_image(latent_pred).clip(-1, 1)
            transferred_images = image_pred.float().cpu()
            save_tensor_to_gif(original_images, f'gradio_cache/{output_file_id}_original.gif', fps=5)
            save_tensor_to_gif(transferred_images, f'gradio_cache/{output_file_id}.gif', fps=5)
            return f'gradio_cache/{output_file_id}_original.gif', f'gradio_cache/{output_file_id}.gif'

video_transfer = VideoTransfer(
    config_path = 'configs/instruct_v2v_inference.yaml',
    model_ckpt = 'insv2v.pth',
    device = 'cuda',
)

def transfer_video(video_path, edit_prompt, negative_prompt, text_cfg, video_cfg, resolution, video_sample_rate, num_frames, start_frame):
    transferred_video_path = video_transfer.transfer_video(
        video_path = video_path,
        edit_prompt = edit_prompt,
        negative_prompt = negative_prompt,
        text_cfg = float(text_cfg),
        video_cfg = float(video_cfg),
        resolution = int(resolution),
        video_sample_rate = int(video_sample_rate),
        num_frames = int(num_frames),
        start_frame = int(start_frame),
    )
    return transferred_video_path # a gif image

with gr.Blocks() as demo:
    with gr.Row():
        video_source = gr.Video(label="Upload Video", interactive=True, width=384)
        video_input = gr.Image(type='filepath', width=384, label='Original Video')
        video_output = gr.Image(type='filepath', width=384, label='Edited Video')

    with gr.Row():
        with gr.Column(scale=3):
            edit_prompt = gr.Textbox(label="Edit Prompt")
            negative_prompt = gr.Textbox(label="Negative Prompt")
        with gr.Column(scale=1):
            submit_btn = gr.Button(label="Transfer")

    with gr.Row():
        with gr.Row():
            text_cfg = gr.Textbox(label="Text classifier-free guidance", value=7.5)
            video_cfg = gr.Textbox(label="Video classifier-free guidance", value=1.2)
            resolution = gr.Textbox(label="Resolution", value=384)
            sample_rate = gr.Textbox(label="Video Sample Rate", value=5)
            num_frames = gr.Textbox(label="Number of frames", value=28)
            start_frame = gr.Textbox(label="Start frame index", value=0)

    gr.Examples(
        examples=[
            EARTH,
            AUDI,
            DIRTROAD,
            CATINSUN_BKG,
            EIFFELTOWER,
            FERRIS,
            GOLDFISH,
            CARTURN,
            ICEHOCKEY,
            MIAMISURF,
            RAINDROP,
            REDROSE_BKG,
            REDROSE_STY,
            SWAN_OBJ,
            AIRPLANE,
        ],
        inputs=[
            video_source,
            edit_prompt,
            negative_prompt,
            text_cfg,
            video_cfg,
            resolution,
            sample_rate,
            num_frames,
            start_frame,
        ],
    )

    submit_btn.click(
        transfer_video,
        inputs=[
            video_source,
            edit_prompt,
            negative_prompt,
            text_cfg,
            video_cfg,
            resolution,
            sample_rate,
            num_frames,
            start_frame,
        ],
        outputs=[
            video_input,
            video_output
        ],
    )

demo.queue(concurrency_count=1).launch(share=True)
