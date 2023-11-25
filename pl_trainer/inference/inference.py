import torch
import numpy as np
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import DDPMScheduler, DDIMScheduler, PNDMScheduler
import torch.nn.functional as nnf
import numpy as np
from einops import rearrange
from misc_utils.flow_utils import warp_image, RAFTFlow, resize_flow
from functools import partial

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg
        
class Inference():
    def __init__(
        self, 
        unet,
        scheduler='ddim',
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        num_ddim_steps=20, guidance_scale=5,
    ):
        self.unet = unet
        if scheduler == 'ddim':
            scheduler_cls = DDIMScheduler
            scheduler_kwargs = {'set_alpha_to_one': False, 'steps_offset': 1, 'clip_sample': False}
        elif scheduler == 'ddpm':
            scheduler_cls = DDPMScheduler
            scheduler_kwargs = {'clip_sample': False}
        else:
            raise NotImplementedError()
        self.scheduler = scheduler_cls(
            beta_start = beta_start,
            beta_end = beta_end,
            beta_schedule = beta_schedule,
            **scheduler_kwargs
        )
        self.scheduler.set_timesteps(num_ddim_steps)
        self.num_ddim_steps = num_ddim_steps
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def __call__(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        uncond_context: torch.Tensor=None,
        start_time: int = 0,
        null_embedding: List[torch.Tensor]=None,
        context_kwargs={},
        model_kwargs={},
    ):
        all_latent = []
        all_pred = [] # x0_hat
        do_classifier_free_guidance = self.guidance_scale > 1 and ((uncond_context is not None) or (null_embedding is not None))
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)
            if do_classifier_free_guidance:
                latent_input = torch.cat([latent, latent], dim=0)
                if null_embedding is not None:
                    context_input = torch.cat([null_embedding[i], context], dim=0)
                else:
                    context_input = torch.cat([uncond_context, context], dim=0)
            else:
                latent_input = latent
                context_input = context
            noise_pred = self.unet(
                latent_input,
                torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
                context={ 'text': context_input, **context_kwargs},
                **model_kwargs
            )

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            pred_samples = self.scheduler.step(noise_pred, t, latent)
            latent = pred_samples.prev_sample
            pred = pred_samples.pred_original_sample
            all_latent.append(latent.detach())
            all_pred.append(pred.detach())

        return {
            'latent': latent,
            'all_latent': all_latent,
            'all_pred': all_pred
        }

class InferenceIP2PEditRef(Inference):
    def zeros(self, x):
        return torch.zeros_like(x)
    @torch.no_grad()
    def __call__(
        self,
        latent: torch.Tensor,
        text_cond: torch.Tensor,
        text_uncond: torch.Tensor,
        img_cond: torch.Tensor,
        edit_cond: torch.Tensor,
        text_cfg = 7.5,
        img_cfg = 1.2,
        edit_cfg = 1.2,
        start_time: int = 0,
    ):
        '''
                latent1 | latent2 | latent3 | latent4
        text       x         x         x         v
        edit       x         x         v         v
        img        x         v         v         v
        '''
        all_latent = []
        all_pred = [] # x0_hat
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)
            latent1 = torch.cat([latent, self.zeros(img_cond), self.zeros(edit_cond)], dim=1)
            latent2 = torch.cat([latent, img_cond, self.zeros(edit_cond)], dim=1)
            latent3 = torch.cat([latent, img_cond, edit_cond], dim=1)
            latent4 = latent3.clone()
            latent_input = torch.cat([latent1, latent2, latent3, latent4], dim=0)
            context_input = torch.cat([text_uncond, text_uncond, text_uncond, text_cond], dim=0)
            noise_pred = self.unet(
                latent_input,
                torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
                context={ 'text': context_input},
            )

            noise_pred1, noise_pred2, noise_pred3, noise_pred4 = noise_pred.chunk(4, dim=0)
            noise_pred = (
                noise_pred1 + 
                img_cfg * (noise_pred2 - noise_pred1) +
                edit_cfg * (noise_pred3 - noise_pred2) +
                text_cfg * (noise_pred4 - noise_pred3)
            ) # when edit_cfg == img_cfg, noise_pred2 is not used

            pred_samples = self.scheduler.step(noise_pred, t, latent)
            latent = pred_samples.prev_sample
            pred = pred_samples.pred_original_sample
            all_latent.append(latent.detach())
            all_pred.append(pred.detach())

        return {
            'latent': latent,
            'all_latent': all_latent,
            'all_pred': all_pred
        }

class InferenceIP2PVideo(Inference):
    def zeros(self, x):
        return torch.zeros_like(x)
    @torch.no_grad()
    def __call__(
        self,
        latent: torch.Tensor,
        text_cond: torch.Tensor,
        text_uncond: torch.Tensor,
        img_cond: torch.Tensor,
        text_cfg = 7.5,
        img_cfg = 1.2,
        start_time: int = 0,
        guidance_rescale: float = 0.0,
    ):
        '''
                latent1 | latent2 | latent3
        text       x         x         v
        img        x         v         v
        '''
        all_latent = []
        all_pred = [] # x0_hat
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)
            latent1 = torch.cat([latent, self.zeros(img_cond)], dim=2)
            latent2 = torch.cat([latent, img_cond], dim=2)
            latent3 = latent2.clone()
            latent_input = torch.cat([latent1, latent2, latent3], dim=0)
            context_input = torch.cat([text_uncond, text_uncond, text_cond], dim=0)

            latent_input = rearrange(latent_input, 'b f c h w -> b c f h w')
            noise_pred = self.unet(
                latent_input,
                torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
                encoder_hidden_states=context_input,
            ).sample
            noise_pred = rearrange(noise_pred, 'b c f h w -> b f c h w')


            noise_pred1, noise_pred2, noise_pred3 = noise_pred.chunk(3, dim=0)
            noise_pred = (
                noise_pred1 + 
                img_cfg * (noise_pred2 - noise_pred1) +
                text_cfg * (noise_pred3 - noise_pred2)
            )

            if guidance_rescale > 0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred1, guidance_rescale=guidance_rescale)

            pred_samples = self.scheduler.step(noise_pred, t, latent)
            latent = pred_samples.prev_sample
            pred = pred_samples.pred_original_sample
            all_latent.append(latent.detach())
            all_pred.append(pred.detach())

        return {
            'latent': latent,
            'all_latent': all_latent,
            'all_pred': all_pred
        }

    @torch.no_grad()
    def second_clip_forward(
        self,
        latent: torch.Tensor,
        text_cond: torch.Tensor,
        text_uncond: torch.Tensor,
        img_cond: torch.Tensor,
        latent_ref: torch.Tensor,
        noise_correct_step: float = 1.,
        text_cfg = 7.5,
        img_cfg = 1.2,
        start_time: int = 0,
        guidance_rescale: float = 0.0,
    ):
        '''
                latent1 | latent2 | latent3
        text       x         x         v
        img        x         v         v
        '''
        num_ref_frames = latent_ref.shape[1]
        all_latent = []
        all_pred = [] # x0_hat
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)

            latent1 = torch.cat([latent, self.zeros(img_cond)], dim=2)
            latent2 = torch.cat([latent, img_cond], dim=2)
            latent3 = latent2.clone()
            latent_input = torch.cat([latent1, latent2, latent3], dim=0)
            context_input = torch.cat([text_uncond, text_uncond, text_cond], dim=0)

            latent_input = rearrange(latent_input, 'b f c h w -> b c f h w')
            noise_pred = self.unet(
                latent_input,
                torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
                encoder_hidden_states=context_input,
            ).sample
            noise_pred = rearrange(noise_pred, 'b c f h w -> b f c h w')

            noise_pred1, noise_pred2, noise_pred3 = noise_pred.chunk(3, dim=0)
            noise_pred = (
                noise_pred1 + 
                img_cfg * (noise_pred2 - noise_pred1) +
                text_cfg * (noise_pred3 - noise_pred2)
            )

            if guidance_rescale > 0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred1, guidance_rescale=guidance_rescale)


            if noise_correct_step * self.num_ddim_steps > i:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                noise_ref = (latent[:, 0:num_ref_frames] - (alpha_prod_t ** 0.5) * latent_ref) / (beta_prod_t ** 0.5) # b 1 c h w
                delta_noise_ref = noise_ref - noise_pred[:, 0:num_ref_frames]
                delta_noise_remaining = delta_noise_ref.mean(dim=1, keepdim=True)
                noise_pred[:, :num_ref_frames] = noise_pred[:, :num_ref_frames] + delta_noise_ref
                noise_pred[:, num_ref_frames:] = noise_pred[:, num_ref_frames:] + delta_noise_remaining

            pred_samples = self.scheduler.step(noise_pred, t, latent)
            latent = pred_samples.prev_sample
            pred = pred_samples.pred_original_sample
            all_latent.append(latent.detach())
            all_pred.append(pred.detach())

        return {
            'latent': latent,
            'all_latent': all_latent,
            'all_pred': all_pred
        }

class InferenceIP2PVideoOpticalFlow(InferenceIP2PVideo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_estimator = RAFTFlow().cuda()

    def obtain_delta_noise(self, delta_noise_ref, flow):
        flow = resize_flow(flow, delta_noise_ref.shape[2:])
        warped_delta_noise_ref = warp_image(delta_noise_ref, flow)
        valid_mask = torch.ones_like(delta_noise_ref)[:, :1]
        valid_mask = warp_image(valid_mask, flow)
        return warped_delta_noise_ref, valid_mask

    def obtain_flow_batched(self, ref_images, query_images):
        ref_images = ref_images.to()
        warp_funcs = []
        for query_image in query_images:
            query_image = query_image.unsqueeze(0).repeat(len(ref_images), 1, 1, 1)
            flow = self.flow_estimator(query_image, ref_images)
            warp_func = partial(self.obtain_delta_noise, flow=flow)
            warp_funcs.append(warp_func)
        return warp_funcs

    @torch.no_grad()
    def second_clip_forward(
        self,
        latent: torch.Tensor,
        text_cond: torch.Tensor,
        text_uncond: torch.Tensor,
        img_cond: torch.Tensor,
        latent_ref: torch.Tensor,
        ref_images: torch.Tensor,
        query_images: torch.Tensor,
        noise_correct_step: float = 1.,
        text_cfg = 7.5,
        img_cfg = 1.2,
        start_time: int = 0,
        guidance_rescale: float = 0.0,
    ):
        '''
                latent1 | latent2 | latent3
        text       x         x         v
        img        x         v         v
        '''
        assert ref_images.shape[0] == 1, 'only support batch size 1'
        warp_funcs = self.obtain_flow_batched(ref_images[0], query_images[0])
        num_ref_frames = latent_ref.shape[1]
        all_latent = []
        all_pred = [] # x0_hat
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)

            latent1 = torch.cat([latent, self.zeros(img_cond)], dim=2)
            latent2 = torch.cat([latent, img_cond], dim=2)
            latent3 = latent2.clone()
            latent_input = torch.cat([latent1, latent2, latent3], dim=0)
            context_input = torch.cat([text_uncond, text_uncond, text_cond], dim=0)

            latent_input = rearrange(latent_input, 'b f c h w -> b c f h w')
            noise_pred = self.unet(
                latent_input,
                torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
                encoder_hidden_states=context_input,
            ).sample
            noise_pred = rearrange(noise_pred, 'b c f h w -> b f c h w')

            noise_pred1, noise_pred2, noise_pred3 = noise_pred.chunk(3, dim=0)
            noise_pred = (
                noise_pred1 + 
                img_cfg * (noise_pred2 - noise_pred1) +
                text_cfg * (noise_pred3 - noise_pred2)
            )

            if guidance_rescale > 0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred1, guidance_rescale=guidance_rescale)


            if noise_correct_step * self.num_ddim_steps > i:
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                noise_ref = (latent[:, 0:num_ref_frames] - (alpha_prod_t ** 0.5) * latent_ref) / (beta_prod_t ** 0.5) # b 1 c h w
                delta_noise_ref = noise_ref - noise_pred[:, 0:num_ref_frames]
                noise_pred[:, :num_ref_frames] = noise_pred[:, :num_ref_frames] + delta_noise_ref

                for refed_index, warp_func in zip(range(num_ref_frames, noise_pred.shape[1]), warp_funcs):
                    delta_noise_remaining, delta_noise_mask = warp_func(delta_noise_ref[0])
                    mask_sum = delta_noise_mask[None].sum(dim=1, keepdim=True)
                    delta_noise_remaining = torch.where(
                        mask_sum > 0.5,
                        delta_noise_remaining[None].sum(dim=1, keepdim=True) / mask_sum,
                        0.
                    )
                    noise_pred[:, refed_index: refed_index+1] += torch.where(
                        mask_sum > 0.5,
                        delta_noise_remaining,
                        0
                    )

            pred_samples = self.scheduler.step(noise_pred, t, latent)
            latent = pred_samples.prev_sample
            pred = pred_samples.pred_original_sample
            all_latent.append(latent.detach())
            all_pred.append(pred.detach())

        return {
            'latent': latent,
            'all_latent': all_latent,
            'all_pred': all_pred
        }