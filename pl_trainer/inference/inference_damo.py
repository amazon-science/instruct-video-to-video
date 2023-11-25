import torch
from typing import List, Union, Tuple
from tqdm import tqdm
from .inference import Inference

class InferenceDAMO(Inference):
    @torch.no_grad()
    def __call__(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        uncond_context: torch.Tensor=None,
        start_time: int = 0,
        null_embedding: List[torch.Tensor]=None,
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
                context_input,
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

class InferenceDAMO_PTP(Inference):
    def infer_old_context(self, latent, context, t, uncond_context=None):
        do_classifier_free_guidance = self.guidance_scale > 1 and (uncond_context is not None)

        if do_classifier_free_guidance:
            latent_input = torch.cat([latent, latent], dim=0)
            context_input = torch.cat([uncond_context, context], dim=0)
        else:
            latent_input = latent
            context_input = context

        noise_pred = self.unet(
            latent_input,
            torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
            context_input,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        pred_samples = self.scheduler.step(noise_pred, t, latent)
        latent = pred_samples.prev_sample
        pred = pred_samples.pred_original_sample
        return latent, pred

    def infer_new_context(self, latent, context, t, uncond_context=None):
        do_classifier_free_guidance = self.guidance_scale > 1 and (uncond_context is not None)

        if do_classifier_free_guidance:
            latent_input = torch.cat([latent, latent], dim=0)
            if isinstance(context, (list, tuple)):
                context_input = (
                    torch.cat([uncond_context, context[0]], dim=0),
                    torch.cat([uncond_context, context[1]], dim=0),
                )
            else:
                context_input = torch.cat([uncond_context, context], dim=0)
        else:
            latent_input = latent
            context_input = context

        noise_pred = self.unet(
            latent_input,
            torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
            context_input,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        pred_samples = self.scheduler.step(noise_pred, t, latent)
        latent = pred_samples.prev_sample
        pred = pred_samples.pred_original_sample
        return latent, pred

    @torch.no_grad()
    def __call__(
        self,
        latent: torch.Tensor,
        context: torch.Tensor, # used when > ca_end_time
        old_context: torch.Tensor=None, # used when < sa_end_time
        old_to_new_context: Union[Tuple, List]=None, # used when sa_end_time < t < ca_end_time
        uncond_context: torch.Tensor=None,
        sa_end_time: float=0.3,
        ca_end_time: float=0.8,
        start_time: int = 0,
    ):
        assert sa_end_time < ca_end_time, f"sa_end_time must be less than ca_end_time, got {sa_end_time} and {ca_end_time} respectively"
        all_latent = []
        all_pred = []
        all_latent_old = []
        all_pred_old = []
        old_latent = latent.clone()
        new_latent = latent.clone()
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)
            old_latent_next_t, pred_old = self.infer_old_context(old_latent, old_context, t, uncond_context)
            if i < sa_end_time * self.num_ddim_steps:
                new_latent_next_t, pred_new = old_latent_next_t, pred_old
            elif sa_end_time * self.num_ddim_steps <= i < ca_end_time * self.num_ddim_steps:
                new_latent_next_t, pred_new = self.infer_new_context(
                    new_latent, old_to_new_context, t, uncond_context
                )
            else:
                new_latent_next_t, pred_new = self.infer_new_context(
                    new_latent, context, t, uncond_context
                )

            old_latent = old_latent_next_t
            new_latent = new_latent_next_t

            all_latent.append(new_latent_next_t.detach())
            all_pred.append(pred_new.detach())
            all_latent_old.append(old_latent_next_t.detach())
            all_pred_old.append(pred_old.detach())

        return {
            'latent': new_latent,
            'latent_old': old_latent,
            'all_latent': all_latent,
            'all_pred': all_pred,
            'all_latent_old': all_latent_old,
            'all_pred_old': all_pred_old,
        }

class InferenceDAMO_PTP_v2(Inference):
    def set_ptp_in_xattn_layers(self, prompt_to_prompt: bool, num_frames=1):
        for m in self.unet.modules():
            if m.__class__.__name__ == 'CrossAttention':
                m.ptp_sa_replace = prompt_to_prompt
                m.num_frames = num_frames

    def infer_both_with_sa_replace(self, old_latent, new_latent, old_context, new_context, t, uncond_context=None):
        do_classifier_free_guidance = self.guidance_scale > 1 and (uncond_context is not None)

        if do_classifier_free_guidance:
            latent_input = torch.cat([old_latent, new_latent, old_latent, new_latent], dim=0)
            context_input = torch.cat([uncond_context, uncond_context, old_context, new_context], dim=0)
        else:
            latent_input = torch.cat([old_latent, new_latent], dim=0)
            context_input = torch.cat([old_context, new_context], dim=0)

        noise_pred = self.unet(
            latent_input,
            torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
            context_input,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        noise_pred_old, noise_pred_new = noise_pred.chunk(2, dim=0)
        pred_samples_old = self.scheduler.step(noise_pred_old, t, old_latent)
        pred_samples_new = self.scheduler.step(noise_pred_new, t, new_latent)

        old_latent = pred_samples_old.prev_sample
        new_latent = pred_samples_new.prev_sample
        old_pred = pred_samples_old.pred_original_sample
        new_pred = pred_samples_new.pred_original_sample

        return old_latent, new_latent, old_pred, new_pred

    def infer_old_context(self, latent, context, t, uncond_context=None):
        do_classifier_free_guidance = self.guidance_scale > 1 and (uncond_context is not None)

        if do_classifier_free_guidance:
            latent_input = torch.cat([latent, latent], dim=0)
            context_input = torch.cat([uncond_context, context], dim=0)
        else:
            latent_input = latent
            context_input = context

        noise_pred = self.unet(
            latent_input,
            torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
            context_input,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        pred_samples = self.scheduler.step(noise_pred, t, latent)
        latent = pred_samples.prev_sample
        pred = pred_samples.pred_original_sample
        return latent, pred

    def infer_new_context(self, latent, context, t, uncond_context=None):
        do_classifier_free_guidance = self.guidance_scale > 1 and (uncond_context is not None)

        if do_classifier_free_guidance:
            latent_input = torch.cat([latent, latent], dim=0)
            if isinstance(context, (list, tuple)):
                context_input = (
                    torch.cat([uncond_context, context[0]], dim=0),
                    torch.cat([uncond_context, context[1]], dim=0),
                )
            else:
                context_input = torch.cat([uncond_context, context], dim=0)
        else:
            latent_input = latent
            context_input = context

        noise_pred = self.unet(
            latent_input,
            torch.full((len(latent_input),), t, device=latent_input.device, dtype=torch.long), 
            context_input,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        pred_samples = self.scheduler.step(noise_pred, t, latent)
        latent = pred_samples.prev_sample
        pred = pred_samples.pred_original_sample
        return latent, pred

    @torch.no_grad()
    def __call__(
        self,
        latent: torch.Tensor,
        context: torch.Tensor, # used when > ca_end_time
        old_context: torch.Tensor=None, # used when < sa_end_time
        old_to_new_context: Union[Tuple, List]=None, # used when sa_end_time < t < ca_end_time
        uncond_context: torch.Tensor=None,
        sa_end_time: float=0.3,
        ca_end_time: float=0.8,
        start_time: int = 0,
    ):
        assert sa_end_time < ca_end_time, f"sa_end_time must be less than ca_end_time, got {sa_end_time} and {ca_end_time} respectively"
        all_latent = []
        all_pred = []
        all_latent_old = []
        all_pred_old = []
        old_latent = latent.clone()
        new_latent = latent.clone()
        for i, t in enumerate(tqdm(self.scheduler.timesteps[start_time:])):
            t = int(t)
            if i < sa_end_time * self.num_ddim_steps:
                self.set_ptp_in_xattn_layers(True, num_frames=latent.shape[2])
                old_latent_next_t, new_latent_next_t, pred_old, pred_new = self.infer_both_with_sa_replace(
                    old_latent, new_latent, old_context, context, t, uncond_context
                )
            elif sa_end_time * self.num_ddim_steps <= i < ca_end_time * self.num_ddim_steps:
                self.set_ptp_in_xattn_layers(False)
                old_latent_next_t, pred_old = self.infer_old_context(old_latent, old_context, t, uncond_context)
                new_latent_next_t, pred_new = self.infer_new_context(
                    new_latent, old_to_new_context, t, uncond_context
                )
            else:
                self.set_ptp_in_xattn_layers(False)
                old_latent_next_t, pred_old = self.infer_old_context(old_latent, old_context, t, uncond_context)
                new_latent_next_t, pred_new = self.infer_new_context(
                    new_latent, context, t, uncond_context
                )

            old_latent = old_latent_next_t
            new_latent = new_latent_next_t

            all_latent.append(new_latent_next_t.detach())
            all_pred.append(pred_new.detach())
            all_latent_old.append(old_latent_next_t.detach())
            all_pred_old.append(pred_old.detach())

        return {
            'latent': new_latent,
            'latent_old': old_latent,
            'all_latent': all_latent,
            'all_pred': all_pred,
            'all_latent_old': all_latent_old,
            'all_pred_old': all_pred_old,
        }