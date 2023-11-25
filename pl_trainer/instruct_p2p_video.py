'''
Use pretrained instruct pix2pix model but add additional channels for reference modification
'''

import torch
from .diffusion import DDIMLDMTextTraining
from einops import rearrange

class InstructP2PVideoTrainer(DDIMLDMTextTraining):
    def __init__(
        self, *args,
        cond_image_dropout=0.1,
        prompt_type='output_prompt',
        text_cfg=7.5,
        img_cfg=1.2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cond_image_dropout = cond_image_dropout

        assert prompt_type in ['output_prompt', 'edit_prompt', 'mixed_prompt']
        self.prompt_type = prompt_type

        self.text_cfg = text_cfg
        self.img_cfg = img_cfg

        self.unet.enable_xformers_memory_efficient_attention()
        self.unet.enable_gradient_checkpointing()

    def encode_text(self, text):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            encoded_text = super().encode_text(text)
        return encoded_text

    def encode_image_to_latent(self, image):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            latent = super().encode_image_to_latent(image)
        return latent

    @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def get_prompt(self, batch, mode):
        if mode == 'train':
            if self.prompt_type == 'output_prompt':
                prompt = batch['output_prompt']
            elif self.prompt_type == 'edit_prompt':
                prompt = batch['edit_prompt']
            elif self.prompt_type == 'mixed_prompt':
                if int(torch.rand(1)) > 0.5:
                    prompt = batch['output_prompt']
                else:
                    prompt = batch['edit_prompt']
        else:
            prompt = batch['output_prompt']
        return self.encode_text(prompt)

    @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def encode_image_to_latent(self, image):
        b, f, c, h, w = image.shape
        image = rearrange(image, 'b f c h w -> (b f) c h w')
        latent = super().encode_image_to_latent(image)
        latent = rearrange(latent, '(b f) c h w -> b f c h w', b=b)
        return latent

    @torch.cuda.amp.autocast(dtype=torch.float16)
    @torch.no_grad()
    def decode_latent_to_image(self, latent):
        b, f, c, h, w = latent.shape
        latent = rearrange(latent, 'b f c h w -> (b f) c h w')

        image = []
        for latent_ in latent:
            image_ = super().decode_latent_to_image(latent_[None])
            image.append(image_)
        image = torch.cat(image, dim=0)
        # image = super().decode_latent_to_image(latent)
        image = rearrange(image, '(b f) c h w -> b f c h w', b=b)
        return image

    @torch.no_grad()
    def get_cond_image(self, batch, mode):
        cond_image = batch['input_video']

        # ip2p does not scale cond image, so we unscale the cond image
        cond_image = self.encode_image_to_latent(cond_image) / self.scale_factor
        if mode == 'train':
            if int(torch.rand(1)) < self.cond_image_dropout:
                cond_image = torch.zeros_like(cond_image)
        return cond_image

    @torch.no_grad()
    def get_diffused_image(self, batch, mode):
        x = batch['edited_video']
        b, *_ = x.shape
        x = self.encode_image_to_latent(x)
        eps = torch.randn_like(x)

        if mode == 'train':
            t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()
        else:
            t = torch.full((b,), self.num_timesteps-1, device=x.device, dtype=torch.long)
        x_t = self.add_noise(x, t, eps)

        if self.prediction_type == 'epsilon':
            return x_t, eps, t
        else:
            return x_t, x, t

    @torch.no_grad()
    def process_batch(self, batch, mode):
        cond_image = self.get_cond_image(batch, mode)
        diffused_image, target, t = self.get_diffused_image(batch, mode)
        prompt = self.get_prompt(batch, mode)

        model_kwargs = {
            'encoder_hidden_states': prompt
        }

        return {
            'diffused_input': diffused_image,
            'condition': cond_image,
            'target': target,
            't': t,
            'model_kwargs': model_kwargs,
        }

    def training_step(self, batch, batch_idx):
        processed_batch = self.process_batch(batch, mode='train')
        diffused_input = processed_batch['diffused_input']
        condition = processed_batch['condition']
        target = processed_batch['target']
        t = processed_batch['t']
        model_kwargs = processed_batch['model_kwargs']

        model_input = torch.cat([diffused_input, condition], dim=2) # b, f, c, h, w
        model_input = rearrange(model_input, 'b f c h w -> b c f h w')

        pred = self.unet(model_input, t, **model_kwargs).sample
        pred = rearrange(pred, 'b c f h w -> b f c h w')

        loss = self.get_loss(pred, target, t)
        self.log('train_loss', loss, sync_dist=True)

        latent_pred = self.predict_x_0_from_x_t(pred, t, diffused_input)
        image_pred = self.decode_latent_to_image(latent_pred)

        res_dict = {
            'loss': loss,
            'pred': image_pred,
        }
        return res_dict

    @torch.no_grad()
    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def validation_step(self, batch, batch_idx):
        from .inference.inference import InferenceIP2PVideo
        inf_pipe = InferenceIP2PVideo(
            self.unet, 
            beta_start=self.scheduler.config.beta_start,
            beta_end=self.scheduler.config.beta_end,
            beta_schedule=self.scheduler.config.beta_schedule,
            num_ddim_steps=20
        )

        processed_batch = self.process_batch(batch, mode='val')
        diffused_input = torch.randn_like(processed_batch['diffused_input'])

        condition = processed_batch['condition']
        img_cond = condition[:, :, :4]

        res = inf_pipe(
            latent = diffused_input,
            text_cond = processed_batch['model_kwargs']['encoder_hidden_states'],
            text_uncond = self.encode_text(['']),
            img_cond = img_cond,
            text_cfg = self.text_cfg,
            img_cfg = self.img_cfg,
        )

        latent_pred = res['latent']
        image_pred = self.decode_latent_to_image(latent_pred)
        res_dict = {
            'pred': image_pred,
        }
        return res_dict

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.unet.parameters(), lr=self.optim_args['lr'])
        import bitsandbytes as bnb
        params = []
        for name, p in self.unet.named_parameters():
            if ('transformer_in' in name) or ('temp_' in name):
                # p.requires_grad = True
                params.append(p)
            else:
                pass
                # p.requires_grad = False
        optimizer = bnb.optim.Adam8bit(params, lr=self.optim_args['lr'], betas=(0.9, 0.999))
        return optimizer

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd = torch.load(unet_init_weights, map_location='cpu')
            model_sd = self.unet.state_dict()
            # fit input conv size
            for k in model_sd.keys():
                if k in sd.keys():
                    pass
                else:
                # handling temporal layers
                    if (('temp_' in k) or ('transformer_in' in k)) and 'proj_out' in k:
                        # print(f'INFO: initialize {k} from {model_sd[k].shape} to zeros')
                        sd[k] = torch.zeros_like(model_sd[k])
                    else:
                        # print(f'INFO: initialize {k} from {model_sd[k].shape} to random')
                        sd[k] = model_sd[k]
            self.unet.load_state_dict(sd)

class InstructP2PVideoTrainerTemporal(InstructP2PVideoTrainer):
    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd_init_weights, motion_module_init_weights = unet_init_weights
            sd = torch.load(sd_init_weights, map_location='cpu')
            motion_sd = torch.load(motion_module_init_weights, map_location='cpu')
            assert len(sd) + len(motion_sd) == len(self.unet.state_dict()), f'Improper state dict length, got {len(sd) + len(motion_sd)} expected {len(self.unet.state_dict())}'
            sd.update(motion_sd)
            for k, v in self.unet.state_dict().items():
                if 'pos_encoder.pe' in k:
                    sd[k] = v # the size of pe may change
            self.unet.load_state_dict(sd)

    def configure_optimizers(self):
        import bitsandbytes as bnb
        motion_params = []
        remaining_params = []
        for name, p in self.unet.named_parameters():
            if ('motion' in name):
                motion_params.append(p)
            else:
                remaining_params.append(p)
        optimizer = bnb.optim.Adam8bit([
            {'params': motion_params, 'lr': self.optim_args['lr']},
        ], betas=(0.9, 0.999))
        return optimizer