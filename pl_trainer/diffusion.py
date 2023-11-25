import torch
from torch import nn
import pytorch_lightning as pl
from misc_utils.model_utils import default, instantiate_from_config
from diffusers import DDPMScheduler

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class DDPM(pl.LightningModule):
    def __init__(
        self, 
        unet,
        beta_schedule_args={
            'beta_start': 0.00085,
            'beta_end': 0.0012,
            'num_train_timesteps': 1000,
            'beta_schedule': 'scaled_linear',
            'clip_sample': False,
            'thresholding': False,
        },
        prediction_type='epsilon', 
        loss_fn='l2',
        optim_args={},
        **kwargs
    ):
        '''
        denoising_fn: a denoising model such as UNet
        beta_schedule_args: a dictionary which contains
            the configurations of the beta schedule
        '''
        super().__init__(**kwargs)
        self.unet = unet
        self.prediction_type = prediction_type
        beta_schedule_args.update({'prediction_type': prediction_type})
        self.set_beta_schedule(beta_schedule_args)
        self.num_timesteps = beta_schedule_args['num_train_timesteps']
        self.optim_args = optim_args
        self.loss = loss_fn
        if loss_fn == 'l2' or loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_fn == 'l1' or loss_fn == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif isinstance(loss_fn, dict):
            self.loss_fn = instantiate_from_config(loss_fn)
        else:
            raise NotImplementedError

    def set_beta_schedule(self, beta_schedule_args):
        self.beta_schedule_args = beta_schedule_args
        self.scheduler = DDPMScheduler(**beta_schedule_args)

    @torch.no_grad()
    def add_noise(self, x, t, noise=None):
        noise = default(noise, torch.randn_like(x))
        return self.scheduler.add_noise(x, noise, t)

    def predict_x_0_from_x_t(self, model_output: torch.Tensor, t: torch.LongTensor, x_t: torch.Tensor):
        ''' recover x_0 from predicted noise. Reverse of Eq(4) in DDPM paper
        \hat(x_0) = 1 / sqrt[\bar(a)]*x_t - sqrt[(1-\bar(a)) / \bar(a)]*noise'''
        # return self.scheduler.step(model_output, int(t), x_t).pred_original_sample
        if self.prediction_type == 'sample':
            return model_output
        # for training target == epsilon
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod[t]).flatten()
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod[t] - 1.).flatten()
        while len(sqrt_recip_alphas_cumprod.shape) < len(x_t.shape):
            sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod.unsqueeze(-1)
            sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod.unsqueeze(-1)
        return sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * model_output

    def predict_x_tm1_from_x_t(self, model_output, t, x_t):
        '''predict x_{t-1} from x_t and model_output'''
        return self.scheduler.step(model_output, t, x_t).prev_sample

class DDPMTraining(DDPM):
    def __init__(
        self, 
        unet, 
        beta_schedule_args, 
        prediction_type='epsilon', 
        loss_fn='l2',
        optim_args={
            'lr': 1e-3,
            'weight_decay': 5e-4
        },
        log_args={}, # for record all arguments with self.save_hyperparameters
        ddim_sampling_steps=20,
        guidance_scale=5.,
        **kwargs
    ):
        super().__init__(
            unet=unet, 
            beta_schedule_args=beta_schedule_args, 
            prediction_type=prediction_type, 
            loss_fn=loss_fn, 
            optim_args=optim_args,
            **kwargs)
        self.log_args = log_args
        self.call_save_hyperparameters()

        self.ddim_sampling_steps = ddim_sampling_steps
        self.guidance_scale = guidance_scale

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['unet'])

    def process_batch(self, x_0, mode):
        assert mode in ['train', 'val', 'test']
        b, *_ = x_0.shape
        noise = torch.randn_like(x_0)
        if mode == 'train':
            t = torch.randint(0, self.num_timesteps, (b,), device=x_0.device).long()
            x_t = self.add_noise(x_0, t, noise=noise)
        else:
            t = torch.full((b,), self.num_timesteps-1, device=x_0.device, dtype=torch.long)
            x_t = self.add_noise(x_0, t, noise=noise)

        model_kwargs = {}
        '''the order of return is 
            1) model input, 
            2) model pred target, 
            3) model time condition
            4) raw image before adding noise
            5) model_kwargs
        '''
        if self.prediction_type == 'epsilon':
            return {
                'model_input': x_t,
                'model_target': noise,
                't': t,
                'model_kwargs': model_kwargs
            }
        else:
            return {
                'model_input': x_t,
                'model_target': x_0,
                't': t,
                'model_kwargs': model_kwargs
            }

    def forward(self, x):
        return self.validation_step(x, 0)

    def get_loss(self, pred, target, t):
        loss_raw = self.loss_fn(pred, target)
        loss_flat = mean_flat(loss_raw)

        loss = loss_flat
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        self.clip_denoised = False
        processed_batch = self.process_batch(batch, mode='train')
        x_t = processed_batch['model_input']
        y = processed_batch['model_target']
        t = processed_batch['t']
        model_kwargs = processed_batch['model_kwargs']
        pred = self.unet(x_t, t, **model_kwargs)
        loss = self.get_loss(pred, y, t)
        x_0_hat = self.predict_x_0_from_x_t(pred, t, x_t)

        self.log(f'train_loss', loss)
        return {
            'loss': loss,
            'model_input': x_t,
            'model_output': pred,
            'x_0_hat': x_0_hat
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler(**self.beta_schedule_args)
        scheduler.set_timesteps(self.ddim_sampling_steps)
        processed_batch = self.process_batch(batch, mode='val')
        x_t = torch.randn_like(processed_batch['model_input'])
        x_hist = []
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            t_ = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
            model_output = self.unet(x_t, t_, **processed_batch['model_kwargs'])
            x_hist.append(
                self.predict_x_0_from_x_t(model_output, t_, x_t)
            )
            x_t = scheduler.step(model_output, t, x_t).prev_sample

        return {
            'x_pred': x_t,
            'x_hist': torch.stack(x_hist, dim=1),
        }

    def test_step(self, batch, batch_idx):
        '''Test is usually not used in a sampling problem'''
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_args)
        return optimizer

class DDPMLDMTraining(DDPMTraining):
    def __init__(
        self, *args,
        vae,
        unet_init_weights=None,
        vae_init_weights=None,
        scale_factor=0.18215,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vae = vae
        self.scale_factor = scale_factor
        self.initialize_unet(unet_init_weights)
        self.initialize_vqvae(vae_init_weights)

    def initialize_unet(self, unet_init_weights):
        if unet_init_weights is not None:
            print(f'INFO: initialize denoising UNet from {unet_init_weights}')
            sd = torch.load(unet_init_weights, map_location='cpu')
            self.unet.load_state_dict(sd)

    def initialize_vqvae(self, vqvae_init_weights):
        if vqvae_init_weights is not None:
            print(f'INFO: initialize VQVAE from {vqvae_init_weights}')
            sd = torch.load(vqvae_init_weights, map_location='cpu')
            self.vae.load_state_dict(sd)
            for param in self.vae.parameters():
                param.requires_grad = False

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['unet', 'vae'])

    @torch.no_grad()
    def encode_image_to_latent(self, x):
        return self.vae.encode(x) * self.scale_factor

    @torch.no_grad()
    def decode_latent_to_image(self, x):
        x = x / self.scale_factor
        return self.vae.decode(x)

    def process_batch(self, x_0, mode):
        x_0 = self.encode_image_to_latent(x_0)
        res = super().process_batch(x_0, mode)
        return res

    def training_step(self, batch, batch_idx):
        res_dict = super().training_step(batch, batch_idx)
        res_dict['x_0_hat'] = self.decode_latent_to_image(res_dict['x_0_hat'])
        return res_dict

class DDIMLDMTextTraining(DDPMLDMTraining):
    def __init__(
        self, *args,
        text_model,
        text_model_init_weights=None,
        **kwargs
    ):
        super().__init__(
            *args, **kwargs
        )
        self.text_model = text_model
        self.initialize_text_model(text_model_init_weights)

    def initialize_text_model(self, text_model_init_weights):
        if text_model_init_weights is not None:
            print(f'INFO: initialize text model from {text_model_init_weights}')
            sd = torch.load(text_model_init_weights, map_location='cpu')
            self.text_model.load_state_dict(sd)
            for param in self.text_model.parameters():
                param.requires_grad = False

    def call_save_hyperparameters(self):
        '''write in a separate function so that the inherit class can overwrite it'''
        self.save_hyperparameters(ignore=['unet', 'vae', 'text_model'])

    @torch.no_grad()
    def encode_text(self, x):
        if isinstance(x, tuple):
            x = list(x)
        return self.text_model.encode(x)

    def process_batch(self, batch, mode):
        x_0 = batch['image']
        text = batch['text']
        processed_batch = super().process_batch(x_0, mode)
        processed_batch['model_kwargs'].update({
            'context': {'text': self.encode_text([text])}
        })
        return processed_batch

    def sampling(self, image_shape=(1, 4, 64, 64), text='', negative_text=None):
        '''
        Usage:
            sampled = self.sampling(text='a cat on the tree', negative_text='')

            x = sampled['x_pred'][0].permute(1, 2, 0).detach().cpu().numpy()
            x = x / 2 + 0.5
            plt.imshow(x)

            y = sampled['x_hist'][0, 10].permute(1, 2, 0).detach().cpu().numpy()
            y = y / 2 + 0.5
            plt.imshow(y)
        '''
        from diffusers import DDIMScheduler
        scheduler = DDIMScheduler(**self.beta_schedule_args)
        scheduler.set_timesteps(self.ddim_sampling_steps)
        x_t = torch.randn(*image_shape, device=self.device)
        
        do_cfg = self.guidance_scale > 1. and negative_text is not None

        if do_cfg:
            context = {'text': self.encode_text([text, negative_text])}
        else:
            context = {'text': self.encode_text([text])}
        x_hist = []
        timesteps = scheduler.timesteps
        for i, t in enumerate(timesteps):
            if do_cfg:
                model_input = torch.cat([x_t]*2)
            else:
                model_input = x_t
            t_ = torch.full((model_input.shape[0],), t, device=x_t.device, dtype=torch.long)
            model_output = self.unet(model_input, t_, context)

            if do_cfg:
                model_output_positive, model_output_negative = model_output.chunk(2)
                model_output = model_output_negative + self.guidance_scale * (model_output_positive - model_output_negative)
            x_hist.append(
                self.decode_latent_to_image(self.predict_x_0_from_x_t(model_output, t_[:x_t.shape[0]], x_t))
            )
            x_t = scheduler.step(model_output, t, x_t).prev_sample

        return {
            'x_pred': self.decode_latent_to_image(x_t),
            'x_hist': torch.stack(x_hist, dim=1),
        }
