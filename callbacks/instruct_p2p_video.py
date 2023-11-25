from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from .common import tensor2image, clip_image, unnorm
from einops import rearrange

def frame_dim_to_batch_dim(x):
    return rearrange(x, 'b f c h w -> (b f) c h w')

class InstructP2PLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    def on_train_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        # record images in first batch
        if batch_idx == 0:
            input_image = tensor2image(frame_dim_to_batch_dim(clip_image(unnorm(
                batch['input_video'][:self.max_num_images]
            ))))
            edited_image = tensor2image(frame_dim_to_batch_dim(clip_image(unnorm(
                batch['edited_video'][:self.max_num_images]
            ))))
            pred_image = tensor2image(frame_dim_to_batch_dim(clip_image(unnorm(
                outputs['pred'][:self.max_num_images]
            ))))
            self.wandb_logger.experiment.log({
                'train/input_image': input_image,
                'train/edited_image': edited_image,
                'train/pred': pred_image,
            })

    def on_validation_batch_end(
        self, trainer, pl_module, 
        outputs, batch, batch_idx
    ):
        """Called when the validation batch ends."""
        if batch_idx == 0:
            input_image = tensor2image(frame_dim_to_batch_dim(clip_image(unnorm(
                batch['input_video'][:self.max_num_images]
            ))))
            edited_image = tensor2image(frame_dim_to_batch_dim(clip_image(unnorm(
                batch['edited_video'][:self.max_num_images]
            ))))
            pred_image = tensor2image(frame_dim_to_batch_dim(clip_image(unnorm(
                outputs['pred'][:self.max_num_images]
            ))))
            self.wandb_logger.experiment.log({
                'val/input_image': input_image,
                'val/edited_image': edited_image,
                'val/pred': pred_image,
            })
