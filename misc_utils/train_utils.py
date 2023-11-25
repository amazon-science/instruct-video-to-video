import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from misc_utils.model_utils import instantiate_from_config, get_obj_from_str

def get_models(args):
    unet = instantiate_from_config(args.unet)
    model_dict = {
        'unet': unet,
    }

    if args.get('vae'):
        vae = instantiate_from_config(args.vae)
        model_dict['vae'] = vae

    if args.get('text_model'):
        text_model = instantiate_from_config(args.text_model)
        model_dict['text_model'] = text_model

    if args.get('ctrlnet'):
        ctrlnet = instantiate_from_config(args.ctrlnet)
        model_dict['ctrlnet'] = ctrlnet

    return model_dict

def get_DDPM(diffusion_configs, log_args={}, **models):
    diffusion_model_class = diffusion_configs['target']
    diffusion_args = diffusion_configs['params']
    DDPM_model = get_obj_from_str(diffusion_model_class)
    ddpm_model = DDPM_model(
        log_args=log_args,
        **models,
        **diffusion_args
    )
    return ddpm_model


def get_logger(args):
    wandb_logger = WandbLogger(
        project=args["expt_name"],
    )
    return wandb_logger

def get_callbacks(args, wandb_logger):
    callbacks = []
    for callback in args['callbacks']:
        if callback.get('require_wandb', False):
            # we need to pass wandb logger to the callback
            callback_obj = get_obj_from_str(callback.target)
            callbacks.append(
                callback_obj(wandb_logger=wandb_logger, **callback.params)
            )
        else:
            callbacks.append(
                instantiate_from_config(callback)
            )
    return callbacks

def get_dataset(args):
    from torch.utils.data import DataLoader
    data_args = args['data']
    train_set = instantiate_from_config(data_args['train'])
    val_set = instantiate_from_config(data_args['val'])
    train_loader = DataLoader(
        train_set, batch_size=data_args['batch_size'], shuffle=True,
        num_workers=4*len(args['trainer_args']['devices']), pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=data_args['val_batch_size'],
        num_workers=len(args['trainer_args']['devices']), pin_memory=True
    )
    return train_loader, val_loader, train_set, val_set

def unit_test_create_model(config_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(config_path)
    models = get_models(conf)
    ddpm = get_DDPM(conf['diffusion'], log_args=conf, **models)
    ddpm = ddpm.to(device)
    return ddpm

def unit_test_create_dataset(config_path, split='train'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    conf = OmegaConf.load(config_path)
    train_loader, val_loader, train_set, val_set = get_dataset(conf)
    if split == 'train':
        batch = next(iter(train_loader))
    else:
        batch = next(iter(val_loader))
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

def unit_test_training_step(config_path):
    ddpm = unit_test_create_model(config_path)
    batch = unit_test_create_dataset(config_path)
    res = ddpm.training_step(batch, 0)
    return res

def unit_test_val_step(config_path):
    ddpm = unit_test_create_model(config_path)
    batch = unit_test_create_dataset(config_path, split='val')
    res = ddpm.validation_step(batch, 0)
    return res

NEGATIVE_PROMPTS = "(((deformed))), blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar, multiple breasts, (mutated hands and fingers:1.5), (long body :1.3), (mutation, poorly drawn :1.2), black-white, bad anatomy, liquid body, liquidtongue, disfigured, malformed, mutated, anatomical nonsense, text font ui, error, malformed hands, long neck, blurred, lowers, low res, bad anatomy, bad proportions, bad shadow, uncoordinated body, unnatural body, fused breasts, bad breasts, huge breasts, poorly drawn breasts, extra breasts, liquid breasts, heavy breasts, missingbreasts, huge haunch, huge thighs, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, fusedears, bad ears, poorly drawn ears, extra ears, liquid ears, heavy ears, missing ears, old photo, low res, black and white, black and white filter, colorless"
