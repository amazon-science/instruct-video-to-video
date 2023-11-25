import torch
import os
import argparse
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from misc_utils.train_utils import get_models, get_DDPM, get_logger, get_callbacks, get_dataset

if __name__ == '__main__':
    # seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, 
        default='config/train.json')
    parser.add_argument(
        '-r', '--resume', action="store_true"
    )
    parser.add_argument(
        '-n', '--nnode', type=int, default=1
    )
    parser.add_argument(
        '--ckpt', type=str, default=None
    )
    parser.add_argument(
        '--manual_load', action="store_true"
    )

    ''' parser configs '''
    args_raw = parser.parse_args()
    args = OmegaConf.load(args_raw.config)
    args.update(vars(args_raw))
    expt_path = os.path.join(args.expt_dir, args.expt_name)
    os.makedirs(expt_path, exist_ok=True)

    '''1. create denoising model'''
    models = get_models(args)

    diffusion_configs = args.diffusion
    ddpm_model = get_DDPM(
        diffusion_configs=diffusion_configs,
        log_args=args,
        **models
    )

    '''2. dataset and dataloader'''
    train_loader, val_loader, train_set, val_set = get_dataset(args)
    
    '''3. create callbacks'''
    wandb_logger = get_logger(args)
    callbacks = get_callbacks(args, wandb_logger)

    '''4. trainer'''
    trainer_args = {
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": [0],
        "limit_val_batches": 1,
        "strategy": "ddp",
        "check_val_every_n_epoch": 1,
        "num_nodes": args.nnode
        # "benchmark" :True
    }
    config_trainer_args = args.trainer_args if args.get('trainer_args') is not None else {}
    trainer_args.update(config_trainer_args)
    print(f'Training args are {trainer_args}')
    trainer = Trainer(
        logger = wandb_logger,
        callbacks = callbacks,
        **trainer_args
    )
    '''5. start training'''
    if args['resume']:
        print('INFO: Try to resume from checkpoint')
        if args['ckpt'] is not None:
            ckpt_path = args['ckpt']
        else:
            ckpt_path = os.path.join(expt_path, 'last.ckpt')
        if os.path.exists(ckpt_path):
            print(f'INFO: Found checkpoint {ckpt_path}')
            if args['manual_load']:
                print('INFO: Manually load checkpoint')
                ckpt = torch.load(ckpt_path, map_location='cpu')
                ddpm_model.load_state_dict(ckpt['state_dict'])
                ckpt_path = None # do not need to load checkpoint in Trainer
        else:
            ckpt_path = None
    else:
        ckpt_path = None
    trainer.fit(
        ddpm_model, train_loader, val_loader,
        ckpt_path=ckpt_path
    )
        