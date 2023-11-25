# %%
import os
import jsonlines
import json
from misc_utils.video_ptp_utils import get_models_of_damo_model
import torch
import numpy as np
from einops import rearrange
from misc_utils.ptp_utils import get_text_embedding_openclip, encode_text_openclip, Text, Edit, Insert, Delete
from misc_utils.video_ptp_utils import compute_diff
from pl_trainer.inference.inference_damo import InferenceDAMO_PTP_v2
import cv2
from misc_utils.image_utils import images_to_gif
from misc_utils.clip_similarity import ClipSimilarity

def save_images_to_folder(source_images, target_images, folder, seed):
    os.makedirs(folder, exist_ok=True)
    for i, (src_image, tgt_image) in enumerate(zip(source_images, target_images)):
        src_img = cv2.cvtColor((src_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        tgt_img = cv2.cvtColor((tgt_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, f'{seed}_0_{i:04d}.jpg'), src_img)
        cv2.imwrite(os.path.join(folder, f'{seed}_1_{i:04d}.jpg'), tgt_img)

def save_images_to_gif(source_images, target_images, folder, seed):
    os.makedirs(folder, exist_ok=True)
    # images_to_gif(source_images, os.path.join(folder, f'{seed}_0.gif'), fps=5)
    # images_to_gif(target_images, os.path.join(folder, f'{seed}_1.gif'), fps=5)
    concat_image = np.concatenate([source_images, target_images], axis=2)
    images_to_gif(concat_image, os.path.join(folder, f'{seed}_concat.gif'), fps=5)

def append_dict_to_jsonl(file_path: str, dict_obj: dict) -> None:
    with open(file_path, 'a') as f:
        f.write(json.dumps(dict_obj))
        f.write("\n")  # Write a new line at the end to prepare for the next JSON object

# %%
def torch_to_numpy(x):
    return (x.float().squeeze().detach().cpu().numpy().transpose(0, 2, 3, 1) / 2 + 0.5).clip(0, 1)
def str_to_prompt(prompt):
    input_text = prompt['input'].strip('.')
    output_text = prompt['output'].strip('.')
    return compute_diff(input_text, output_text)

def get_ptp_prompt(prompt, edit_weight=1.):
    PROMPT = str_to_prompt(prompt)
    for p in PROMPT:
        if isinstance(p, (Edit, Insert)):
            p.weight = edit_weight
    print(PROMPT)
    source_prompt = ' '.join(x.old for x in PROMPT)
    target_prompt = ' '.join(x.new for x in PROMPT)
    context_uncond = get_text_embedding_openclip("", text_encoder, text_model.device)
    old_context = get_text_embedding_openclip(source_prompt, text_encoder, text_model.device)
    context = get_text_embedding_openclip(target_prompt, text_encoder, text_model.device)
    key, value = encode_text_openclip(PROMPT, text_encoder, text_model.device)
    return {
        'source_prompt': source_prompt,
        'target_prompt': target_prompt,
        'context_uncond': context_uncond,
        'old_context': old_context,
        'context': context,
        'key': key,
        'value': value,
    }
def process_one_sample(prompt, seed=None, guidance_scale=9, num_ddim_steps=30, scheduler='ddim', sa_end_time=0.4, ca_end_time=0.8, edit_weight=1.):
    prompt_dict = get_ptp_prompt(prompt, edit_weight=edit_weight)

    if seed is None:
        seed = np.random.randint(0, 1000000)
    torch.random.manual_seed(seed)
    latent = torch.randn(1, 4, 16, 32, 32).cuda()
    inf_pipe = InferenceDAMO_PTP_v2(unet=unet, guidance_scale=guidance_scale, num_ddim_steps=num_ddim_steps, scheduler=scheduler)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        res = inf_pipe(
            latent=latent,
            context = prompt_dict['context'],
            old_context = prompt_dict['old_context'],
            old_to_new_context=[prompt_dict['key'], prompt_dict['value']],
            uncond_context = prompt_dict['context_uncond'],
            sa_end_time=sa_end_time,
            ca_end_time=ca_end_time,
        )
        pred_latent = res['latent']
        pred_latent_old = res['latent_old']

    latents = rearrange(pred_latent, 'b d f h w -> f b d h w')
    latents_old = rearrange(pred_latent_old, 'b d f h w -> f b d h w')
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        pred_image = [vae.decode(latent / 0.18215) for latent in latents]
        pred_image_old = [vae.decode(latent / 0.18215) for latent in latents_old]
    pred_images = torch.stack(pred_image, dim=0)
    pred_images_old = torch.stack(pred_image_old, dim=0)

    old_prompt_images_np = torch_to_numpy(pred_images_old)
    new_prompt_images_np = torch_to_numpy(pred_images)

    return old_prompt_images_np, new_prompt_images_np, seed, prompt_dict


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--prompt_source', type=str, default='ip2p')
    parser.add_argument('--num_sample_each_prompt', '-n', type=int, default=1)
    args = parser.parse_args()

    if args.prompt_source == 'ip2p':
        prompt_meta_file = 'data/gpt-generated-prompts.jsonl'
        with open(prompt_meta_file, 'r') as f:
            prompts = [(i, json.loads(line)) for i, line in enumerate(f.readlines())]
        output_dir = 'video_ptp/raw_generated'
    elif args.prompt_source == 'webvid':
        root_dir = 'webvid_edit_prompt'
        files = os.listdir(root_dir)
        files = sorted(files, key=lambda x: int(x.split('.')[0]))
        prompts = []
        for file in files:
            with open(os.path.join(root_dir, file), 'r') as f:
                prompt_idx = int(file.split('.')[0])
                prompts.append((prompt_idx, json.load(f)))
        output_dir = 'video_ptp/raw_generated_webvid'
    else:
        raise ValueError(f'Unknown prompt source: {args.prompt_source}')


    # %%
    vae_config = 'configs/instruct_v2v.yaml'
    unet_config = 'modules/damo_text_to_video/configuration.json'
    vae_ckpt = 'VAE_PATH'
    unet_ckpt = 'UNet_PATH'
    text_model_ckpt = 'Text_MODEL_PATH'

    vae, unet, text_model = get_models_of_damo_model(
        unet_config=unet_config,
        unet_ckpt=unet_ckpt,
        vae_config=vae_config,
        vae_ckpt=vae_ckpt,
        text_model_ckpt=text_model_ckpt,
    )
    text_encoder = text_model.encode_with_transformer
    clip_sim = ClipSimilarity(name='ViT-L/14').cuda()
    # %%

    for prompt_idx, prompt in prompts:
        if prompt_idx < args.start:
            continue
        if prompt_idx >= args.end:
            break
        print(prompt_idx, prompt)
        output_folder_idx = f'{prompt_idx:07d}'

        prompt_json_file = os.path.join(output_dir, output_folder_idx, 'prompt.json')
        os.makedirs(os.path.dirname(prompt_json_file), exist_ok=True)
        with open(prompt_json_file, 'w') as f:
            json.dump(prompt, f)

        meta_file_path = os.path.join(output_dir, output_folder_idx, 'metadata.jsonl')
        if os.path.exists(meta_file_path):
            with jsonlines.open(meta_file_path, 'r') as f:
                used_seed = [int(line['seed']) for line in f]
            num_existing_samples = len(used_seed)
        else:
            num_existing_samples = 0
            used_seed = []
        print(f'num_existing_samples: {num_existing_samples}, used_seed: {used_seed}')

        for _ in range(num_existing_samples, args.num_sample_each_prompt):
            # generate a random configure
            seed = np.random.randint(0, 1000000)
            while seed in used_seed:
                seed = np.random.randint(0, 1000000)
            used_seed.append(seed)

            rng = np.random.RandomState(seed=seed)
            guidance_scale = rng.randint(5, 13)
            sa_end_time = float('{:.2f}'.format(rng.choice(np.linspace(0.3, 0.45, 4))))
            ca_end_time = float('{:.2f}'.format(rng.choice(np.linspace(0.6, 0.85, 6))))
            edit_weight = rng.randint(1, 6)
            generate_config = {
                'seed': seed,
                'guidance_scale': guidance_scale,
                'sa_end_time': sa_end_time,
                'ca_end_time': ca_end_time,
                'edit_weight': edit_weight,
            }
            print(generate_config)

            # %%
            source_prompt_images, target_prompt_images, seed, prompt_dict = process_one_sample(
                prompt,
                **generate_config,
            )

            source_prompt_images_torch = torch.from_numpy(source_prompt_images.transpose(0, 3, 1, 2)).cuda()
            target_prompt_images_torch = torch.from_numpy(target_prompt_images.transpose(0, 3, 1, 2)).cuda()

            with torch.no_grad():
                sim_0, sim_1, sim_dir, sim_image = clip_sim(source_prompt_images_torch, target_prompt_images_torch, [prompt_dict['source_prompt']], [prompt_dict['target_prompt']])

            simi_dict = {
                'sim_0': sim_0.mean().item(),
                'sim_1': sim_1.mean().item(),
                'sim_dir': sim_dir.mean().item(),
                'sim_image': sim_image.mean().item(),
            }

            generate_config.update(simi_dict)

            output_folder_img = os.path.join(output_dir, output_folder_idx, 'image')
            output_folder_gif = os.path.join(output_dir, output_folder_idx, 'gif')

            if (
                sim_0.mean().item() > 0.2 and sim_1.mean().item() > 0.2 and sim_dir.mean().item() > 0.2 and sim_image.mean().item() > 0.5
            ):
                save_images_to_folder(source_prompt_images, target_prompt_images, output_folder_img, seed)
                save_images_to_gif(source_prompt_images, target_prompt_images, output_folder_gif, seed)
            
            append_dict_to_jsonl(meta_file_path, generate_config)


