import torch
from modules.damo_text_to_video.unet_sd import UNetSD
from misc_utils.train_utils import instantiate_from_config
from omegaconf import OmegaConf
from modules.damo_text_to_video.text_model import FrozenOpenCLIPEmbedder
from typing import List, Tuple, Union
from diff_match_patch import diff_match_patch
import difflib
from misc_utils.ptp_utils import get_text_embedding_openclip, encode_text_openclip, Text, Edit, Insert, Delete


def get_models_of_damo_model(
    unet_config: str,
    unet_ckpt: str,
    vae_config: str,
    vae_ckpt: str,
    text_model_ckpt: str,
):
    vae_conf = OmegaConf.load(vae_config)
    unet_conf = OmegaConf.load(unet_config)

    vae = instantiate_from_config(vae_conf.vae)
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'))
    vae = vae.half().cuda()

    unet = UNetSD(**unet_conf.model.model_cfg)
    unet.load_state_dict(torch.load(unet_ckpt, map_location='cpu'))
    unet = unet.half().cuda()

    text_model = FrozenOpenCLIPEmbedder(version=text_model_ckpt, layer='penultimate')
    text_model = text_model.half().cuda()

    return vae, unet, text_model

def compute_diff_old(old_sentence: str, new_sentence: str) -> List[Tuple[Union[Text, Edit, Insert, Delete], str, str]]:
    dmp = diff_match_patch()
    diff = dmp.diff_main(old_sentence, new_sentence)
    dmp.diff_cleanupSemantic(diff)

    result = []
    i = 0
    while i < len(diff):
        op, data = diff[i]
        if op == 0:  # Equal
            # result.append((Text, data, data))
            result.append(Text(text=data))
        elif op == -1:  # Delete
            if i + 1 < len(diff) and diff[i + 1][0] == 1:  # If next operation is Insert
                result.append(Edit(old=data, new=diff[i + 1][1]))  # Append as Edit operation
                i += 1  # Skip next operation because we've handled it here
            else:
                result.append(Delete(text=data))
        elif op == 1:  # Insert
            if i == 0 or diff[i - 1][0] != -1:  # If previous operation wasn't Delete
                result.append(Insert(text=data))
        i += 1

    return result

def compute_diff(old_sentence: str, new_sentence: str) -> List[Union[Text, Edit, Insert, Delete]]:
    differ = difflib.Differ()
    diff = list(differ.compare(old_sentence.split(), new_sentence.split()))

    result = []
    i = 0
    while i < len(diff):
        if diff[i][0] == ' ':  # Equal
            equal_words = [diff[i][2:]]
            while i + 1 < len(diff) and diff[i + 1][0] == ' ':
                i += 1
                equal_words.append(diff[i][2:])
            result.append(Text(text=' '.join(equal_words)))
        elif diff[i][0] == '-':  # Delete
            deleted_words = [diff[i][2:]]
            while i + 1 < len(diff) and diff[i + 1][0] == '-':
                i += 1
                deleted_words.append(diff[i][2:])
            result.append(Delete(text=' '.join(deleted_words)))
        elif diff[i][0] == '+':  # Insert
            inserted_words = [diff[i][2:]]
            while i + 1 < len(diff) and diff[i + 1][0] == '+':
                i += 1
                inserted_words.append(diff[i][2:])
            result.append(Insert(text=' '.join(inserted_words)))
        i += 1

    # Post-process to merge adjacent inserts and deletes into edits
    i = 0
    while i < len(result) - 1:
        if isinstance(result[i], Delete) and isinstance(result[i+1], Insert):
            result[i:i+2] = [Edit(old=result[i].text, new=result[i+1].text)]
        elif isinstance(result[i], Insert) and isinstance(result[i+1], Delete):
            result[i:i+2] = [Edit(old=result[i+1].text, new=result[i].text)]
        else:
            i += 1

    return result