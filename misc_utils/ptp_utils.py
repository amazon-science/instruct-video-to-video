from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class Edit:
    old: str
    new: str
    weight: float = 1.0


@dataclass
class Insert:
    text: str
    weight: float = 1.0

    @property
    def old(self):
        return ""

    @property
    def new(self):
        return self.text


@dataclass
class Delete:
    text: str
    weight: float = 1.0

    @property
    def old(self):
        return self.text

    @property
    def new(self):
        return ""


@dataclass
class Text:
    text: str
    weight: float = 1.0

    @property
    def old(self):
        return self.text

    @property
    def new(self):
        return self.text

@torch.inference_mode()
def get_text_embedding(prompt, tokenizer, text_encoder):
    text_input_ids = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    text_embeddings = text_encoder(text_input_ids.to(text_encoder.device))[0]
    return text_embeddings

@torch.inference_mode()
def encode_text(text_pieces, tokenizer, text_encoder):
    n_old_tokens = 0
    n_new_tokens = 0
    new_id_to_old_id = []
    weights = []
    for piece in text_pieces:
        old, new = piece.old, piece.new
        old_tokens = tokenizer.tokenize(old)
        new_tokens = tokenizer.tokenize(new)
        if len(old_tokens) == 0 and len(new_tokens) == 0:
            continue
        elif old == new:
            n_old_tokens += len(old_tokens)
            n_new_tokens += len(new_tokens)
            new_id_to_old_id.extend(range(n_old_tokens - len(old_tokens), n_old_tokens))
        elif len(old_tokens) == 0:
            # insert
            new_id_to_old_id.extend([-1] * len(new_tokens))
            n_new_tokens += len(new_tokens)
        elif len(new_tokens) == 0:
            # delete
            n_old_tokens += len(old_tokens)
        else:
            # replace
            n_old_tokens += len(old_tokens)
            n_new_tokens += len(new_tokens)
            start = n_old_tokens - len(old_tokens)
            end = n_old_tokens
            ids = np.linspace(start, end, len(new_tokens), endpoint=False).astype(int)
            new_id_to_old_id.extend(list(ids))
        weights.extend([piece.weight] * len(new_tokens))

    old_prompt = " ".join([piece.old for piece in text_pieces])
    new_prompt = " ".join([piece.new for piece in text_pieces])
    old_text_input_ids = tokenizer(
        old_prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    new_text_input_ids = tokenizer(
        new_prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    old_text_embeddings = text_encoder(old_text_input_ids.to(text_encoder.device))[0]
    new_text_embeddings = text_encoder(new_text_input_ids.to(text_encoder.device))[0]
    value = new_text_embeddings.clone()  # batch (1), seq, dim
    key = new_text_embeddings.clone()

    for i, (j, weight) in enumerate(zip(new_id_to_old_id, weights)):
        if 0 <= j < old_text_embeddings.shape[1]:
            key[0, i] = old_text_embeddings[0, j]
        value[0, i] *= weight
    return key, value

@torch.inference_mode()
def get_text_embedding_openclip(prompt, text_encoder, device='cuda'):
    import open_clip
    text_input_ids = open_clip.tokenize(prompt)
    text_embeddings = text_encoder(text_input_ids.to(device))
    return text_embeddings

@torch.inference_mode()
def encode_text_openclip(text_pieces, text_encoder, device='cuda'):
    import open_clip
    n_old_tokens = 0
    n_new_tokens = 0
    new_id_to_old_id = []
    weights = []
    for piece in text_pieces:
        old, new = piece.old, piece.new
        old_tokens = open_clip.tokenize(old)
        new_tokens = open_clip.tokenize(new)
        if len(old_tokens) == 0 and len(new_tokens) == 0:
            continue
        elif old == new:
            n_old_tokens += len(old_tokens)
            n_new_tokens += len(new_tokens)
            new_id_to_old_id.extend(range(n_old_tokens - len(old_tokens), n_old_tokens))
        elif len(old_tokens) == 0:
            # insert
            new_id_to_old_id.extend([-1] * len(new_tokens))
            n_new_tokens += len(new_tokens)
        elif len(new_tokens) == 0:
            # delete
            n_old_tokens += len(old_tokens)
        else:
            # replace
            n_old_tokens += len(old_tokens)
            n_new_tokens += len(new_tokens)
            start = n_old_tokens - len(old_tokens)
            end = n_old_tokens
            ids = np.linspace(start, end, len(new_tokens), endpoint=False).astype(int)
            new_id_to_old_id.extend(list(ids))
        weights.extend([piece.weight] * len(new_tokens))

    old_prompt = " ".join([piece.old for piece in text_pieces])
    new_prompt = " ".join([piece.new for piece in text_pieces])
    old_text_input_ids = open_clip.tokenize(old_prompt)
    new_text_input_ids = open_clip.tokenize(new_prompt)

    old_text_embeddings = text_encoder(old_text_input_ids.to(device))
    new_text_embeddings = text_encoder(new_text_input_ids.to(device))
    value = new_text_embeddings.clone()  # batch (1), seq, dim
    key = new_text_embeddings.clone()

    for i, (j, weight) in enumerate(zip(new_id_to_old_id, weights)):
        if 0 <= j < old_text_embeddings.shape[1]:
            key[0, i] = old_text_embeddings[0, j]
        value[0, i] *= weight
    return key, value