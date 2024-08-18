from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
# import sentencepiece as spm
from tqdm import tqdm

from model import Transformer
from model_args import ModelArgs


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            # load the model checkpoint
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f"Loading model from {chk_path}")
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f} seconds")

            # reset time because we will now load model parameters
            prev_time = time.time()
        
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        # load tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path) # type: ignore
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # we are computing rope freqs, so we don't load it from model checkpoint
            # their key name is also different
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {(time.time() - prev_time):.2f} seconds')

        return LLaMA(model, tokenizer, model_args)

if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    print('Built and loaded model')

    # generate text



