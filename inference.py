from typing import Optional, List
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
# import sentencepiece as spm
from tqdm import tqdm
# import tqdm as tqdm


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
    
    def text_completion(self, prompts: List[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        
        # Convert each prompt into tokens
        # This is inference code --> we need to pass BOS to the inputs yes, 
        # but don't need to pass EOS tokens to the input
        prompt_tokens = [self.tokenizer.encode(prompt, out_type = int, add_bos = True, add_eos = False) for prompt in prompts]
        
        # make sure batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, "Batch size is too large"

        # Make sure prompt length is not larger than maximum sequence length
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, "Prompts are too long"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that contains the generated tokens, along with initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(size = (batch_size, total_len), 
                            fill_value = pad_id, 
                            dtype=torch.long, device=device)

        for i, t in enumerate(prompt_tokens):
            # Populate the initial tokens with prompt tokens
            tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # True if the token is a prompt token, False otherwise

        for cur_pos in tqdm(range(1, total_len), desc = "generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)

            if temperature > 0:
                # Temperature applied before softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim = -1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the max probability
                next_token = torch.argmax(logits[:, -1], dim = -1)

            # only replace the token if it is a padding token
            next_token = torch.where(condition = prompt_tokens_mask[:, cur_pos], 
                                     input = tokens[:, cur_pos], 
                                     other = next_token)
            tokens[:, cur_pos] = next_token

            # EOS is reached only if we found an EOS token for a padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []

        for prompt_idx, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]

            # generate (one token at a time)
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text
        

    def _sample_top_p(self, probs, p):
        # sort in descending order
        probs_sort, probs_idx = torch.sort(probs, dim = -1, descending = True)
        # take cumulative sum 
        probs_sum = torch.cumsum(probs_sort, dim = -1)

        # (probs_sum - probs_sort) is like shifted cumsum (starts from 0)
        # We need tokens with cumsum upto p
        mask = (probs_sum - probs_sort) > p
        probs_sort[mask] = 0.0 # zero out all tokens non-top p tokens

        # redistribute the probs of top p
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        # now take a sample
        next_tokens = torch.multinomial(probs_sort, num_samples = 1)
        
        # these are good tokens, but we had sorted the probs so order is gone
        next_tokens = torch.gather(input = probs_idx, dim = -1, index = next_tokens)
        return next_tokens



if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    # prompts 
    prompts = [
        "By the law of natural selection, we mean to say that ",
        "Geostatics is a study of"
    ]


    # LLaMA class's build method returns object of type LLaMA only!
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device
    )

    print('Built and loaded model')

    # generate text i.e. inference
    out_tokens, out_text = (model.text_completion(prompts, max_gen_len = 64))
    assert len(out_text) == len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')



