import torch
from transformers import AutoTokenizer

import config
from model import Transformer


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('./mini-deepseek-tokenizer')

    model = Transformer(config).to(device)
    model.load_state_dict(torch.load('./model/mini_deepseek_pretrain.pt'))
    total_param = 0
    for param in model.parameters():
        if param.requires_grad:
            total_param += param.numel()
    print(total_param)

    prompt = '人类大脑的主要功能是什么？'

    # messages = []
    # messages = messages[-config.history_cnt:] if config.history_cnt else []
    # messages.append({'role': 'user', 'content': prompt})

    # new_prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize = False,
    #     add_generation_prompt = True
    # )

    new_prompt = tokenizer.bos_token + prompt
    with torch.no_grad():
        x = torch.tensor(tokenizer(new_prompt)['input_ids']).to(device).unsqueeze(0)
        out = model.inference(
            x,
            eos_token_id = tokenizer.eos_token_id,
            max_new_tokens = config.max_new_tokens,
            temperature = config.temperature,
            top_p = config.top_p,
            rp = config.rp,
            use_cache = True
        )

        history_idx = 0
        for y in out:
            answer = tokenizer.decode(y[0].tolist(), skip_special_tokens = True)
            if (answer and answer[-1] == '�') or not answer:
                continue
            print(answer[history_idx:], end = '', flush = True)
            history_idx = len(answer)