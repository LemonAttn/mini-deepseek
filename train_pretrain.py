import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from dataset import PretrainDataset
from model import Transformer


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ctx = torch.amp.autocast(device_type = device)

    tokenizer = AutoTokenizer.from_pretrained('./mini-deepseek-tokenizer')
    train_dataloader = DataLoader(
        PretrainDataset(
            data_path = '/root/autodl-tmp/data/pretrain_hq.jsonl',
            tokenizer = tokenizer,
            max_seq_len = config.max_seq_len
        ),
        batch_size = config.max_batch_size,
        pin_memory = True,
        drop_last = False,
        shuffle = True,
        num_workers = config.num_workers
    )

    model = Transformer(config).to(device)

    scaler = torch.amp.GradScaler()
    loss_func = nn.CrossEntropyLoss(reduction = 'none')
    opt = torch.optim.AdamW(model.parameters(), lr = config.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max = config.epoch * len(train_dataloader) / config.accumulation_steps, eta_min = config.lr / 10
    )

    with open('pretrain_moe_log.txt', 'w', encoding = 'utf-8') as f:
        f.write('epoch\tstep\tloss\ttime\tlr\t\n')
    for e in range(config.epoch):
        start_time = time.time()
        for step, (x, y, loss_mask) in enumerate(train_dataloader):
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)

            with ctx:
                out, aux_loss = model(x)
                loss = loss_func(out.view(-1, out.shape[-1]), y.view(-1)).view(loss_mask.shape)
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                loss += aux_loss
                loss = loss / config.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % config.accumulation_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                scaler.step(opt)
                scaler.update()

                opt.zero_grad(set_to_none = True)
                lr_scheduler.step()

            if step % config.print_step == 0:
                print(f'epoch:{e+1}/{config.epoch} step:{step+1}/{len(train_dataloader)} loss:{loss.item():.4f} '
                      f'time:{time.time() - start_time:.4f} lr:{lr_scheduler.get_last_lr()[0]:.6f}')
                with open('pretrain_moe_log.txt', 'a', encoding = 'utf-8') as f:
                    f.write(f'{e+1}\t{step+1}\t{loss.item():.4f}\t{time.time() - start_time:.4f}\t'
                            f'{lr_scheduler.get_last_lr()[0]:.6f}\t\n')

            if (step + 1) % config.save_step == 0:
                model.eval()
                model_path = f'./model/mini_deepseek_pretrain_moe.pt' if config.use_moe \
                    else './model/mini_deepseek_pretrain.pt'
                torch.save(model.state_dict(), model_path)
                model.train()
