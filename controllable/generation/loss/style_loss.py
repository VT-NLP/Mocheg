

import torch
import time
from torch import cuda
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import datetime
import os  
import numpy as np

from controllable.classification.train_verify import classifier_predict 

def get_sc_loss(tgt,tokenizer,logits,cls, labels,device,ignore_pad_token_for_loss, classifier_tokenizer,max_len,sample_method="sample"):
    
    label_pad_token_id = -100 if  ignore_pad_token_for_loss else tokenizer.pad_token_id
  
    sen_len = tgt.ne(label_pad_token_id).sum(-1)
    loss_sc,tgt_reward = cal_sc_loss(logits, sen_len, cls, tokenizer,  labels, device,classifier_tokenizer,max_len,sample_method)
    return loss_sc ,tgt_reward



def cal_sc_loss(out, idx, cls, tokenizer,  labels,device, classifier_tokenizer,max_len,sample_method="sample"):
    '''Caculate the loss of SC-based reward'''
    if sample_method=="greedy":
        sample_probs, sample_idx = torch.max(out, dim=-1)
    else:
        out = F.softmax(out, dim=-1)
        sample_probs, sample_idx = sample_3d(out,device)

    tgt = []
    for i, s in zip(idx.cpu(), sample_idx):
        e = torch.arange(len(s))[s.eq(tokenizer.eos_token_id)]
        e = e[0] if 0<len(e) and 4<e[0]<i else i-1
        tgt.append(s[:e].cpu().tolist())
    tgt_text_list=tokenizer.batch_decode(tgt)
    tgt_cls=classifier_predict(tgt_text_list,max_len,device,classifier_tokenizer,cls)
 
    reward_for_label_1=tgt_cls[:, 1] - tgt_cls[:, 0]- tgt_cls[:, 2]
    reward_for_label_0=tgt_cls[:, 0] - tgt_cls[:, 1]- tgt_cls[:, 2]
    reward_for_label_2=tgt_cls[:, 2] - tgt_cls[:, 0]- tgt_cls[:, 1]
    tgt_reward=torch.where(labels==0,reward_for_label_0,reward_for_label_1)
    tgt_reward=torch.where(labels==2,reward_for_label_2,tgt_reward)
     
    
    # if style == 0:  
    #     tgt_reward = tgt_cls[:, 1] - tgt_cls[:, 0]
    # else:
    #     tgt_reward = tgt_cls[:, 0] - tgt_cls[:, 1]

    loss_sc = cal_reward_loss(sample_probs, tgt_reward,device, idx)

    return loss_sc,tgt_reward


def collate_fn(insts, pad_token_id=1):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq

def cal_reward_loss(sample_probs, reward, device,idxs=None):
    sample_probs = sample_probs.contiguous()
    sample_logprobs = torch.log(sample_probs)
    reward = reward.unsqueeze(1).contiguous()
    if idxs is not None:
        batch_size, max_len = sample_probs.size()
        mask = torch.zeros(batch_size, max_len).to(device)
        for i, l in enumerate(idxs):
            mask[i, :l] = 1
        mask = mask.float().contiguous()
        output = -sample_logprobs * reward * mask
        output = (output.sum(-1)/mask.sum(-1)).mean()
    else:
        output = -sample_logprobs * reward
        output = output.mean()

    return output



def sample_3d(probs,device, temperature=1):
    '''probs.shape = (batch, seq_len, dim)'''
    sample_idx = torch.zeros(probs.size(0), probs.size(1)).to(device)
    sample_probs = torch.zeros(probs.size(0), probs.size(1)).to(device)
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
    else:
        temp = probs
    for i, s in enumerate(temp):
        temp_idx = torch.multinomial(s, 1)  # shape = (seq_len, 1)
        temp_probs = s.gather(1, temp_idx)  # shape = (seq_len, 1)
        sample_idx[i] = temp_idx.squeeze(1)
        sample_probs[i] = temp_probs.squeeze(1)

    return sample_probs, sample_idx.long()