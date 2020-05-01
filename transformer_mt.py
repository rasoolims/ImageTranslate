import copy
import pickle

import torch.nn.functional as F
from transformers.modeling_albert import *

from lm import LM
import torch.nn as nn
from textprocessor import TextProcessor

class TransformerMT(nn.Module):
    def __init__(self, lm: LM, sep_encoder_decoder: bool = True):
        super(TransformerMT, self).__init__()
        self.text_processor: TextProcessor = lm.text_processor
        self.transformer = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1, activation='relu')
        self.embedding = nn.Embedding(num_embeddings=lm.text_processor.vocab_size(), embedding_dim=128)
        self.output_layer = nn.Linear(in_features=128, out_features=lm.text_processor.vocab_size())
        #self.output_layer.weight = self.embedding.weight # tie the weights!

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, device, src_inputs, tgt_inputs, src_mask, tgt_mask, log_softmax: bool = False,
                flatten: bool = False):
        "Take in and process masked src and target sequences."

        src_inputs = src_inputs.T.to(device)
        tgt_inputs = tgt_inputs[:,:-1].T.to(device) # Hide the last word since we don't need it!
        src_pad_mask = (~src_mask).to(device)
        tgt_pad_mask = (~tgt_mask[:,:-1]).to(device) # Hide the last word since we don't need it!
        subseq_mask = self._generate_square_subsequent_mask(tgt_inputs.size(0)).to(device)

        src_embed = self.embedding(src_inputs)
        tgt_embed = self.embedding(tgt_inputs)

        outputs =  self.transformer(src=src_embed, tgt=tgt_embed, tgt_mask=subseq_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask,)
        outputs = outputs.transpose(0, 1)
        outputs = self.output_layer(outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        if flatten:
            outputs = outputs.view(-1, outputs.size(-1))

        return outputs

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        self.text_processor.tokenizer.save(directory=out_dir)

    @staticmethod
    def load(out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            lm.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return lm