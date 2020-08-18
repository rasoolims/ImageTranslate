import os
import pickle

import torch
import torch.nn as nn

import lm_config
from bert_seq2seq import BertEncoderModel, BertConfig
from lm import LM
from textprocessor import TextProcessor


class SenSim(nn.Module):
    def __init__(self, text_processor: TextProcessor, enc_layer: int = 6, embed_dim: int = 768,
                 intermediate_dim: int = 3072):
        super(SenSim, self).__init__()
        self.text_processor: TextProcessor = text_processor
        self.config = lm_config.get_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                           pad_token_id=text_processor.pad_token_id(),
                                           bos_token_id=text_processor.bos_token_id(),
                                           eos_token_id=text_processor.sep_token_id(),
                                           enc_layer=enc_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim)

        self.enc_layer = enc_layer
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.config["type_vocab_size"] = len(text_processor.languages)
        self.config = BertConfig(**self.config)
        self.encoder = BertEncoderModel(self.config)
        self.encoder.init_weights()
        self.input_attention = nn.Linear(self.config.hidden_size, 1)

    def init_from_lm(self, lm: LM):
        self.encoder = lm.encoder

    def encode(self, src_inputs, src_mask, src_langs):
        device = self.encoder.embeddings.word_embeddings.weight.device
        if src_inputs.device != device:
            src_inputs = src_inputs.to(device)
            src_mask = src_mask.to(device)
            src_langs = src_langs.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask, token_type_ids=src_langs)
        attention_scores = self.input_attention(encoder_states).squeeze(-1)
        attention_scores.masked_fill_(~src_mask, -10000.0)
        attention_prob = nn.Softmax(dim=1)(attention_scores)
        sentence_embeddings = torch.einsum("bfd,bf->bd", encoder_states, attention_prob)
        return sentence_embeddings

    def forward(self, src_inputs, src_mask, src_langs, tgt_inputs, tgt_mask, tgt_langs, src_neg_inputs=None,
                src_neg_mask=None, src_neg_langs=None, tgt_neg_inputs=None, tgt_neg_mask=None, tgt_neg_langs=None,
                normalize: bool = False):
        "Take in and process masked src and target sequences."
        device = self.encoder.embeddings.word_embeddings.weight.device
        src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        src_inputs = src_inputs.to(device)
        src_langs = src_langs.to(device)

        if src_mask.device != device:
            src_mask = src_mask.to(device)
        src_embed = self.encode(src_inputs, src_mask, src_langs)

        tgt_langs = tgt_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        if tgt_inputs.device != device:
            tgt_inputs = tgt_inputs.to(device)
            tgt_mask = tgt_mask.to(device)
        tgt_embed = self.encode(tgt_inputs, tgt_mask, tgt_langs)

        src_norm = torch.norm(src_embed, dim=-1, p=2).unsqueeze(-1) + 1e-4
        src_embed = torch.div(src_embed, src_norm)
        tgt_norm = torch.norm(tgt_embed, dim=-1, p=2).unsqueeze(-1) + 1e-4
        tgt_embed = torch.div(tgt_embed, tgt_norm)
        if normalize:
            if src_neg_langs is not None:
                src_neg_langs = src_neg_langs.unsqueeze(-1).expand(-1, src_neg_inputs.size(-1))
                src_neg_inputs = src_neg_inputs.to(device)
                src_neg_langs = src_neg_langs.to(device)

                if src_neg_mask.device != device:
                    src_neg_mask = src_neg_mask.to(device)
                src_neg_embed = self.encode(src_neg_inputs, src_neg_mask, src_neg_langs)
                src_neg_norm = torch.norm(src_neg_embed, dim=-1, p=2).unsqueeze(-1) + 1e-4
                src_neg_embed = torch.div(src_neg_embed, src_neg_norm)

                tgt_neg_langs = tgt_neg_langs.unsqueeze(-1).expand(-1, tgt_neg_inputs.size(-1))
                tgt_neg_inputs = tgt_neg_inputs.to(device)
                tgt_neg_langs = tgt_neg_langs.to(device)

                if tgt_neg_mask.device != device:
                    tgt_neg_mask = tgt_neg_mask.to(device)
                tgt_neg_embed = self.encode(tgt_neg_inputs, tgt_neg_mask, tgt_neg_langs)
                tgt_neg_norm = torch.norm(tgt_neg_embed, dim=-1, p=2).unsqueeze(-1) + 1e-4
                tgt_neg_embed = torch.div(tgt_neg_embed, tgt_neg_norm)

                tgt_neg_embd = torch.cat([tgt_neg_embed, tgt_embed])
                src_neg_embd = torch.cat([src_neg_embed, src_embed])

                nominator = torch.sum(src_embed * tgt_embed, dim=-1) + 1e-4

                cross_dot = torch.mm(src_embed, tgt_neg_embd.T)
                cross_dot_rev = torch.mm(tgt_embed, src_neg_embd.T)
                cross_dot_all = torch.cat([cross_dot, cross_dot_rev], dim=1)
                denom = torch.log(torch.sum(torch.exp(cross_dot_all), dim=-1) + 1e-4)
                log_neg = torch.sum(denom - nominator) / len(cross_dot)
            else:
                cross_dot = torch.mm(src_embed, tgt_embed.T)
                denom = torch.log(torch.sum(torch.exp(cross_dot), dim=-1) + 1e-4)
                nominator = torch.diagonal(cross_dot[:, :], 0) + 1e-4
                log_neg = torch.sum(denom - nominator) / len(cross_dot)

            return log_neg
        else:
            dot_prod = torch.sum(src_embed * tgt_embed, dim=-1)
            return dot_prod

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump(
                (self.enc_layer, self.embed_dim, self.intermediate_dim), fp)
        try:
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        except:
            torch.cuda.empty_cache()
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))

    @staticmethod
    def load(out_dir: str, tok_dir: str):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            enc_layer, embed_dim, intermediate_dim = pickle.load(
                fp)

            model = SenSim(text_processor=text_processor, enc_layer=enc_layer, embed_dim=embed_dim,
                           intermediate_dim=intermediate_dim)
            model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict"), map_location=device),
                                  strict=False)
            return model, text_processor
