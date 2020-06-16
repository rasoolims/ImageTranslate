import copy
import pickle

import torch.nn.functional as F
from transformers.modeling_albert import *

from lm import LM
from textprocessor import TextProcessor


def future_mask(tgt_mask):
    attn_shape = (tgt_mask.size(0), tgt_mask.size(1), tgt_mask.size(1))
    future_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type_as(tgt_mask)
    return ~future_mask & tgt_mask.unsqueeze(-1)


class AlbertSeq2Seq(nn.Module):
    def __init__(self, config: AlbertConfig, encoder, decoder, output_layer: AlbertMLMHead,
                 text_processor: TextProcessor, checkpoint: int = 5):
        super(AlbertSeq2Seq, self).__init__()
        self.text_processor: TextProcessor = text_processor
        self.config: AlbertConfig = config
        self.encoder = encoder
        self.encoder.__class__ = AlbertEncoderModel
        self.decoder: AlbertDecoderModel = AlbertDecoderModel(decoder) if isinstance(decoder, AlbertModel) else decoder
        self.output_layer: AlbertMLMHead = output_layer
        self.checkpoint = checkpoint
        self.checkpoint_num = 0
        self.decoder._tie_or_clone_weights(self.output_layer.decoder, self.decoder.embeddings.word_embeddings)

    def encode(self, device, src_inputs, src_mask, src_langs):
        src_inputs = src_inputs.to(device)
        src_mask = src_mask.to(device)
        src_langs = src_langs.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask, token_type_ids=src_langs)
        return encoder_states

    def forward(self, device, src_inputs, tgt_inputs, src_mask, tgt_mask, src_langs, tgt_langs,
                log_softmax: bool = False):
        "Take in and process masked src and target sequences."
        src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        encoder_states = self.encode(device, src_inputs, src_mask, src_langs)[0]

        tgt_langs = tgt_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        if tgt_inputs.device != encoder_states.device:
            tgt_inputs = tgt_inputs.to(device)

        subseq_mask = future_mask(tgt_mask[:, :-1])
        if subseq_mask.device != tgt_inputs.device:
            subseq_mask = subseq_mask.to(device)
        decoder_output = self.decoder(encoder_states, tgt_inputs[:, :-1], tgt_mask[:, :-1], src_mask, subseq_mask,
                                      token_type_ids=tgt_langs[:, :-1])
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump((self.config, self.checkpoint), fp)

        torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))

    @staticmethod
    def load(out_dir: str, tok_dir: str, sep_decoder: bool):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            decoder = copy.deepcopy(lm.encoder) if sep_decoder else lm.encoder
            mt_model = AlbertSeq2Seq(config=config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                     text_processor=lm.text_processor, checkpoint=checkpoint)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm


class MassSeq2Seq(AlbertSeq2Seq):
    def forward(self, device, src_inputs, src_pads, tgt_inputs, src_langs, tgt_langs=None, pad_idx: int = 1,
                tgt_positions=None,
                log_softmax: bool = False):
        """
        :param mask_pad_mask: # Since MASS also generates MASK tokens, we do not backpropagate them during training.
        :return:
        """
        tgt_inputs = tgt_inputs.to(device)
        tgt_mask = tgt_inputs != pad_idx

        if tgt_langs is not None:
            # Use back-translation loss
            return super().forward(device=device, src_inputs=src_inputs, src_mask=src_pads, tgt_inputs=tgt_inputs,
                                   tgt_mask=tgt_mask, src_langs=src_langs, tgt_langs=tgt_langs, log_softmax=log_softmax)

        "Take in and process masked src and target sequences."
        src_langs_t = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        encoder_states = self.encode(device, src_inputs, src_pads, src_langs_t)[0]

        tgt_langs = src_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)

        subseq_mask = future_mask(tgt_mask[:, :-1])
        decoder_output = self.decoder(encoder_states=encoder_states, input_ids=tgt_inputs[:, :-1],
                                      input_ids_mask=tgt_mask[:, :-1], src_attn_mask=src_pads,
                                      tgt_attn_mask=subseq_mask,
                                      position_ids=tgt_positions[:, :-1] if tgt_positions is not None else None,
                                      token_type_ids=tgt_langs[:, :-1])
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))

        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    @staticmethod
    def load(out_dir: str, tok_dir: str, sep_decoder: bool):
        mt_model, lm = AlbertSeq2Seq.load(out_dir, tok_dir, sep_decoder)
        mt_model.__class__ = MassSeq2Seq
        return mt_model, lm


class AlbertDecoderAttention(nn.Module):
    def __init__(self, albert_attention: AlbertAttention):
        super().__init__()
        self.output_attentions = albert_attention.output_attentions  # config.output_attentions
        self.dropout = albert_attention.dropout
        self.num_attention_heads = albert_attention.num_attention_heads
        self.hidden_size = albert_attention.hidden_size
        self.attention_head_size = albert_attention.attention_head_size
        self.all_head_size = albert_attention.all_head_size

        self.query = copy.deepcopy(albert_attention.query)
        self.key = copy.deepcopy(albert_attention.key)
        self.value = copy.deepcopy(albert_attention.value)

        self.src_attn_query = copy.deepcopy(albert_attention.query)
        self.src_attn_key = copy.deepcopy(albert_attention.key)
        self.src_attn_value = copy.deepcopy(albert_attention.value)

        self.dense = copy.deepcopy(albert_attention.dense)
        self.LayerNorm = copy.deepcopy(
            albert_attention.LayerNorm)  # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        if x.dim() == 4:
            return x.permute(0, 2, 1, 3)
        else:
            return x.permute(0, 3, 1, 2, 4)

    def forward(self, encoder_states, decoder_inputs, src_attn_mask=None, tgt_attn_mask=None):
        output_attention = self.attention(self.query(decoder_inputs), self.key(decoder_inputs),
                                          self.value(decoder_inputs), attn_mask=tgt_attn_mask)
        cross_attention = self.attention(self.src_attn_query(output_attention[0]), self.src_attn_key(encoder_states),
                                         self.src_attn_value(encoder_states), attn_mask=src_attn_mask)
        return cross_attention

    def attention(self, q, k, v, attn_mask=None):
        q_layer = self.transpose_for_scores(q)
        k_layer = self.transpose_for_scores(k)
        v_layer = self.transpose_for_scores(v)

        bs, qlen, dim = q.size()
        klen = k.size(1)
        attn_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        mask_reshape = (bs, 1, qlen, klen) if attn_mask.dim() == 3 else (bs, 1, 1, klen)
        attn_mask = (attn_mask == 0).view(mask_reshape).expand_as(attn_scores)  # (bs, n_heads, qlen, klen)
        attn_scores.masked_fill_(attn_mask, -1000000)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attn_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
                .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
                .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)

        layernormed_context_layer = self.LayerNorm(q + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)


class AlbertDecoderLayer(nn.Module):
    def __init__(self, albert_layer: AlbertLayer):
        super().__init__()

        self.full_layer_layer_norm = albert_layer.full_layer_layer_norm  # nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps) #todo clone
        self.attention = AlbertDecoderAttention(albert_layer.attention)
        self.ffn = albert_layer.ffn  # nn.Linear(self.config.hidden_size, self.config.intermediate_size) #todo clone
        self.ffn_output = albert_layer.ffn_output  # nn.Linear(self.config.intermediate_size, self.config.hidden_size) #todo clone
        self.activation = albert_layer.activation  # ACT2FN[self.config.hidden_act]

    def forward(self, encoder_states, hidden_states, src_attention_mask=None, tgt_attention_mask=None):
        attention_output = self.attention(encoder_states, hidden_states, src_attention_mask, tgt_attention_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them


class AlbertDecoderLayerGroup(nn.Module):
    def __init__(self, layer_groups: AlbertLayerGroup):
        super().__init__()
        self.albert_layers = nn.ModuleList([AlbertDecoderLayer(layer) for layer in layer_groups.albert_layers])

    def forward(self, encoder_states, hidden_states, src_attention_mask=None, tgt_attention_mask=None):
        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(encoder_states, hidden_states, src_attention_mask, tgt_attention_mask)
            hidden_states = layer_output[0]

        outputs = (hidden_states,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertDecoderTransformer(nn.Module):
    def __init__(self, albert_transformer: AlbertTransformer):
        super().__init__()

        self.config = albert_transformer.config
        self.embedding_hidden_mapping_in = albert_transformer.embedding_hidden_mapping_in
        self.albert_layer_groups = nn.ModuleList(
            [AlbertDecoderLayerGroup(albert_transformer.albert_layer_groups[i]) for i in
             range(self.config.num_hidden_groups)])

    def forward(self, encoder_states, hidden_states, src_attention_mask=None, tgt_attention_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        for i in range(self.config.num_hidden_layers):
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                encoder_states,
                hidden_states,
                src_attention_mask,
                tgt_attention_mask,
            )
            hidden_states = layer_group_output[0]

        return hidden_states


class AlbertDecoderModel(AlbertPreTrainedModel):
    config_class = AlbertConfig
    base_model_prefix = "albert"

    def __init__(self, encoder_layer: AlbertModel):
        super().__init__(encoder_layer.config)
        self.encoder_layer = encoder_layer
        self.config = encoder_layer.config
        self.embeddings = encoder_layer.embeddings
        self.decoder = AlbertDecoderTransformer(encoder_layer.encoder)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        self.encoder_layer._resize_token_embeddings(new_num_tokens)

    def forward(
            self,
            encoder_states,
            input_ids=None,
            input_ids_mask=None,
            src_attn_mask=None,
            tgt_attn_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if input_ids_mask.device != device:
            input_ids_mask = input_ids_mask.to(device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        embedding_output *= input_ids_mask.unsqueeze(-1)
        outputs = self.decoder(encoder_states, embedding_output, src_attn_mask, tgt_attn_mask)
        return outputs


class AlbertEncoderModel(AlbertModel):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        embedding_output *= attention_mask.unsqueeze(-1)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output) + encoder_outputs[
                                                     1:
                                                     ]  # add hidden_states and attentions if they are here
        return outputs
