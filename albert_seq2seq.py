import collections
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
        self.encoder: AlbertModel = encoder
        self.decoder: AlbertDecoderModel = AlbertDecoderModel(decoder) if isinstance(decoder, AlbertModel) else decoder
        self.output_layer: AlbertMLMHead = output_layer
        self.checkpoint = checkpoint
        self.checkpoint_num = 0

    def encode(self, device, src_inputs, src_mask):
        src_inputs = src_inputs.to(device)
        src_mask = src_mask.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask)
        return encoder_states

    def forward(self, device, src_inputs, tgt_inputs, src_mask, tgt_mask, log_softmax: bool = False):
        "Take in and process masked src and target sequences."
        encoder_states = self.encode(device, src_inputs, src_mask)[0]

        tgt_inputs = tgt_inputs.to(device)

        subseq_mask = future_mask(tgt_mask[:, :-1]).to(device)
        decoder_output = self.decoder(encoder_states, tgt_inputs[:, :-1], src_mask, subseq_mask)
        diag_outputs = torch.stack([decoder_output[:, d, d, :] for d in range(decoder_output.size(2))], 1)
        diag_outputs_flat = diag_outputs.view(-1, diag_outputs.size(-1))
        tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def save_checkpoint(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict." + str(self.checkpoint_num)))
        self.checkpoint_num = (self.checkpoint_num + 1) % self.checkpoint

    def save_state_dict(self, out_dir: str, state_dict):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(state_dict, os.path.join(out_dir, "mt_model.state_dict." + str(self.checkpoint_num)))
        self.checkpoint_num = (self.checkpoint_num + 1) % self.checkpoint

    def save_config_and_tok(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump((self.config, self.checkpoint), fp)
        self.text_processor.tokenizer.save(directory=out_dir)

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump((self.config, self.checkpoint), fp)

        torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        self.text_processor.tokenizer.save(directory=out_dir)

    @staticmethod
    def load(out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            mt_model = MassSeq2Seq(config=config, encoder=lm.encoder, decoder=lm.encoder, output_layer=lm.masked_lm,
                                   text_processor=lm.text_processor, checkpoint=checkpoint)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm

    def load_avg_model(self, out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            mt_model = self.__class__(config=config, encoder=lm.encoder, decoder=lm.encoder, output_layer=lm.masked_lm,
                                      text_processor=lm.text_processor, checkpoint=checkpoint)

            params_dict = collections.OrderedDict()
            num_models = 0
            for i in range(self.checkpoint):
                path = os.path.join(out_dir, "mt_model.state_dict." + str(i))
                if os.path.exists(path):
                    num_models += 1
                    state_dict = torch.load(path)

                    for k in state_dict.keys():
                        p = state_dict[k]
                        if isinstance(p, torch.HalfTensor):
                            p = p.float()
                        if k not in params_dict:
                            params_dict[k] = p.clone()
                        else:
                            params_dict[k] += p

            averaged_params = collections.OrderedDict()
            for k, v in params_dict.items():
                averaged_params[k] = v
                averaged_params[k].div_(num_models)
            mt_model.load_state_dict(averaged_params)
            return mt_model, lm


class MassSeq2Seq(AlbertSeq2Seq):
    def forward(self, device, src_inputs, tgt_inputs, src_pads, mask_pad_mask, log_softmax: bool = False):
        """
        :param mask_pad_mask: # Since MASS also generates MASK tokens, we do not backpropagate them during training.
        :return:
        """
        "Take in and process masked src and target sequences."
        encoder_states = self.encode(device, src_inputs, src_pads)[0]

        tgt_inputs = tgt_inputs.to(device)

        subseq_mask = future_mask(src_pads[:, :-1]).to(device)
        decoder_output = self.decoder(encoder_states, tgt_inputs[:, :-1], src_pads, subseq_mask)
        diag_outputs = torch.stack([decoder_output[:, d, d, :] for d in range(decoder_output.size(2))], 1)
        diag_outputs_flat = diag_outputs.view(-1, diag_outputs.size(-1))

        tgt_mask_flat = mask_pad_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_mask_flat]

        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs


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

    def forward(self, encoder_states, decoder_inputs, src_attention_mask=None, tgt_attention_mask=None):
        output_attention = self.attention(self.query(decoder_inputs), self.key(decoder_inputs),
                                          self.value(decoder_inputs),
                                          attention_mask=tgt_attention_mask)
        cross_attention = self.attention(self.src_attn_query(output_attention[0]), self.src_attn_key(encoder_states),
                                         self.src_attn_value(encoder_states),
                                         attention_mask=src_attention_mask)
        return cross_attention

    def attention(self, q, k, v, attention_mask=None):
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if query_layer.dim() == 5 and key_layer.dim() == 4:
            attention_scores = torch.matmul(query_layer, key_layer.unsqueeze(2).transpose(-1, -2))
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            if attention_mask.dim() > 4 and attention_scores.dim() == 4:
                # attention_scores = attention_scores.unsqueeze(2).expand(-1, -1, attention_scores.size(2), -1, -1)
                attention_scores = attention_scores.unsqueeze(2) + attention_mask
            elif attention_mask.dim() == 4 and attention_scores.dim() == 5:
                attention_scores = attention_scores + attention_mask.unsqueeze(2)
            else:
                attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        if attention_probs.dim() > 4 and value_layer.dim() == 4:
            context_layer = torch.matmul(attention_probs, value_layer.unsqueeze(2))
        else:
            context_layer = torch.matmul(attention_probs, value_layer)

        if context_layer.dim() == 4:
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        else:
            context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
                .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
                .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        if context_layer.dim() == 4:
            projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        else:
            projected_context_layer = torch.einsum("bfxnd,ndh->bfxh", context_layer, w) + b

        projected_context_layer_dropout = self.dropout(projected_context_layer)

        query = q
        if projected_context_layer_dropout.dim() == 4 and query.dim() < 4:
            query = query.unsqueeze(1).expand(-1, query.size(1), -1, -1)
        layernormed_context_layer = self.LayerNorm(query + projected_context_layer_dropout)
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
            src_attention_mask=None,
            tgt_attention_mask=None,
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

        if src_attention_mask is None:
            extended_src_attention_mask = None
        else:
            extended_src_attention_mask = src_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_src_attention_mask = extended_src_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_src_attention_mask = (1.0 - extended_src_attention_mask) * -10000.0

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_tgt_attention_mask = tgt_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_tgt_attention_mask = extended_tgt_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_tgt_attention_mask = (1.0 - extended_tgt_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        outputs = self.decoder(encoder_states, embedding_output, extended_src_attention_mask,
                               extended_tgt_attention_mask)

        return outputs
