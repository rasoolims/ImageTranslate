import copy

from transformers.modeling_albert import *

from seq2seq import Seq2Seq
from textprocessor import TextProcessor


class AlbertSeq2Seq(Seq2Seq):
    def __init__(self, text_processor: TextProcessor, sep_decoder: bool = True, lang_dec: bool = True,
                 size: int = 6, use_proposals=False):
        super(AlbertSeq2Seq, self).__init__(decoder_cls=AlbertDecoderModel, encoder_cls=AlbertEncoderModel,
                                            output_cls=AlbertMLMHead, config_cls=AlbertConfig,
                                            text_processor=text_processor, sep_decoder=sep_decoder, lang_dec=lang_dec,
                                            size=size, use_proposals=use_proposals)

    @staticmethod
    def load(out_dir: str, tok_dir: str):
        mt_model = Seq2Seq.load(out_dir, tok_dir)
        mt_model.__class__ = AlbertSeq2Seq
        return mt_model


class AlbertDecoderAttention(nn.Module):
    def __init__(self, albert_attention: AlbertAttention):
        super().__init__()
        self.output_attentions = albert_attention.output_attentions  # config.output_attentions
        self.dropout = albert_attention.dropout
        self.num_attention_heads = albert_attention.num_attention_heads
        self.hidden_size = albert_attention.hidden_size
        self.attention_head_size = albert_attention.attention_head_size
        self.all_head_size = albert_attention.all_head_size

        self.query = albert_attention.query
        self.key = albert_attention.key
        self.value = albert_attention.value

        self.src_attn_query = copy.deepcopy(albert_attention.query)
        self.src_attn_key = copy.deepcopy(albert_attention.key)
        self.src_attn_value = copy.deepcopy(albert_attention.value)

        self.dense = albert_attention.dense
        self.LayerNorm = albert_attention.LayerNorm
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
        attn_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))

        if attn_mask is not None:
            bs, qlen, dim = q.size()
            klen = k.size(1)
            mask_reshape = (bs, 1, qlen, klen) if attn_mask.dim() == 3 else (bs, 1, 1, klen)
            attn_mask = (attn_mask == 0).view(mask_reshape).expand_as(attn_scores)
            attn_scores.masked_fill_(attn_mask, -10000.0)

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

        self.full_layer_layer_norm = albert_layer.full_layer_layer_norm
        self.attention = AlbertDecoderAttention(albert_layer.attention)
        self.ffn = albert_layer.ffn
        self.ffn_output = albert_layer.ffn_output
        self.activation = albert_layer.activation

    def forward(self, encoder_states, hidden_states, src_attn_mask=None, tgt_attn_mask=None):
        attention_output = self.attention(encoder_states, hidden_states, src_attn_mask, tgt_attn_mask)
        ffn_output = self.ffn(attention_output[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]


class AlbertDecoderLayerGroup(nn.Module):
    def __init__(self, layer_groups: AlbertLayerGroup):
        super().__init__()
        self.albert_layers = nn.ModuleList([AlbertDecoderLayer(layer) for layer in layer_groups.albert_layers])

    def forward(self, encoder_states, hidden_states, src_attn_mask=None, tgt_attn_mask=None):
        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(encoder_states, hidden_states, src_attn_mask, tgt_attn_mask)
            hidden_states = layer_output[0]

        outputs = (hidden_states,)
        return outputs


class AlbertDecoderTransformer(nn.Module):
    def __init__(self, albert_transformer: AlbertTransformer):
        super().__init__()

        self.config = albert_transformer.config
        self.embedding_hidden_mapping_in = albert_transformer.embedding_hidden_mapping_in
        self.albert_layer_groups = nn.ModuleList(
            [AlbertDecoderLayerGroup(albert_transformer.albert_layer_groups[i]) for i in
             range(self.config.num_hidden_groups)])

    def forward(self, encoder_states, hidden_states, src_attn_mask=None, tgt_attn_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        for i in range(self.config.num_hidden_layers):
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                encoder_states,
                hidden_states,
                src_attn_mask,
                tgt_attn_mask,
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


class AlbertEncoderModel(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.init_weights()

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
        outputs = (sequence_output,) + encoder_outputs[1:]
        return outputs
