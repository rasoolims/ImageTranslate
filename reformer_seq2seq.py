import collections
import pickle

import torch.nn.functional as F
from transformers.modeling_albert import *
from transformers.modeling_reformer import *
from transformers.modeling_reformer import _get_least_common_mult_chunk_len, _ReversibleFunction

from reformer_lm import ReformerLM
from textprocessor import TextProcessor


def future_mask(tgt_mask):
    attn_shape = (tgt_mask.size(0), tgt_mask.size(1), tgt_mask.size(1))
    future_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type_as(tgt_mask)
    return ~future_mask & tgt_mask.unsqueeze(-1)


class ReformerSeq2Seq(nn.Module):
    def __init__(self, config: ReformerConfig, encoder, decoder, output_layer: ReformerOnlyLMHead,
                 text_processor: TextProcessor, checkpoint: int = 5):
        super(ReformerSeq2Seq, self).__init__()
        self.text_processor: TextProcessor = text_processor
        self.config: ReformerConfig = config
        self.encoder: ReformerLM = encoder
        self.decoder: ReformerDecoderModel = decoder if isinstance(decoder,
                                                                   ReformerDecoderModel) else ReformerDecoderModel(
            decoder)
        self.output_layer: ReformerOnlyLMHead = output_layer
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
            lm = ReformerLM(text_processor=text_processor, config=config)
            mt_model = ReformerSeq2Seq(config=config, encoder=lm.encoder, decoder=lm.encoder, output_layer=lm.masked_lm,
                                       text_processor=lm.text_processor, checkpoint=checkpoint)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm

    def load_avg_model(self, out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = ReformerLM(text_processor=text_processor, config=config)
            mt_model = ReformerSeq2Seq(config=config, encoder=lm.encoder, decoder=lm.encoder, output_layer=lm.masked_lm,
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


class LocalCrossAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, pretrained_attention: LocalSelfAttention):
        super().__init__()

        self.num_attention_heads = pretrained_attention.num_attention_heads
        self.chunk_length = pretrained_attention.chunk_length
        self.num_chunks_before = pretrained_attention.num_chunks_before
        self.num_chunks_after = pretrained_attention.num_chunks_after
        self.is_decoder = pretrained_attention.is_decoder
        self.pad_token_id = pretrained_attention.pad_token_id

        self.attention_head_size = pretrained_attention.attention_head_size
        self.all_head_size = pretrained_attention.all_head_size
        self.hidden_size = pretrained_attention.hidden_size

        # projection matrices
        self.query = pretrained_attention.query
        self.key = pretrained_attention.key
        self.value = pretrained_attention.value

        self.dropout = pretrained_attention.dropout

        # save mask value here
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(self, decoder_states, hidden_states, attention_mask=None, head_mask=None, do_output_attentions=False,
                **kwargs):
        tgt_sequence_length = decoder_states.shape[-1]
        sequence_length = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]

        # project hidden_states to query, key and value
        query_vectors = self.query(decoder_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        # split last dim into `config.num_attention_heads` and `config.attention_head_size`
        query_vectors = self._split_hidden_size_dim(query_vectors, self.num_attention_heads, self.attention_head_size)
        key_vectors = self._split_hidden_size_dim(key_vectors, self.num_attention_heads, self.attention_head_size)
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert (
                query_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            query_vectors.shape[-1], self.attention_head_size
        )
        assert (
                key_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            key_vectors.shape[-1], self.attention_head_size
        )
        assert (
                value_vectors.shape[-1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            value_vectors.shape[-1], self.attention_head_size
        )

        if self.chunk_length is None:
            assert (
                    self.num_chunks_before == 0 and self.num_chunks_after == 0
            ), "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0."

        # normalize key vectors
        key_vectors = key_vectors / torch.sqrt(
            torch.tensor(self.attention_head_size, device=key_vectors.device, dtype=key_vectors.dtype)
        )

        # chunk vectors
        # B x Num_Attn_Head x Seq_Len // chunk_len x chunk_len  x  attn_head_size
        query_vectors = self._split_seq_length_dim_to(
            query_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )
        key_vectors = self._split_seq_length_dim_to(
            key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )
        value_vectors = self._split_seq_length_dim_to(
            value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )

        # chunk indices
        src_indices = torch.arange(tgt_sequence_length, device=query_vectors.device).repeat(
            batch_size, self.num_attention_heads, 1
        )
        indices = torch.arange(sequence_length, device=key_vectors.device).repeat(
            batch_size, self.num_attention_heads, 1
        )
        query_indices = self._split_seq_length_dim_to(src_indices, -1, self.chunk_length, self.num_attention_heads)
        key_indices = self._split_seq_length_dim_to(indices, -1, self.chunk_length, self.num_attention_heads)

        # append chunks before and after
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)
        key_indices = self._look_adjacent(key_indices, self.num_chunks_before, self.num_chunks_after)

        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        mask = self._compute_attn_mask(query_indices, key_indices, attention_mask, query_key_dots.shape)

        if mask is not None:
            # get mask tensor depending on half precision or not
            if query_key_dots.dtype == torch.float16:
                mask_value = self.mask_value_float16
            else:
                mask_value = self.mask_value_float32

            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # softmax
        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del logits

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        assert out_vectors.shape == (
        batch_size, self.num_attention_heads, tgt_sequence_length, self.attention_head_size,)

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if do_output_attentions is False:
            attention_probs = ()

        return LocalSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs)

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dots_shape):
        mask = None

        # chunk attention mask and look before and after
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.uint8)[:, None, :]
            attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
            attention_mask_key = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)

        # Causal mask
        if self.is_decoder is True:
            mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

        # Attention mask
        if attention_mask is not None:
            # create attn_mask
            attn_mask = (attention_mask.unsqueeze(-1) * attention_mask_key.unsqueeze(-2)).expand(query_key_dots_shape)
            # multiply by casaul mask if necessary
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask
        return mask


class LSHSelfCrossAttention(nn.Module, EfficientAttentionMixin):
    def __init__(self, lsh_pretrained: LSHSelfAttention):
        super().__init__()
        self.chunk_length = lsh_pretrained.chunk_length
        self.num_hashes = lsh_pretrained.num_hashes
        self.num_buckets = lsh_pretrained.num_buckets
        self.src_num_buckets = lsh_pretrained.num_buckets
        self.num_chunks_before = lsh_pretrained.num_chunks_before
        self.num_chunks_after = lsh_pretrained.num_chunks_after
        self.hash_seed = lsh_pretrained.hash_seed
        self.is_decoder = lsh_pretrained.is_decoder
        self.max_position_embeddings = lsh_pretrained.max_position_embeddings

        self.dropout = lsh_pretrained.dropout

        self.num_attention_heads = lsh_pretrained.num_attention_heads
        self.attention_head_size = lsh_pretrained.attention_head_size
        self.all_head_size = lsh_pretrained.all_head_size
        self.hidden_size = lsh_pretrained.hidden_size

        # projection matrices
        self.query_key = lsh_pretrained.query_key
        self.value = lsh_pretrained.value

        # save mask value here. Need fp32 and fp16 mask values
        self.register_buffer("self_mask_value_float16", torch.tensor(-1e3))
        self.register_buffer("self_mask_value_float32", torch.tensor(-1e5))
        self.register_buffer("mask_value_float16", torch.tensor(-1e4))
        self.register_buffer("mask_value_float32", torch.tensor(-1e9))

    def forward(
            self,
            decoder_states,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            num_hashes=None,
            do_output_attentions=False,
            **kwargs
    ):
        tgt_sequence_length = decoder_states.shape[-1]
        sequence_length = hidden_states.shape[-1]
        batch_size = hidden_states.shape[0]

        # num hashes can optionally be overwritten by user
        num_hashes = num_hashes if num_hashes is not None else self.num_hashes

        # project hidden_states to query_key and value
        query_vectors = self.query_key(decoder_states)
        key_vectors = self.query_key(hidden_states)
        value_vectors = self.value(hidden_states)

        # free memory
        del hidden_states

        query_vectors = self._split_hidden_size_dim(
            query_vectors, self.num_attention_heads, self.attention_head_size
        )
        key_vectors = self._split_hidden_size_dim(
            key_vectors, self.num_attention_heads, self.attention_head_size
        )
        value_vectors = self._split_hidden_size_dim(value_vectors, self.num_attention_heads, self.attention_head_size)

        assert (
                query_vectors.shape[-1] == self.attention_head_size and key_vectors.shape[
            -1] == self.attention_head_size
        ), "last dim of query_key_vectors is {} but should be {}.".format(
            query_vectors.shape[-1], key_vectors.shape[-1], self.attention_head_size
        )
        assert (
                value_vectors.shape[-1] == self.attention_head_size
        ), "last dim of value_vectors is {} but should be {}.".format(
            value_vectors.shape[-1], self.attention_head_size
        )

        # set `num_buckets` on the fly, recommended way to do it
        if self.num_buckets is None:
            self.num_buckets = self._set_num_buckets(sequence_length)
            self.src_num_buckets = self._set_num_buckets(tgt_sequence_length)

        # use cached buckets for backprop only
        # hash query key vectors into buckets
        query_buckets = self._hash_vectors(query_vectors, num_hashes, self.src_num_buckets)
        key_buckets = self._hash_vectors(key_vectors, num_hashes, self.num_buckets)

        assert (
                int(query_buckets.shape[-1]) == num_hashes * tgt_sequence_length
        ), "last dim of buckets is {}, but should be {}".format(query_buckets.shape[-1],
                                                                num_hashes * tgt_sequence_length)

        sorted_query_bucket_idx, undo_sorted_query_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
            tgt_sequence_length, query_buckets, num_hashes
        )
        sorted_key_bucket_idx, undo_sorted_key_bucket_idx = self._get_sorted_bucket_idx_and_undo_sorted_bucket_idx(
            sequence_length, key_buckets, num_hashes
        )

        # make sure bucket idx is not longer then sequence length
        sorted_query_bucket_idx = sorted_query_bucket_idx % tgt_sequence_length
        sorted_key_bucket_idx = sorted_key_bucket_idx % sequence_length

        # cluster query key value vectors according to hashed buckets
        query_vectors = self._gather_by_expansion(query_vectors, sorted_query_bucket_idx, num_hashes)
        key_vectors = self._gather_by_expansion(key_vectors, sorted_key_bucket_idx, num_hashes)
        value_vectors = self._gather_by_expansion(value_vectors, sorted_key_bucket_idx, num_hashes)

        query_vectors = self._split_seq_length_dim_to(
            query_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )
        key_vectors = self._split_seq_length_dim_to(
            key_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )
        value_vectors = self._split_seq_length_dim_to(
            value_vectors, -1, self.chunk_length, self.num_attention_heads, self.attention_head_size,
        )

        if self.chunk_length is None:
            assert (
                    self.num_chunks_before == 0 and self.num_chunks_after == 0
            ), "If `config.chunk_length` is `None`, make sure `config.num_chunks_after` and `config.num_chunks_before` are set to 0."

        # scale key vectors
        key_vectors = self._len_and_dim_norm(key_vectors)

        # get attention probs
        out_vectors, logits, attention_probs = self._attend(
            query_vectors=query_vectors,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            sorted_bucket_idx=sorted_query_bucket_idx,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # free memory
        del query_vectors, key_vectors, value_vectors

        # sort clusters back to correct ordering
        out_vectors, logits = ReverseSort.apply(
            out_vectors, logits, sorted_key_bucket_idx, undo_sorted_key_bucket_idx, self.num_hashes
        )

        # sum up all hash rounds
        if num_hashes > 1:
            out_vectors = self._split_seq_length_dim_to(
                out_vectors, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size,
            )
            logits = self._split_seq_length_dim_to(
                logits, num_hashes, sequence_length, self.num_attention_heads, self.attention_head_size,
            ).unsqueeze(-1)

            probs_vectors = torch.exp(logits - torch.logsumexp(logits, dim=2, keepdim=True))
            out_vectors = torch.sum(out_vectors * probs_vectors, dim=2)
            # free memory
            del probs_vectors

        # free memory
        del logits

        assert out_vectors.shape == (
            batch_size,
            self.num_attention_heads,
            sequence_length,
            self.attention_head_size,
        ), "out_vectors have be of shape `[batch_size, config.num_attention_heads, sequence_length, config.attention_head_size]`."

        out_vectors = self._merge_hidden_size_dims(out_vectors, self.num_attention_heads, self.attention_head_size)

        if do_output_attentions is False:
            attention_probs = ()

        return LSHSelfAttentionOutput(hidden_states=out_vectors, attention_probs=attention_probs,
                                      buckets=self.src_num_buckets)

    def _hash_vectors(self, vectors, num_hashes, num_buckets):
        batch_size = vectors.shape[0]

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        if isinstance(num_buckets, int):
            assert (
                    num_buckets % 2 == 0
            ), "There should be an even number of bucktes, but `self.num_bucktes`: {}".format(self.num_buckets)
            rotation_size = num_buckets
            num_buckets = num_buckets
        else:
            # Factorize the hash if self.num_buckets is a list or tuple
            rotation_size, num_buckets = 0, 1
            for bucket_factor in self.num_buckets:
                assert bucket_factor % 2 == 0, "The number of buckets should be even, but `num_bucket`: {}".format(
                    bucket_factor
                )
                rotation_size = rotation_size + bucket_factor
                num_buckets = num_buckets * bucket_factor

        # remove gradient
        vectors = vectors.detach()

        if self.hash_seed is not None:
            # for determinism
            torch.manual_seed(self.hash_seed)

        rotations_shape = (self.num_attention_heads, vectors.shape[-1], num_hashes, rotation_size // 2)
        # create a random self.attention_head_size x num_hashes x num_buckets/2
        random_rotations = torch.randn(rotations_shape, device=vectors.device, dtype=vectors.dtype)

        # Output dim: Batch_Size x Num_Attn_Heads x Num_Hashes x Seq_Len x Num_Buckets/2
        rotated_vectors = torch.einsum("bmtd,mdhr->bmhtr", vectors, random_rotations)

        if isinstance(self.num_buckets, int) or len(self.num_buckets) == 1:
            rotated_vectors = torch.cat([rotated_vectors, -rotated_vectors], dim=-1)
            buckets = torch.argmax(rotated_vectors, dim=-1)
        else:
            # Get the buckets for them and combine.
            buckets, cur_sum, cur_product = None, 0, 1
            for bucket_factor in self.num_buckets:
                rotated_vectors_factor = rotated_vectors[..., cur_sum: cur_sum + (bucket_factor // 2)]
                cur_sum = cur_sum + bucket_factor // 2
                rotated_vectors_factor = torch.cat([rotated_vectors_factor, -rotated_vectors_factor], dim=-1)

                if buckets is None:
                    buckets = torch.argmax(rotated_vectors_factor, dim=-1)
                else:
                    buckets = buckets + (cur_product * torch.argmax(rotated_vectors_factor, dim=-1))

                cur_product = cur_product * bucket_factor

        # buckets is now (Batch_size x Num_Attn_Heads x Num_Hashes x Seq_Len).
        # Next we add offsets so that bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(num_hashes, device=vectors.device)
        offsets = (offsets * num_buckets).view((1, 1, -1, 1))

        # expand to batch size and num attention heads
        offsets = offsets.expand((batch_size, self.num_attention_heads) + offsets.shape[-2:])
        offset_buckets = (buckets + offsets).flatten(start_dim=2, end_dim=3)

        return offset_buckets

    def _get_sorted_bucket_idx_and_undo_sorted_bucket_idx(self, sequence_length, buckets, num_hashes):
        # no gradients are needed
        with torch.no_grad():
            batch_size = buckets.shape[0]

            # arange and expand
            orig_indices = torch.arange(num_hashes * sequence_length, device=buckets.device).view(1, 1, -1)
            orig_indices = orig_indices.expand(batch_size, self.num_attention_heads, orig_indices.shape[-1])

            # scale buckets
            scaled_buckets = sequence_length * buckets + (orig_indices % sequence_length)

            # remove gradient
            scaled_buckets = scaled_buckets.detach()

            # Hash-based sort
            sorted_bucket_idx = torch.argsort(scaled_buckets, dim=-1)

            # create simple indices to scatter to, to have undo sort
            indices = (
                torch.arange(sorted_bucket_idx.shape[-1], device=buckets.device)
                    .view(1, 1, -1)
                    .expand(sorted_bucket_idx.shape)
            )

            # get undo sort
            undo_sorted_bucket_idx = sorted_bucket_idx.new(*sorted_bucket_idx.size())
            undo_sorted_bucket_idx.scatter_(-1, sorted_bucket_idx, indices)

        return sorted_bucket_idx, undo_sorted_bucket_idx

    def _num_buckets(self, sequence_length):
        # recommended `num_buckets` from paper
        num_buckets = 2 * sequence_length // self.chunk_length

        # factorize `num_buckets` if `num_buckets` becomes too large
        num_buckets_limit = max(int((self.max_position_embeddings // self.chunk_length) ** (0.5)), self.chunk_length, )
        if num_buckets > 2 * num_buckets_limit:
            num_buckets = [num_buckets_limit, num_buckets // num_buckets_limit + 1]

        logger.warning("config.num_buckets is not set. Setting config.num_buckets to {}...".format(num_buckets))
        return num_buckets

    def _attend(
            self, query_vectors, key_vectors, value_vectors, sorted_bucket_idx, attention_mask, head_mask,
    ):
        key_vectors = self._look_adjacent(key_vectors, self.num_chunks_before, self.num_chunks_after)
        value_vectors = self._look_adjacent(value_vectors, self.num_chunks_before, self.num_chunks_after)

        # get logits and dots
        query_key_dots = torch.matmul(query_vectors, key_vectors.transpose(-1, -2))

        # free memory
        del query_vectors, key_vectors

        query_bucket_idx = self._split_seq_length_dim_to(
            sorted_bucket_idx, -1, self.chunk_length, self.num_attention_heads
        )
        key_value_bucket_idx = self._look_adjacent(query_bucket_idx, self.num_chunks_before, self.num_chunks_after)

        # get correct mask values depending on precision
        if query_key_dots.dtype == torch.float16:
            self_mask_value = self.self_mask_value_float16
            mask_value = self.mask_value_float16
        else:
            self_mask_value = self.self_mask_value_float32
            mask_value = self.mask_value_float32

        mask = self._compute_attn_mask(query_bucket_idx, key_value_bucket_idx, attention_mask)

        if mask is not None:
            query_key_dots = torch.where(mask, query_key_dots, mask_value)

        # free memory
        del mask

        # Self mask is ALWAYS applied.
        # From the reformer paper (https://arxiv.org/pdf/2001.04451.pdf):
        # " While attention to the future is not allowed, typical implementations of the
        # Transformer do allow a position to attend to itself.
        # Such behavior is undesirable in a shared-QK formulation because the dot-product
        # of a query vector with itself will almost always be greater than the dot product of a
        # query vector with a vector at another position. We therefore modify the masking
        # to forbid a token from attending to itself, except in situations
        # where a token has no other valid attention targets (e.g. the first token in a sequence) "

        self_mask = torch.ne(query_bucket_idx.unsqueeze(-1), key_value_bucket_idx.unsqueeze(-2)).to(
            query_bucket_idx.device
        )

        # apply self_mask
        query_key_dots = torch.where(self_mask, query_key_dots, self_mask_value)

        # free memory
        del self_mask

        logits = torch.logsumexp(query_key_dots, dim=-1, keepdim=True)
        # dots shape is `[batch_size, num_attn_heads, num_hashes * seq_len // chunk_length, chunk_length, chunk_length * (1 + num_chunks_before + num_chunks_after)]`
        attention_probs = torch.exp(query_key_dots - logits)

        # free memory
        del query_key_dots

        # dropout
        attention_probs = nn.functional.dropout(attention_probs, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # attend values
        out_vectors = torch.matmul(attention_probs, value_vectors)

        # free memory
        del value_vectors

        # merge chunk length
        logits = logits.flatten(start_dim=2, end_dim=3).squeeze(-1)
        out_vectors = out_vectors.flatten(start_dim=2, end_dim=3)

        return out_vectors, logits, attention_probs

    def _compute_attn_mask(self, query_indices, key_indices, attention_mask):
        mask = None

        # Causal mask
        if self.is_decoder:
            mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)

        # Attention mask: chunk, look up correct mask value from key_value_bucket_idx
        # IMPORTANT: official trax code does not use a mask for LSH Atttention. Not sure why.
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.uint8)[:, None, None, :]
            # expand attn_mask to fit with key_value_bucket_idx shape
            attention_mask = attention_mask.expand(query_indices.shape[:-1] + (-1,))
            key_attn_mask = torch.gather(attention_mask, -1, key_indices)
            query_attn_mask = torch.gather(attention_mask, -1, query_indices)
            # expand to query_key_dots shape: duplicate along query axis since key sorting is the same for each query position in chunk
            attn_mask = query_attn_mask.unsqueeze(-1) * key_attn_mask.unsqueeze(-2)
            # free memory
            del query_attn_mask, key_attn_mask, attention_mask

            # multiply by casaul mask if necessary
            if mask is not None:
                mask = mask * attn_mask
            else:
                mask = attn_mask

        return mask

    def _len_and_dim_norm(self, vectors):
        """
            length and attention head size dim normalization
        """
        vectors = self._len_norm(vectors)
        vectors = vectors * torch.rsqrt(
            torch.tensor(self.attention_head_size, device=vectors.device, dtype=vectors.dtype)
        )
        return vectors

    def _len_norm(self, x, epsilon=1e-6):
        """
            length normalization
        """
        variance = torch.mean(x ** 2, -1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + epsilon)
        return norm_x

    def _gather_by_expansion(self, vectors, idxs, num_hashes):
        """
            expand dims of idxs and vectors for all hashes and gather
        """
        expanded_idxs = idxs.unsqueeze(-1).expand(-1, -1, -1, self.attention_head_size)
        vectors = vectors.repeat(1, 1, num_hashes, 1)
        return torch.gather(vectors, 2, expanded_idxs)


class ReformerDecoderAttention(nn.Module):
    def __init__(self, pretrainedAttention: ReformerAttention):
        super().__init__()
        self.layer_id = pretrainedAttention.layer_id
        self.attn_layers = pretrainedAttention.attn_layers

        self.layer_norm = pretrainedAttention.layer_norm
        if isinstance(pretrainedAttention.self_attention, LSHSelfAttention):
            self.self_attention = LSHSelfCrossAttention(pretrainedAttention.self_attention)
        else:
            self.self_attention = LocalCrossAttention(pretrainedAttention.self_attention)
        self.output = pretrainedAttention.output

    def forward(
            self,
            encoder_states,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            num_hashes=None,
            do_output_attentions=False,
            buckets=None,
    ):
        hidden_states = self.layer_norm(hidden_states)

        # use cached buckets for backprob if buckets not None for LSHSelfAttention
        self_attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            do_output_attentions=do_output_attentions,
            buckets=buckets,
        )
        attention_output = self.output(self_attention_outputs.hidden_states)

        # add buckets if necessary
        if hasattr(self_attention_outputs, "buckets"):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None

        return AttentionOutput(
            hidden_states=attention_output, attention_probs=self_attention_outputs.attention_probs, buckets=buckets,
        )


class ReformerDecoderLayer(nn.Module):
    def __init__(self, pretrainedLayer: ReformerLayer):
        super().__init__()
        self.attention: ReformerDecoderAttention = ReformerDecoderAttention(pretrainedLayer.attention)
        # dropout requires to have the same
        # seed for forward and backward pass
        self.attention_seed = None
        self.feed_forward_seed = None

        self.feed_forward = pretrainedLayer.feed_forward

    def _init_attention_seed(self):
        """
            This function sets a new seed for the
            attention layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        # randomize seeds
        if next(self.parameters()).device.type == "cuda":
            # GPU
            device_idx = torch.cuda.current_device()
            self.attention_seed = torch.cuda.default_generators[device_idx].seed()
            torch.cuda.manual_seed(self.attention_seed)
        else:
            # CPU
            self.attention_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(self.attention_seed)

    def _init_feed_forward_seed(self):
        """
            This function sets a new seed for the
            feed forward layer to make dropout deterministic
            for both forward calls: 1 normal forward
            call and 1 forward call in backward
            to recalculate activations.
        """

        # randomize seeds
        if next(self.parameters()).device.type == "cuda":
            # GPU
            device_idx = torch.cuda.current_device()
            self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()
            torch.cuda.manual_seed(self.feed_forward_seed)
        else:
            # CPU
            self.feed_forward_seed = int(torch.seed() % sys.maxsize)
            torch.manual_seed(self.feed_forward_seed)

    def forward(
            self,
            prev_attn_output,
            encoder_states,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            num_hashes=None,
            do_output_attentions=False,
    ):
        with torch.no_grad():
            # every forward pass we sample a different seed
            # for dropout and save for forward fn in backward pass
            # to have correct dropout
            self._init_attention_seed()
            attn_outputs = self.attention(
                encoder_states=encoder_states,
                hidden_states=hidden_states,
                head_mask=head_mask,
                attention_mask=attention_mask,
                num_hashes=num_hashes,
                do_output_attentions=do_output_attentions,
            )
            attn_output = attn_outputs.hidden_states

            # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
            # Y_1 = X_1 + f(X_2)
            attn_output = prev_attn_output + attn_output

            # free memory
            del prev_attn_output

            # every forward pass we sample a different seed
            # for dropout and save seed for forward fn in backward
            # to have correct dropout
            self._init_feed_forward_seed()
            # Y_2 = X_2 + g(Y_1)
            hidden_states = hidden_states + self.feed_forward(attn_output)

        return ReformerOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            attention_probs=attn_outputs.attention_probs,
            buckets=attn_outputs.buckets,
        )

    def backward_pass(
            self,
            next_attn_output,
            hidden_states,
            grad_attn_output,
            grad_hidden_states,
            attention_mask=None,
            head_mask=None,
            buckets=None,
    ):
        # Implements the backward pass for reversible ResNets.
        # A good blog post on how this works can be found here:
        # Implementation of RevNet (see Fig. 6 in https://towardsdatascience.com/illustrating-the-reformer-393575ac6ba0)
        # This code is heavily inspired by https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reversible.py

        with torch.enable_grad():
            next_attn_output.requires_grad = True

            # set seed to have correct dropout
            torch.manual_seed(self.feed_forward_seed)
            # g(Y_1)
            res_hidden_states = self.feed_forward(next_attn_output)
            res_hidden_states.backward(grad_hidden_states, retain_graph=True)

        with torch.no_grad():
            # X_2 = Y_2 - g(Y_1)
            hidden_states = hidden_states - res_hidden_states
            del res_hidden_states

            grad_attn_output = grad_attn_output + next_attn_output.grad
            next_attn_output.grad = None

        with torch.enable_grad():
            hidden_states.requires_grad = True

            # set seed to have correct dropout
            torch.manual_seed(self.attention_seed)
            # f(X_2)
            # use cached buckets for backprob if buckets not None for LSHSelfAttention
            output = self.attention(
                hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, buckets=buckets,
            ).hidden_states
            output.backward(grad_attn_output, retain_graph=True)

        with torch.no_grad():
            # X_1 = Y_1 - f(X_2)
            attn_output = next_attn_output - output
            del output, next_attn_output

            grad_hidden_states = grad_hidden_states + hidden_states.grad
            hidden_states.grad = None
            hidden_states = hidden_states.detach()

        return ReformerBackwardOutput(
            attn_output=attn_output,
            hidden_states=hidden_states,
            grad_attn_output=grad_attn_output,
            grad_hidden_states=grad_hidden_states,
        )


class ReformerDecoder(nn.Module):
    def __init__(self, pretrained_encoder: ReformerEncoder):
        super().__init__()
        self.dropout = pretrained_encoder.dropout

        self.layers = nn.ModuleList([ReformerDecoderLayer(layer) for layer in pretrained_encoder.layers])
        # Reformer is using Rev Nets, thus last layer outputs are concatenated and
        # Layer Norm is done over 2 * hidden_size
        self.layer_norm = pretrained_encoder.layer_norm

    def forward(
            self,
            encoder_states,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            num_hashes=None,
            do_output_hidden_states=False,
            do_output_attentions=False,
    ):
        # hidden_states and attention lists to be filled if wished
        all_hidden_states = []
        all_attentions = []

        # concat same tensor for reversible ResNet
        encoder_states = torch.cat([encoder_states, encoder_states], dim=-1)
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(
            hidden_states,
            self.layers,
            attention_mask,
            head_mask,
            num_hashes,
            all_hidden_states,
            all_attentions,
            do_output_hidden_states,
            do_output_attentions,
        )

        # Apply layer norm to concatenated hidden states
        hidden_states = self.layer_norm(hidden_states)

        # Apply dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        return ReformerEncoderOutput(
            hidden_states=hidden_states, all_hidden_states=all_hidden_states, all_attentions=all_attentions
        )


class ReformerDecoderModel(ReformerPreTrainedModel):
    def __init__(self, pretrained_model: ReformerModel):
        super().__init__(pretrained_model.config)
        self.config = pretrained_model.config
        assert (
                self.config.num_hidden_layers > 0
        ), "`config.attn_layers` is empty. Select at least one attn layer form ['lsh', 'local']"

        self.embeddings = pretrained_model.embeddings
        self.encoder = ReformerDecoder(pretrained_model.encoder)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(REFORMER_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            num_hashes=None,
            do_output_hidden_states=False,
            do_output_attentions=False,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        all_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        all_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``do_output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import ReformerModel, ReformerTokenizer
        import torch

        tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
        model =  ReformerModel.from_pretrained('google/reformer-crime-and-punishment')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        """

        # TODO(PVP): delete when PR to change output_attentions is made
        do_output_attentions = self.config.output_attentions
        do_output_hidden_states = self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()  # noqa: F841
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]  # noqa: F841
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert (
                len(input_shape) == 2
        ), "`input_ids` have be of shape `[batch_size, sequence_length]`, but got shape: {}".format(input_shape)

        # prepare head mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers, is_attention_chunked=True)

        # original sequence length for padding
        orig_sequence_length = input_shape[-1]

        # if needs padding
        least_common_mult_chunk_length = _get_least_common_mult_chunk_len(self.config)
        must_pad_to_match_chunk_length = input_shape[-1] % least_common_mult_chunk_length != 0

        if must_pad_to_match_chunk_length:
            padding_length = least_common_mult_chunk_length - input_shape[-1] % least_common_mult_chunk_length

            if self.training is True:
                raise ValueError(
                    "If training, sequence Length {} has to be a multiple of least common multiple chunk_length {}. Please consider padding the input to a length of {}.".format(
                        input_shape[-1], least_common_mult_chunk_length, input_shape[-1] + padding_length
                    )
                )

            # pad input
            input_ids, inputs_embeds, attention_mask, position_ids, input_shape = self._pad_to_mult_of_chunk_length(
                input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_shape=input_shape,
                padding_length=padding_length,
                padded_seq_length=least_common_mult_chunk_length,
                device=device,
            )

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            num_hashes=num_hashes,
            do_output_hidden_states=do_output_hidden_states,
            do_output_attentions=do_output_attentions,
        )
        sequence_output = encoder_outputs.hidden_states

        # if padding was applied
        if must_pad_to_match_chunk_length:
            sequence_output = sequence_output[:, :orig_sequence_length]

        outputs = (sequence_output,)
        # TODO(PVP): Replace by named tuple after namedtuples are introduced in the library.
        if do_output_hidden_states is True:
            outputs = outputs + (encoder_outputs.all_hidden_states,)
        if do_output_attentions is True:
            outputs = outputs + (encoder_outputs.all_attentions,)
        return outputs
