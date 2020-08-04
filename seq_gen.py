import torch
import torch.nn as nn
import torch.nn.functional as F


def get_outputs_until_eos(eos, outputs, size_limit=None, remove_first_token: bool = False):
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    found_eos = torch.nonzero(outputs == eos).cpu()
    outputs = outputs.cpu()
    actual_outputs = {}
    for idx in range(found_eos.size(0)):
        r, c = int(found_eos[idx, 0]), int(found_eos[idx, 1])
        if r not in actual_outputs:
            # disregard end of sentence in output!
            actual_outputs[r] = outputs[r, 1 if remove_first_token else 0:c]
    final_outputs = []
    for r in range(outputs.size(0)):
        if r not in actual_outputs:
            last_index = len(outputs[r]) if size_limit is None else int(size_limit[r])
            actual_outputs[r] = outputs[r, 1 if remove_first_token else 0:last_index]
        final_outputs.append(actual_outputs[r])

    return final_outputs


class BeamDecoder(nn.Module):
    def __init__(self, seq2seq_model, beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8):
        super(BeamDecoder, self).__init__()
        self.seq2seq_model = seq2seq_model
        self.beam_width = beam_width
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.len_penalty_ratio = len_penalty_ratio

    def len_penalty(self, lengths: torch.Tensor):
        """
        Based on https://arxiv.org/pdf/1609.08144.pdf, section 7
        :param cur_beam_elements: of size (batch_size * beam_width) * length [might have eos before length]
        :return:
        """
        length_penalty = torch.pow((lengths + 6.0) / 6.0, self.len_penalty_ratio)
        return length_penalty.unsqueeze(-1)

    def forward(self, src_inputs=None, src_sizes=None, first_tokens=None, src_mask=None, src_langs=None, tgt_langs=None,
                pad_idx=None, max_len: int = None, unpad_output: bool = True, beam_width: int = None, images=None,
                proposals=None):
        """

        :param device:
        :param src_inputs:
        :param first_tokens: First token that is language identifier
        :param src_mask:
        :return:
        """
        if isinstance(tgt_langs, list):
            assert len(tgt_langs) == 1
            tgt_langs = tgt_langs[0]

            first_tokens = first_tokens[0]
        if isinstance(src_langs, list):
            src_langs = src_langs[0]
            src_mask = src_mask[0]
            src_sizes = src_sizes[0]
            src_inputs = src_inputs[0]
        if isinstance(images, list):
            images = images[0]
        if isinstance(proposals, list):
            proposals = proposals[0]

        if beam_width is None:
            beam_width = self.beam_width
        device = self.seq2seq_model.encoder.embeddings.word_embeddings.weight.device
        batch_lang = int(tgt_langs[0])
        if src_inputs is not None:
            batch_size = src_inputs.size(0)
            src_mask = src_mask.to(device)
        else:
            batch_size = images.size(0)

        images = images.to(device)
        if self.seq2seq_model.encoder.embeddings.word_embeddings.weight.dtype == torch.float16:
            images = images.half()
            
        if src_inputs is not None and images is None:
            src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
            encoder_states = self.seq2seq_model.encode(src_inputs, src_mask, src_langs)[0]
        elif src_inputs is None:
            encoder_states = self.seq2seq_model.encode(images=images)  # = image embeddings
        else:
            encoder_states, image_embeddings = self.seq2seq_model.encode(src_inputs, src_mask, src_langs, images)
        eos = self.seq2seq_model.text_processor.sep_token_id()

        first_position_output = first_tokens.unsqueeze(1).to(device)
        top_beam_outputs = first_position_output
        top_beam_scores = torch.zeros(first_position_output.size()).to(first_position_output.device)

        max_len_func = lambda s: min(int(self.max_len_a * s + self.max_len_b),
                                     self.seq2seq_model.encoder.embeddings.position_embeddings.num_embeddings)

        if max_len is None:
            max_len = max_len_func(src_inputs.size(1))
        if src_inputs is None:
            max_lens = torch.LongTensor([max_len] * batch_size).to(device)
        else:
            max_lens = torch.LongTensor(list(map(lambda x: max_len_func(x), src_sizes))).to(device)

        cur_size = torch.zeros(top_beam_outputs.size(0)).to(device) if beam_width > 1 else None

        seq2seq_model = (
            self.seq2seq_model.module if hasattr(self.seq2seq_model, "module") else self.seq2seq_model
        )
        vocab = torch.stack([torch.LongTensor([range(seq2seq_model.config.vocab_size)])] * beam_width, dim=1).view(
            -1).to(device)

        for i in range(1, max_len):
            cur_outputs = top_beam_outputs.view(-1, top_beam_outputs.size(-1))

            if int(torch.sum(torch.any(cur_outputs == eos, 1))) == beam_width * batch_size:
                # All beam items have end-of-sentence token.
                break

            reached_eos_limit = max_lens < (i + 1)
            reached_eos_limit = reached_eos_limit.unsqueeze(-1).expand(-1, beam_width)

            # Keeps track of those items for which we know should be masked for their score, because they already reached
            # end of sentence.
            eos_mask = torch.any(cur_outputs == eos, 1)
            cur_scores = top_beam_scores.view(-1).unsqueeze(-1)
            output_mask = torch.ones(cur_outputs.size()).to(cur_outputs.device)
            enc_states = encoder_states if i == 1 else torch.repeat_interleave(encoder_states, beam_width, 0)
            dst_langs = tgt_langs.unsqueeze(-1).expand(-1, cur_outputs.size(1)).to(device)
            if i > 1:
                dst_langs = torch.repeat_interleave(dst_langs, beam_width, 0)
            if src_inputs is not None:
                cur_src_mask = src_mask if i == 1 else torch.repeat_interleave(src_mask, beam_width, 0)
            else:
                cur_src_mask = None

            decoder = self.seq2seq_model.decoder if not self.seq2seq_model.lang_dec else self.seq2seq_model.decoder[
                batch_lang]
            output_layer = self.seq2seq_model.output_layer if not self.seq2seq_model.lang_dec else \
                self.seq2seq_model.output_layer[batch_lang]

            if images is None or src_inputs is None:
                decoder_states = decoder(encoder_states=enc_states, input_ids=cur_outputs,
                                         encoder_attention_mask=cur_src_mask,
                                         tgt_attention_mask=output_mask, token_type_ids=dst_langs)
            else:
                text_decoder_output = decoder(encoder_states=enc_states, input_ids=cur_outputs,
                                              encoder_attention_mask=cur_src_mask, tgt_attention_mask=output_mask,
                                              token_type_ids=dst_langs)
                image_decoder_output = decoder(encoder_states=image_embeddings, input_ids=cur_outputs,
                                               tgt_attention_mask=output_mask, token_type_ids=dst_langs)
                eps = 1e-7
                sig_gate = torch.sigmoid(self.seq2seq_model.multimodal_attention_gate + eps)
                decoder_states = sig_gate * text_decoder_output + (1 - sig_gate) * image_decoder_output

            decoder_states = decoder_states[:, -1, :]
            if self.seq2seq_model.use_proposals:
                decoder_states = self.seq2seq_model.attend_proposal(decoder_states, proposals, pad_idx)

            output = F.log_softmax(output_layer(decoder_states), dim=-1)
            output[eos_mask] = 0  # Disregard those items with EOS in them!
            if i > 1:
                output[reached_eos_limit.contiguous().view(-1)] = 0  # Disregard those items over size limt!
            if beam_width > 1:
                beam_scores = ((cur_scores + output) / self.len_penalty(cur_size.view(-1))).view(batch_size, -1)
            else:
                beam_scores = (cur_scores + output).view(batch_size, -1)

            top_scores, indices = torch.topk(beam_scores, k=beam_width, dim=1)

            if i > 1:
                # Regardless of output, if reached to the maximum length, make it PAD!
                indices[reached_eos_limit] = pad_idx

            flat_indices = indices.view(-1)

            if i > 1:
                # Regardless of output, if already reached EOS, make it PAD!
                flat_indices[eos_mask] = pad_idx

            if i > 1:
                beam_indices = indices / output.size(-1)
                beam_indices_to_select = torch.stack([beam_indices] * top_beam_outputs.size(-1), dim=2)
                beam_to_use = top_beam_outputs.gather(1, beam_indices_to_select).view(-1, i)
                sizes_to_use = cur_size.gather(1, beam_indices).view(-1) if beam_width > 1 else None
            else:
                beam_to_use = torch.repeat_interleave(top_beam_outputs, beam_width, 0)
                sizes_to_use = torch.repeat_interleave(cur_size, beam_width, 0) if beam_width > 1 else None
            word_indices = vocab[flat_indices].unsqueeze(-1)
            top_beam_outputs = torch.cat([beam_to_use, word_indices], dim=1).view(batch_size, beam_width, i + 1)
            if beam_width > 1:
                cur_size = (sizes_to_use + ~(word_indices.squeeze() == pad_idx)).view(batch_size, beam_width)
            top_beam_scores = top_scores

        outputs = top_beam_outputs[:, 0, :]
        if unpad_output:
            actual_outputs = get_outputs_until_eos(eos, outputs, size_limit=max_lens)
        else:
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            outputs = outputs.cpu()
            actual_outputs = list(map(lambda i: outputs[i], range(outputs.size(0))))

        # Force free memory.
        del outputs
        del top_beam_outputs

        return actual_outputs
