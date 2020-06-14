import torch
import torch.nn as nn
import torch.nn.functional as F


def get_outputs_until_eos(eos, outputs, pad_idx, remove_first_token: bool = False):
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    found_eos = torch.nonzero(outputs == eos).cpu()
    found_pad = torch.nonzero(outputs == pad_idx).cpu()
    outputs = outputs.cpu()
    actual_outputs = {}
    for idx in range(found_eos.size(0)):
        r, c = int(found_eos[idx, 0]), int(found_eos[idx, 1])
        if r not in actual_outputs:
            # disregard end of sentence in output!
            actual_outputs[r] = outputs[r, 1 if remove_first_token else 0:c]
    final_outputs = []
    if len(actual_outputs) < int(outputs.size(0)):
        for idx in range(found_pad.size(0)):
            r, c = int(found_pad[idx, 0]), int(found_pad[idx, 1])
            if r not in actual_outputs:
                # disregard end of sentence in output!
                actual_outputs[r] = outputs[r, 1 if remove_first_token else 0:c]

    for r in range(outputs.size(0)):
        if r not in actual_outputs:
            actual_outputs[r] = outputs[r, 1 if remove_first_token else 0:]
        final_outputs.append(actual_outputs[r])

    return final_outputs


class BatchedBeamElement():
    def __init__(self, outputs, scores=None):
        self.outputs = outputs

        if scores is not None:
            assert len(scores) == self.outputs.size(0)
            self.scores = scores
        else:
            self.scores = torch.zeros(self.outputs.size()).to(self.outputs.device)


class BeamDecoder(nn.Module):
    def __init__(self, seq2seq_model, beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8):
        super(BeamDecoder, self).__init__()
        self.seq2seq_model = seq2seq_model
        self.beam_width = beam_width
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.len_penalty_ratio = len_penalty_ratio

    def len_penalty(self, cur_beam_elements: torch.Tensor, eos_idx: int):
        """
        Based on https://arxiv.org/pdf/1609.08144.pdf, section 7
        :param cur_beam_elements: of size (batch_size * beam_width) * length [might have eos before length]
        :return:
        """
        lengths = [int(cur_beam_elements.size(-1)) + 5] * int(cur_beam_elements.size(0))
        found_eos = torch.nonzero(cur_beam_elements == eos_idx).cpu()
        eos_indices = {}
        for idx in range(found_eos.size(0)):
            r, c = int(found_eos[idx, 0]), int(found_eos[idx, 1])
            if r not in eos_indices:
                eos_indices[r] = c
                lengths[r] = c + 6

        length_penalty = torch.pow(torch.tensor(lengths).to(cur_beam_elements.device) / 6.0, self.len_penalty_ratio)
        return length_penalty.unsqueeze(-1)

    def forward(self, device, src_inputs, src_sizes, first_tokens, src_mask, src_langs, tgt_langs, pad_idx,
                max_len: int = None,
                unpad_output: bool = True):
        """

        :param device:
        :param src_inputs:
        :param first_tokens: First token that is language identifier
        :param src_mask:
        :return:
        """
        batch_size = src_inputs.size(0)
        src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        encoder_states = self.seq2seq_model.encode(device, src_inputs, src_mask, src_langs)[0]
        eos = self.seq2seq_model.text_processor.sep_token_id()

        src_mask = src_mask.to(device)
        first_position_output = first_tokens.unsqueeze(1).to(device)
        top_beam_outputs = first_position_output
        top_beam_scores = torch.zeros(first_position_output.size()).to(first_position_output.device)

        max_len_func = lambda s: min(int(self.max_len_a * s + self.max_len_b),
                                     self.seq2seq_model.encoder.embeddings.position_embeddings.num_embeddings)
        if max_len is None:
            max_len = max_len_func(src_inputs.size(1))
        max_lens = torch.LongTensor(list(map(lambda x: max_len_func(x), src_sizes))).to(device)
        for i in range(1, max_len):
            cur_outputs = top_beam_outputs.view(-1, top_beam_outputs.size(-1))

            if int(torch.sum(torch.any(cur_outputs == eos, 1))) == self.beam_width * batch_size:
                # All beam items have end-of-sentence token.
                break

            reached_eos_limit = max_lens < (i + 1)
            reached_eos_limit = reached_eos_limit.unsqueeze(-1).expand(-1, self.beam_width)

            # Keeps track of those items for which we know should be masked for their score, because they already reached
            # end of sentence.
            eos_mask = torch.any(cur_outputs == eos, 1)

            cur_scores = top_beam_scores.view(-1).unsqueeze(-1)
            output_mask = torch.ones(cur_outputs.size()).to(cur_outputs.device)
            enc_states = encoder_states if i == 1 else torch.repeat_interleave(encoder_states, self.beam_width, 0)
            dst_langs = tgt_langs.unsqueeze(-1).expand(-1, cur_outputs.size(1)).to(device)
            if i > 1:
                dst_langs = torch.repeat_interleave(dst_langs, self.beam_width, 0)
            cur_src_mask = src_mask if i == 1 else torch.repeat_interleave(src_mask, self.beam_width, 0)
            decoder_states = self.seq2seq_model.decoder(enc_states, cur_outputs, cur_outputs != pad_idx, cur_src_mask,
                                                        output_mask, token_type_ids=dst_langs)
            output = F.log_softmax(self.seq2seq_model.output_layer(decoder_states[:, -1, :]), dim=-1)
            output[eos_mask] = 0  # Disregard those items with EOS in them!
            if i > 1:
                output[reached_eos_limit.contiguous().view(-1)] = 0  # Disregard those items over size limt!
            beam_scores = ((cur_scores + output) / self.len_penalty(cur_outputs, eos)).view(batch_size, -1)
            top_scores, indices = torch.topk(beam_scores, k=self.beam_width, dim=1)

            if i > 1:
                # Regardless of output, if reached to the maximum length, make it PAD!
                indices[reached_eos_limit] = pad_idx

            flat_indices = indices.view(-1)
            word_indices = torch.stack([torch.LongTensor([range(output.size(1))])] * self.beam_width, dim=1).view(-1)
            if i > 1:
                beam_indices = indices / output.size(-1)
                beam_indices_to_select = torch.stack([beam_indices] * top_beam_outputs.size(-1), dim=2)
                beam_to_use = top_beam_outputs.gather(1, beam_indices_to_select).view(-1, i)
            else:
                beam_to_use = torch.repeat_interleave(top_beam_outputs, self.beam_width, 0)
            word_indices = word_indices[flat_indices].unsqueeze(-1).to(beam_to_use.device)
            top_beam_outputs = torch.cat([beam_to_use, word_indices], dim=1).view(batch_size, self.beam_width, i + 1)

            top_beam_scores = top_scores

        outputs = top_beam_outputs[:, 0, :]
        if unpad_output:
            actual_outputs = get_outputs_until_eos(eos, outputs, pad_idx=pad_idx)
        else:
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            outputs = outputs.cpu()
            actual_outputs = list(map(lambda i: outputs[i], range(outputs.size(0))))

        # Force free memory.
        del outputs
        del top_beam_outputs

        return actual_outputs
