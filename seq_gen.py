import torch
import torch.nn as nn
import torch.nn.functional as F

from albert_seq2seq import AlbertSeq2Seq


def get_outputs_until_eos(eos, outputs, remove_first_token: bool = False):
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
    def __init__(self, seq2seq_model: AlbertSeq2Seq, beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5):
        super(BeamDecoder, self).__init__()
        self.seq2seq_model = seq2seq_model
        self.beam_width = beam_width
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

    def forward(self, device, src_inputs, tgt_langs, src_mask):
        """

        :param device:
        :param src_inputs:
        :param tgt_langs: First token that is language identifier
        :param src_mask:
        :return:
        """
        batch_size = src_inputs.size(0)
        encoder_states = self.seq2seq_model.encode(device, src_inputs, src_mask)[0]
        eos = self.seq2seq_model.text_processor.sep_token_id()

        src_mask = src_mask.to(device)
        first_position_output = tgt_langs.unsqueeze(1).to(device)
        top_beam_outputs = first_position_output
        top_beam_scores = torch.zeros(first_position_output.size()).to(first_position_output.device)

        max_len = min(int(self.max_len_a * src_inputs.size(1) + self.max_len_b) + 1,
                      self.seq2seq_model.encoder.embeddings.position_embeddings.num_embeddings)
        for i in range(1, max_len):
            cur_outputs = top_beam_outputs.view(-1, top_beam_outputs.size(-1))
            cur_scores = top_beam_scores.view(-1).unsqueeze(-1)
            output_mask = torch.ones(cur_outputs.size()).to(cur_outputs.device)
            enc_states = encoder_states if i == 1 else torch.repeat_interleave(encoder_states, self.beam_width, 0)
            cur_src_mask = src_mask if i == 1 else torch.repeat_interleave(src_mask, self.beam_width, 0)
            decoder_states = self.seq2seq_model.decoder(enc_states, cur_outputs, cur_src_mask, output_mask)
            output = F.log_softmax(self.seq2seq_model.output_layer(decoder_states[:, -1, :]), dim=-1)
            beam_scores = (cur_scores + output).view(batch_size, -1)
            top_scores, indices = torch.topk(beam_scores, k=self.beam_width, dim=1)
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

            seen_eos = torch.any(top_beam_outputs[:, 0, :] == eos, 1)

            if torch.all(seen_eos):
                break
        outputs = top_beam_outputs[:, 0, :]
        actual_outputs = get_outputs_until_eos(eos, outputs)

        return actual_outputs


class GreedyDecoder(nn.Module):
    def __init__(self, seq2seq_model: AlbertSeq2Seq, max_len_a: float = 1.1, max_len_b: int = 5):
        super(GreedyDecoder, self).__init__()
        self.seq2seq_model = seq2seq_model
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

    def forward(self, device, src_inputs, tgt_langs, src_mask):
        """

        :param device:
        :param src_inputs:
        :param tgt_langs: First token that is language identifier
        :param src_mask:
        :return:
        """
        encoder_states = self.seq2seq_model.encode(device, src_inputs, src_mask)[0]
        eos = self.seq2seq_model.text_processor.sep_token_id()
        pad_id = self.seq2seq_model.text_processor.pad_token_id()
        src_mask = src_mask.to(device)
        max_len = min(int(self.max_len_a * src_inputs.size(1) + self.max_len_b) + 1,
                      self.seq2seq_model.encoder.embeddings.position_embeddings.num_embeddings)

        outputs = tgt_langs.unsqueeze(1).to(device)
        seen_eos = torch.zeros(outputs.size(0), dtype=torch.bool).to(outputs.device)
        for i in range(1, max_len):
            output_mask = (outputs != pad_id)
            decoder_states = self.seq2seq_model.decoder(encoder_states, outputs, src_mask, output_mask)
            output = self.seq2seq_model.output_layer(decoder_states[:, -1, :])
            best_outputs = torch.argmax(output, dim=1)
            outputs = torch.cat([outputs, best_outputs.unsqueeze(1)], dim=1)
            seen_eos |= best_outputs == eos
            if torch.all(seen_eos):
                break

        return get_outputs_until_eos(eos, outputs)
