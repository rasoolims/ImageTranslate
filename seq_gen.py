import torch
import torch.nn as nn
import torch.nn.functional as F

from albert_seq2seq import AlbertSeq2Seq


def get_outputs_until_eos(eos, outputs, remove_first_token: bool = True):
    found_eos = torch.nonzero(outputs == eos)
    actual_outputs = {}
    for idx in range(found_eos.size(0)):
        r, c = found_eos[idx, 0], found_eos[idx, 1]
        if r not in actual_outputs:
            actual_outputs[int(r)] = outputs[r,
                                     1 if remove_first_token else 0:c]  # disregard end of sentence in output!
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
    def __init__(self, seq2seq_model: AlbertSeq2Seq, beam_width: int = 4):
        super(BeamDecoder, self).__init__()
        self.seq2seq_model = seq2seq_model
        self.beam_width = beam_width

    def forward(self, device, src_inputs, tgt_inputs, src_mask, tgt_mask):
        batch_size = src_inputs.size(0)
        encoder_states = self.seq2seq_model.encode(device, src_inputs, src_mask)[0]
        eos = self.seq2seq_model.text_processor.sep_token_id()

        first_position_output = tgt_inputs[:, 0].unsqueeze(1)
        top_beam_outputs = first_position_output
        top_beam_scores = torch.zeros(first_position_output.size()).to(tgt_inputs.device)

        for i in range(1, self.seq2seq_model.encoder.embeddings.position_embeddings.num_embeddings):
            cur_outputs = top_beam_outputs.view(-1, top_beam_outputs.size(-1))
            cur_scores = top_beam_scores.view(-1).unsqueeze(-1)
            output_mask = torch.ones(cur_outputs.size()).to(cur_outputs.device)
            enc_states = encoder_states if i == 1 else torch.repeat_interleave(encoder_states, self.beam_width, 0)
            cur_src_mask = src_mask if i == 1 else torch.repeat_interleave(src_mask, self.beam_width, 0)
            decoder_states = self.seq2seq_model.decoder(enc_states, cur_outputs, cur_src_mask, output_mask)
            output = F.log_softmax(self.seq2seq_model.output_layer(decoder_states[:, -1, :]), dim=1)
            beam_scores = (cur_scores + output).view(batch_size, -1)
            top_scores, indices = torch.topk(beam_scores, k=self.beam_width, dim=1)
            flat_indices = indices.view(-1)
            word_indices = torch.stack([torch.LongTensor([range(output.size(1))])] * self.beam_width, dim=1).view(-1)
            if i > 1:
                beam_indices = indices / output.size(-1)
                beam_indices_to_select = torch.stack([beam_indices] * top_beam_outputs.size(-1), dim=2)
                beam_elements_to_use = top_beam_outputs.gather(1, beam_indices_to_select).view(-1, i)
            else:
                beam_elements_to_use = torch.repeat_interleave(top_beam_outputs, self.beam_width, 0)
            top_beam_outputs = torch.cat([beam_elements_to_use, word_indices[flat_indices].unsqueeze(-1)], dim=1).view(
                batch_size, self.beam_width, i + 1)

            top_beam_scores = top_scores

            seen_eos = torch.any(top_beam_outputs[:, 0, :] == eos, 1)

            if torch.all(seen_eos):
                break
        outputs = top_beam_outputs[:, 0, :]
        actual_outputs = get_outputs_until_eos(eos, outputs)

        return actual_outputs
