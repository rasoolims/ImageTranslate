import torch.nn.functional as F

from seq2seq import Seq2Seq, future_mask


class MassSeq2Seq(Seq2Seq):
    def forward(self, src_inputs, tgt_inputs, src_langs, tgt_langs=None, pad_idx: int = 0,
                tgt_positions=None, log_softmax: bool = False, proposals=None):
        """
        :param mask_pad_mask: # Since MASS also generates MASK tokens, we do not backpropagate them during training.
        :return:
        """
        device = self.encoder.embeddings.word_embeddings.weight.device
        if isinstance(tgt_inputs, list):
            assert len(tgt_inputs) == 1
            tgt_inputs = tgt_inputs[0]
            src_langs = src_langs[0]
        if isinstance(src_inputs, list):
            src_inputs = src_inputs[0]
        if isinstance(tgt_positions, list):
            tgt_positions = tgt_positions[0]

        tgt_inputs = tgt_inputs.to(device)
        src_pads = src_inputs != pad_idx
        tgt_mask = tgt_inputs != pad_idx

        if tgt_langs is not None:
            # Use back-translation loss
            return super().forward(src_inputs=src_inputs, src_mask=src_pads, tgt_inputs=tgt_inputs, proposals=proposals,
                                   tgt_mask=tgt_mask, src_langs=src_langs, tgt_langs=tgt_langs, log_softmax=log_softmax)

        "Take in and process masked src and target sequences."
        src_pads = src_pads.to(device)
        src_inputs = src_inputs.to(device)
        src_langs_t = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        src_langs_t = src_langs_t.to(device)
        batch_lang = int(src_langs[0])
        encoder_states = self.encode(src_inputs, src_pads, src_langs_t)[0]

        tgt_langs = src_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        tgt_positions = tgt_positions.to(device)

        subseq_mask = future_mask(tgt_mask[:, :-1])
        decoder = self.decoder if not self.lang_dec else self.decoder[batch_lang]
        output_layer = self.output_layer if not not (self.lang_dec or self.tie_embed) else self.output_layer[batch_lang]

        decoder_output = decoder(encoder_states=encoder_states, input_ids=tgt_inputs[:, :-1],
                                 encoder_attention_mask=src_pads, tgt_attention_mask=subseq_mask,
                                 position_ids=tgt_positions[:, :-1] if tgt_positions is not None else None,
                                 token_type_ids=tgt_langs[:, :-1])
        if self.use_proposals:
            decoder_output = self.attend_proposal(decoder_output, proposals, pad_idx)
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))

        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs
