import torch.nn.functional as F
from torchvision import models
from transformers.modeling_albert import *

from mass_seq2seq import MassSeq2Seq, future_mask
from textprocessor import TextProcessor


class ModifiedResnet(models.ResNet):
    def _forward_impl(self, x):
        input = x
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        grid_hidden = self.layer4(x)
        grid_hidden = grid_hidden.view(grid_hidden.size(0), grid_hidden.size(1), -1)
        grid_hidden = grid_hidden.permute((0, 2, 1))
        if self.dropout > 0:
            grid_hidden = F.dropout(grid_hidden, p=self.dropout)
        grid_outputs = F.relu(self.fc(grid_hidden))
        location_embedding = self.location_embedding.weight.unsqueeze(0)
        out = grid_outputs + location_embedding
        out_norm = self.layer_norm(out)
        if self.dropout > 0:
            out_norm = F.dropout(out_norm, p=self.dropout)
        return out_norm


def init_net(embed_dim: int, dropout: float = 0.1, freeze: bool = False, depth: int = 1):
    if depth == 1:
        model = models.resnet18(pretrained=True)
    elif depth == 2:
        model = models.resnet34(pretrained=True)
    elif depth == 3:
        model = models.resnet50(pretrained=True)
    elif depth == 4:
        model = models.resnet101(pretrained=True)
    elif depth == 5:
        model = models.resnet152(pretrained=True)

    model.__class__ = ModifiedResnet
    model.dropout = dropout
    model.layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    current_weight = model.state_dict()["fc.weight"]
    model.fc = torch.nn.Linear(in_features=current_weight.size()[1], out_features=embed_dim, bias=False)
    model.fc.train()

    # Learning embedding of each CNN region.
    model.location_embedding = nn.Embedding(49, embed_dim)
    model.location_embedding.train(True)

    return model


class ImageMassSeq2Seq(MassSeq2Seq):
    def __init__(self, is_bert: bool, text_processor: TextProcessor, freeze_image: bool = False,
                 resnet_depth: int = 1, lang_dec: bool = False, use_proposals: bool = False, enc_layer: int = 6,
                 dec_layer: int = 3, embed_dim: int = 768, intermediate_dim: int = 3072):
        super(ImageMassSeq2Seq, self).__init__(is_bert=is_bert, text_processor=text_processor,
                                               lang_dec=lang_dec, use_proposals=use_proposals, enc_layer=enc_layer,
                                               dec_layer=dec_layer, embed_dim=embed_dim,
                                               intermediate_dim=intermediate_dim)
        self.image_model: ModifiedResnet = init_net(embed_dim=self.config.hidden_size,
                                                    dropout=self.config.hidden_dropout_prob,
                                                    freeze=freeze_image, depth=resnet_depth)
        self.multimodal_attention_gate = nn.Parameter(torch.zeros(1, self.config.hidden_size).fill_(0.1),
                                                      requires_grad=True)

        self.image_attention_w = nn.Linear(self.config.hidden_size, 1)  # For Constrastive loss
        self.encoder_attention_w = nn.Linear(self.config.hidden_size, 1)  # For Constrastive loss

    def encode(self, src_inputs, src_mask, src_langs, images=None):
        encoder_states = super().encode(src_inputs, src_mask, src_langs)
        if images is not None:
            device = self.encoder.embeddings.word_embeddings.weight.device
            if isinstance(images, list):
                images = images[0]
            if images.device != device:
                images = images.to(device)
            image_embeddings = self.image_model(images)
            return encoder_states[0], image_embeddings
        return encoder_states

    def forward(self, src_inputs=None, src_pads=None, tgt_inputs=None, src_langs=None, tgt_langs=None, pad_idx: int = 0,
                tgt_positions=None, batch=None, neg_samples=None, neg_mask=None, proposals=None,
                log_softmax: bool = False, **kwargs):
        device = self.encoder.embeddings.word_embeddings.weight.device
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]
        if isinstance(src_langs, list):
            src_langs = src_langs[0]
        if isinstance(src_pads, list):
            src_pads = src_pads[0]
        if isinstance(src_inputs, list):
            src_inputs = src_inputs[0]
        if isinstance(tgt_positions, list):
            tgt_positions = tgt_positions[0]
        if isinstance(tgt_inputs, list):
            tgt_inputs = tgt_inputs[0]
        if isinstance(proposals, list):
            proposals = proposals[0]

        if batch is None:
            return super().forward(src_inputs=src_inputs, src_pads=src_pads, tgt_inputs=tgt_inputs, src_langs=src_langs,
                                   tgt_langs=tgt_langs, pad_idx=pad_idx, tgt_positions=tgt_positions,
                                   proposals=proposals,
                                   log_softmax=log_softmax)

        assert src_inputs is not None
        images = batch["images"].to(device)
        src_pads = src_pads.to(device)
        src_inputs = src_inputs.to(device)
        src_langs_t = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1)).to(device)
        encoder_states, image_embeddings = self.encode(src_inputs, src_pads, src_langs_t, images)

        if neg_samples is None:
            assert tgt_inputs is not None
            tgt_inputs = tgt_inputs.to(device)
            tgt_mask = tgt_inputs != pad_idx

            batch_lang = int(src_langs[0])

            decoder = self.decoder if not self.lang_dec else self.decoder[batch_lang]
            output_layer = self.output_layer if not self.lang_dec else self.output_layer[batch_lang]
            tgt_langs = src_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
            if tgt_positions is not None:
                tgt_positions = tgt_positions[:, :-1].to(device)

            subseq_mask = future_mask(tgt_mask[:, :-1])

            text_decoder_output = decoder(encoder_states=encoder_states, input_ids=tgt_inputs[:, :-1],
                                          encoder_attention_mask=src_pads,
                                          tgt_attention_mask=subseq_mask,
                                          position_ids=tgt_positions,
                                          token_type_ids=tgt_langs[:, :-1])
            image_decoder_output = decoder(encoder_states=image_embeddings, input_ids=tgt_inputs[:, :-1],
                                           tgt_attention_mask=subseq_mask,
                                           position_ids=tgt_positions,
                                           token_type_ids=tgt_langs[:, :-1])
            eps = 1e-7
            sig_gate = torch.sigmoid(self.multimodal_attention_gate + eps)
            decoder_output = sig_gate * text_decoder_output + (1 - sig_gate) * image_decoder_output

            if self.use_proposals:
                decoder_output = self.attend_proposal(decoder_output, proposals, pad_idx)

            diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
            tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
            non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
            outputs = output_layer(non_padded_outputs)
            if log_softmax:
                outputs = F.log_softmax(outputs, dim=-1)
            return outputs
        else:
            if isinstance(neg_samples, list):
                neg_samples = neg_samples[0]
                neg_mask = neg_mask[0]
            neg_langs = src_langs[0].squeeze().unsqueeze(-1).expand(len(neg_samples), neg_samples.size(-1)).to(device)

            neg_samples = neg_samples.to(device)
            neg_mask = neg_mask.to(device)
            neg_states = self.encode(neg_samples, neg_mask, neg_langs)[0]
            neg_attend_scores = self.encoder_attention_w(neg_states).squeeze(-1)
            neg_attend_scores.masked_fill_(~neg_mask, -10000.0)
            neg_attend = nn.Softmax(dim=1)(neg_attend_scores)
            neg_state_attended = torch.einsum("bfd,bf->bd", neg_states, neg_attend)

            encoder_attend_scores = self.encoder_attention_w(encoder_states).squeeze(-1)
            encoder_attend_scores.masked_fill_(~src_pads, -10000.0)
            encoder_attend = nn.Softmax(dim=1)(encoder_attend_scores)
            encoder_state_attended = torch.einsum("bfd,bf->bd", encoder_states, encoder_attend)

            text_vectors = torch.cat([encoder_state_attended, neg_state_attended])

            image_attend = nn.Softmax(dim=1)(self.image_attention_w(image_embeddings).squeeze(-1))
            image_state_attended = torch.einsum("bfd,bf->bd", image_embeddings, image_attend)

            text_norm = torch.norm(text_vectors, dim=-1, p=2).unsqueeze(-1) + 1e-8
            text_vectors = torch.div(text_vectors, text_norm)
            image_norm = torch.norm(image_state_attended, dim=-1, p=2).unsqueeze(-1) + 1e-8
            image_state_attended = torch.div(image_state_attended, image_norm)

            cross_dot = torch.mm(image_state_attended, text_vectors.T)
            denom = torch.log(torch.sum(torch.exp(cross_dot), dim=-1) + 1e-8)
            nominator = torch.diagonal(cross_dot[:, :len(encoder_state_attended)], 0) + 1e-8
            log_neg = torch.sum(denom - nominator) / len(encoder_state_attended)
            return log_neg
