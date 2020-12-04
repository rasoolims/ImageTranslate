import pickle

from torchvision import models
from transformers.modeling_albert import *

import lm_config
from bert_seq2seq import BertEncoderModel, BertConfig
from faster_rcnn_feats import *
from mass_seq2seq import MassSeq2Seq, future_mask
from seq2seq import Seq2Seq
from textprocessor import TextProcessor


class ModifiedResnet(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):

        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)

    def forward(self, x, fcnn: ModifiedFasterRCNN = None):
        return self._forward_impl(x, fcnn=fcnn)

    def _forward_impl(self, x, fcnn: ModifiedFasterRCNN = None):
        input = x
        x1 = self.conv1(input)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x4 = self.maxpool(x3)

        x5 = self.layer1(x4)
        x6 = self.layer2(x5)
        x7 = self.layer3(x6)
        x8 = self.layer4(x7)
        grid_hidden = x8.view(x8.size(0), x8.size(1), -1)
        grid_hidden = grid_hidden.permute((0, 2, 1))
        if self.dropout > 0:
            grid_hidden = F.dropout(grid_hidden, p=self.dropout)
        grid_outputs = F.relu(self.fc(grid_hidden))
        location_embedding = self.location_embedding.weight.unsqueeze(0)
        out = grid_outputs + location_embedding

        # Getting object features from faster RCNN
        if fcnn is not None:
            with torch.no_grad():
                fcnn_results = fcnn(x)
                max_feature_nums = max(map(lambda x: x["boxes"].size(0), fcnn_results))
        else:
            max_feature_nums = 0

        if max_feature_nums > 0:  # Found objects
            with torch.no_grad():
                feat_dim = fcnn_results[0]["features"].size(-1)
                features = torch.zeros((len(fcnn_results), max_feature_nums, feat_dim + 7),
                                       dtype=location_embedding.dtype).fill_(1e-4).to(location_embedding.device)
                object_labels = torch.zeros((len(fcnn_results), max_feature_nums), dtype=torch.long).to(
                    location_embedding.device)
                for i in range(len(fcnn_results)):
                    x1 = fcnn_results[i]["boxes"][:, 0] / 800
                    x2 = fcnn_results[i]["boxes"][:, 2] / 800
                    y1 = fcnn_results[i]["boxes"][:, 1] / 800
                    y2 = fcnn_results[i]["boxes"][:, 3] / 800
                    w = x2 - x1
                    h = y2 - y1
                    wh = h * w
                    locs = torch.stack([x1, x2, y1, y2, w, h, wh], dim=-1)
                    features[i, :fcnn_results[i]["features"].size(0)] = torch.cat([fcnn_results[i]["features"], locs],
                                                                                  dim=-1)
                    object_labels[i, :fcnn_results[i]["labels"].size(0)] = fcnn_results[i]["labels"]
            object_embed = self.object_embedding(object_labels)
            object_feats = torch.cat([object_embed, features], dim=-1)
            object_feat_fc = F.relu(self.object_feat_fc(object_feats))

            compound_out = torch.cat([object_feat_fc, out], dim=1)

            out_norm = self.layer_norm(compound_out)
        else:
            out_norm = self.layer_norm(out)

        if self.dropout > 0 and self.training:
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

    model.object_feat_fc = torch.nn.Linear(in_features=1024 + 7 + embed_dim, out_features=embed_dim, bias=False)
    model.object_feat_fc.train()

    # Learning embedding of each CNN region.
    model.location_embedding = nn.Embedding(49, embed_dim)
    model.location_embedding.train(True)
    model.object_embedding = nn.Embedding(91, embed_dim)
    model.object_embedding.train(True)

    return model


class ImageMassSeq2Seq(MassSeq2Seq):
    def __init__(self, text_processor: TextProcessor, freeze_image: bool = False, resnet_depth: int = 1,
                 lang_dec: bool = False, use_proposals: bool = False, tie_embed: bool = False, enc_layer: int = 6,
                 dec_layer: int = 3, embed_dim: int = 768, intermediate_dim: int = 3072):
        super(ImageMassSeq2Seq, self).__init__(text_processor=text_processor, tie_embed=tie_embed,
                                               lang_dec=lang_dec, use_proposals=use_proposals, enc_layer=enc_layer,
                                               dec_layer=dec_layer, embed_dim=embed_dim,
                                               intermediate_dim=intermediate_dim, freeze_image=freeze_image,
                                               resnet_depth=resnet_depth)
        self.image_model: ModifiedResnet = init_net(embed_dim=self.config.hidden_size,
                                                    dropout=self.config.hidden_dropout_prob,
                                                    freeze=freeze_image, depth=resnet_depth)
        self.multimodal_attention_gate = nn.Parameter(torch.zeros(1, self.config.hidden_size).fill_(0.1),
                                                      requires_grad=True)

        self.image_attention_w = nn.Linear(self.config.hidden_size, 1)  # For Constrastive loss
        self.encoder_attention_w = nn.Linear(self.config.hidden_size, 1)  # For Constrastive loss

    def encode(self, src_inputs, src_mask, src_langs, images=None, fcnn: ModifiedFasterRCNN = None):
        encoder_states = super().encode(src_inputs, src_mask, src_langs)
        if images is not None:
            device = self.encoder.embeddings.word_embeddings.weight.device
            if isinstance(images, list):
                images = images[0]
            if images.device != device:
                images = images.to(device)
            image_embeddings = self.image_model(images, fcnn=fcnn)
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
            return super().forward(src_inputs=src_inputs, tgt_inputs=tgt_inputs, src_langs=src_langs,
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
            output_layer = self.output_layer if (not self.lang_dec) and self.tie_embed else self.output_layer[
                batch_lang]
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

            text_norm = torch.norm(text_vectors, dim=-1, p=2).unsqueeze(-1) + 1e-4
            text_vectors = torch.div(text_vectors, text_norm)
            image_norm = torch.norm(image_state_attended, dim=-1, p=2).unsqueeze(-1) + 1e-4
            image_state_attended = torch.div(image_state_attended, image_norm)

            cross_dot = torch.mm(image_state_attended, text_vectors.T)
            denom = torch.log(torch.sum(torch.exp(cross_dot), dim=-1) + 1e-4)
            nominator = torch.diagonal(cross_dot[:, :len(encoder_state_attended)], 0) + 1e-4
            log_neg = torch.sum(denom - nominator) / len(encoder_state_attended)
            return log_neg


class ImageCaptioning(Seq2Seq):
    def __init__(self, text_processor: TextProcessor, freeze_image: bool = False, resnet_depth: int = 1,
                 lang_dec: bool = False, use_proposals: bool = False, tie_embed: bool = False, enc_layer: int = 6,
                 dec_layer: int = 3, embed_dim: int = 768, intermediate_dim: int = 3072):
        super(ImageCaptioning, self).__init__(text_processor=text_processor, tie_embed=tie_embed,
                                              lang_dec=lang_dec, use_proposals=use_proposals, enc_layer=enc_layer,
                                              dec_layer=dec_layer, embed_dim=embed_dim,
                                              intermediate_dim=intermediate_dim, freeze_image=freeze_image,
                                              resnet_depth=resnet_depth)
        self.image_model: ModifiedResnet = init_net(embed_dim=self.config.hidden_size,
                                                    dropout=self.config.hidden_dropout_prob,
                                                    freeze=freeze_image, depth=resnet_depth)

    def encode(self, src_inputs=None, src_mask=None, src_langs=None, images=None, fcnn: ModifiedFasterRCNN = None):
        if images is not None:
            device = self.encoder.embeddings.word_embeddings.weight.device
            if isinstance(images, list):
                images = images[0]
            if images.device != device:
                images = images.to(device)
            image_embeddings = self.image_model(images, fcnn=fcnn)
            return image_embeddings
        else:
            encoder_states = super().encode(src_inputs, src_mask, src_langs)
            return encoder_states

    def forward(self, src_inputs=None, src_pads=None, tgt_inputs=None, src_langs=None, tgt_langs=None, tgt_mask=None,
                pad_idx: int = 0, tgt_positions=None, batch=None, proposals=None, log_softmax: bool = False,
                encode_only: bool = False, **kwargs):
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
        if isinstance(tgt_mask, list):
            tgt_mask = tgt_mask[0]

        if batch is None:
            return super().forward(src_inputs=src_inputs, src_mask=src_pads, tgt_inputs=tgt_inputs, src_langs=src_langs,
                                   tgt_langs=tgt_langs, proposals=proposals, log_softmax=log_softmax)

        images = batch["images"].to(device)
        image_embeddings = self.encode(images=images)
        if encode_only:
            return image_embeddings

        assert tgt_inputs is not None
        tgt_inputs = tgt_inputs.to(device)
        tgt_mask = tgt_mask.to(device)

        batch_lang = int(src_langs[0])

        decoder = self.decoder if not self.lang_dec else self.decoder[batch_lang]
        output_layer = self.output_layer if (not self.lang_dec) and self.tie_embed else self.output_layer[batch_lang]
        tgt_langs = src_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        if tgt_positions is not None:
            tgt_positions = tgt_positions[:, :-1].to(device)

        subseq_mask = future_mask(tgt_mask[:, :-1])

        decoder_output = decoder(encoder_states=image_embeddings, input_ids=tgt_inputs[:, :-1],
                                 encoder_attention_mask=src_pads,
                                 tgt_attention_mask=subseq_mask,
                                 position_ids=tgt_positions,
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


class Caption2Image(nn.Module):
    def __init__(self, text_processor: TextProcessor, enc_layer: int = 6, embed_dim: int = 768,
                 intermediate_dim: int = 3072):
        super(Caption2Image, self).__init__()
        self.text_processor: TextProcessor = text_processor
        self.config = lm_config.get_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                           pad_token_id=text_processor.pad_token_id(),
                                           bos_token_id=text_processor.bos_token_id(),
                                           eos_token_id=text_processor.sep_token_id(),
                                           enc_layer=enc_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim)

        self.enc_layer = enc_layer
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.config["type_vocab_size"] = len(text_processor.languages)
        self.config = BertConfig(**self.config)

        self.encoder = BertEncoderModel(self.config)
        self.encoder.init_weights()

        self.input_attention = nn.Linear(self.config.hidden_size, 1)
        self.decoder = nn.Linear(self.config.hidden_size, 49 * self.config.hidden_size)

    def encode(self, src_inputs, src_mask, src_langs):
        device = self.encoder.embeddings.word_embeddings.weight.device
        if src_inputs.device != device:
            src_inputs = src_inputs.to(device)
            src_mask = src_mask.to(device)
            src_langs = src_langs.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask, token_type_ids=src_langs)
        return (encoder_states, None)

    def forward(self, src_inputs, src_mask, src_langs):
        "Take in and process masked src and target sequences."
        device = self.encoder.embeddings.word_embeddings.weight.device
        if isinstance(src_langs, list):
            src_langs = src_langs[0]
        if isinstance(src_mask, list):
            src_mask = src_mask[0]
        if isinstance(src_inputs, list):
            src_inputs = src_inputs[0]

        src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        src_inputs = src_inputs.to(device)
        src_langs = src_langs.to(device)
        if src_mask.device != device:
            src_mask = src_mask.to(device)

        encoder_states = self.encode(src_inputs, src_mask, src_langs)[0]

        if self.training:
            encoder_states = F.dropout(encoder_states, p=self.config.hidden_dropout_prob)

        attention_scores = self.input_attention(encoder_states).squeeze(-1)
        attention_scores.masked_fill_(~src_mask, -10000.0)
        attention_prob = nn.Softmax(dim=1)(attention_scores)
        sentence_embeddings = torch.einsum("bfd,bf->bd", encoder_states, attention_prob)

        image_embeddings = self.decoder(sentence_embeddings)

        return image_embeddings

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump((self.enc_layer, self.embed_dim, self.intermediate_dim), fp)
        try:
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        except:
            torch.cuda.empty_cache()
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))

    @staticmethod
    def load(out_dir: str, tok_dir: str):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            enc_layer, embed_dim, intermediate_dim = pickle.load(fp)

            mt_model = Caption2Image(text_processor=text_processor, enc_layer=enc_layer, embed_dim=embed_dim,
                                     intermediate_dim=intermediate_dim)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict"), map_location=device),
                                     strict=False)
            return mt_model
