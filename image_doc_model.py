import copy
import pickle

import torch.nn.functional as F
from torchvision import models
from transformers.modeling_albert import *

from albert_seq2seq import MassSeq2Seq, future_mask, AlbertDecoderTransformer
from lm import LM
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


def init_net(embed_dim: int, dropout: float = 0.1, freeze: bool = False):
    model = models.resnet18(pretrained=True)
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


class ImageCaptionSeq2Seq(MassSeq2Seq):
    def __init__(self, config: AlbertConfig, encoder: AlbertModel, decoder, output_layer: AlbertMLMHead,
                 text_processor: TextProcessor, checkpoint: int = 5, freeze_image: bool = False,
                 share_decoder: bool = False):
        super(ImageCaptionSeq2Seq, self).__init__(config, encoder, decoder, output_layer, text_processor, checkpoint)
        self.image_model: ModifiedResnet = init_net(embed_dim=config.embedding_size, dropout=config.hidden_dropout_prob,
                                                    freeze=freeze_image)
        self.image_mapper = nn.Linear(config.embedding_size, config.hidden_size)

    def forward(self, batch, log_softmax: bool = False, **kwargs):
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]

        device = self.encoder.embeddings.word_embeddings.weight.device
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)
        caption_mask = batch["caption_mask"].to(device)
        langs = batch["langs"].unsqueeze(-1).expand(-1, captions.size(-1)).to(device)

        image_embeddings = self.image_mapper(self.image_model(images))

        subseq_mask = future_mask(caption_mask[:, :-1]).to(device)
        decoder_output = self.decoder(encoder_states=image_embeddings, input_ids=captions[:, :-1],
                                      input_ids_mask=caption_mask[:, :-1], tgt_attn_mask=subseq_mask,
                                      token_type_ids=langs[:, :-1])
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = caption_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    @staticmethod
    def load(out_dir: str, tok_dir: str, sep_decoder: bool, share_decoder: bool = False):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            decoder = copy.deepcopy(lm.encoder) if sep_decoder else lm.encoder
            mt_model = ImageCaptionSeq2Seq(config=config, encoder=lm.encoder, decoder=decoder,
                                           output_layer=lm.masked_lm,
                                           text_processor=lm.text_processor, checkpoint=checkpoint)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm


class ImageDocSeq2Seq(MassSeq2Seq):
    def __init__(self, config: AlbertConfig, encoder: AlbertModel, decoder, output_layer: AlbertMLMHead,
                 text_processor: TextProcessor, checkpoint: int = 5, freeze_image: bool = False,
                 share_decoder: bool = False):
        super(ImageDocSeq2Seq, self).__init__(config, encoder, decoder, output_layer, text_processor, checkpoint)
        self.image_model: ModifiedResnet = init_net(embed_dim=config.embedding_size, dropout=config.hidden_dropout_prob,
                                                    freeze=freeze_image)
        self.image_decoder = self.decoder.decoder if share_decoder else AlbertDecoderTransformer(
            AlbertTransformer(config))

    def forward(self, batch, log_softmax: bool = False, **kwargs):
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]

        device = self.encoder.embeddings.word_embeddings.weight.device
        images = batch["images"].to(device)
        captions = batch["captions"].to(device)
        docs = batch["docs"].to(device)
        doc_mask = batch["doc_mask"].to(device)
        caption_mask = batch["caption_mask"].to(device)
        doc_idx = batch["doc_idx"].to(device)
        doc_split = batch["doc_split"]
        src_langs = batch["langs"].unsqueeze(-1).expand(-1, docs.size(-1))
        caption_langs = batch["caption_langs"].unsqueeze(-1).expand(-1, captions.size(-1)).to(device)

        "Take in and process masked src and target sequences."
        doc_states = self.encode(docs, doc_mask, src_langs)[0]

        image_embeddings = self.image_model(images)

        # For each document, extract its image embedding.
        doc_image_embeddings = image_embeddings[doc_idx]

        # Attend images to documents.
        image_attended = self.image_decoder(encoder_states=doc_states, hidden_states=doc_image_embeddings,
                                            src_attn_mask=doc_mask)

        # Split back based on images
        image_attended_split = torch.split(image_attended, doc_split)

        max_list = list(map(lambda spl: torch.max(spl, dim=0).values, image_attended_split))
        max_images_attended = torch.stack(max_list, 0)
        subseq_mask = future_mask(caption_mask[:, :-1]).to(device)
        decoder_output = self.decoder(encoder_states=max_images_attended, input_ids=captions[:, :-1],
                                      input_ids_mask=caption_mask[:, :-1], tgt_attn_mask=subseq_mask,
                                      token_type_ids=caption_langs[:, :-1])
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = caption_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    @staticmethod
    def load(out_dir: str, tok_dir: str, sep_decoder: bool, share_decoder: bool):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            decoder = copy.deepcopy(lm.encoder) if sep_decoder else lm.encoder
            mt_model = ImageDocSeq2Seq(config=config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                       text_processor=lm.text_processor, checkpoint=checkpoint,
                                       share_decoder=share_decoder)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm
