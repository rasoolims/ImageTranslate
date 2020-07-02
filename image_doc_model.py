import copy
import pickle

import torch.nn.functional as F
from transformers.modeling_albert import *

from albert_seq2seq import MassSeq2Seq, future_mask, AlbertDecoderTransformer
from image_model import init_net, ModifiedResnet
from lm import LM
from textprocessor import TextProcessor


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

        image_embeddings = self.image_mapper(self.image_model(images))[batch["split"]]

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
