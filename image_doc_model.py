import copy
import pickle

import torch.nn.functional as F
from transformers.modeling_albert import *

from albert_seq2seq import AlbertSeq2Seq, future_mask, AlbertDecoderTransformer
from image_model import init_net, ModifiedResnet
from lm import LM
from textprocessor import TextProcessor


class ImageSeq2Seq(AlbertSeq2Seq):
    def __init__(self, config: AlbertConfig, encoder: AlbertModel, decoder, output_layer: AlbertMLMHead,
                 text_processor: TextProcessor, checkpoint: int = 5, freeze_image: bool = False):
        super(ImageSeq2Seq, self).__init__(config, encoder, decoder, output_layer, text_processor, checkpoint)
        self.image_model: ModifiedResnet = init_net(embed_dim=config.hidden_size, dropout=config.hidden_dropout_prob,
                                                    freeze=freeze_image)
        self.image_decoder = AlbertImageTransformer(AlbertTransformer(config))

    def forward(self, device, batch, log_softmax: bool = False, **kwargs):
        if isinstance(batch, list):
            assert len(batch) == 1
            batch = batch[0]

        images = batch["images"].to(device)
        captions = batch["captions"].to(device)
        docs = batch["docs"].to(device)
        doc_mask = batch["doc_mask"].to(device)
        caption_mask = batch["caption_mask"].to(device)
        doc_idx = batch["doc_idx"].to(device)
        doc_split = batch["doc_split"]

        "Take in and process masked src and target sequences."
        doc_states = self.encode(device, docs, doc_mask)[0]

        image_embeddings = self.image_model(images)

        # For each document, extract its image embedding.
        doc_image_embeddings = image_embeddings[doc_idx]

        # Attend images to documents.
        image_attended = self.image_decoder(doc_states, doc_image_embeddings, doc_mask)

        # Split back based on images
        image_attended_split = torch.split(image_attended, doc_split)

        max_images_attended = torch.stack([torch.max(spl, dim=0).values for spl in image_attended_split], 0)
        subseq_mask = future_mask(caption_mask[:, :-1]).to(device)
        decoder_output = self.decoder(encoder_states=max_images_attended, input_ids=captions[:, :-1],
                                      tgt_attention_mask=subseq_mask)
        diag_outputs = torch.stack([decoder_output[:, d, d, :] for d in range(decoder_output.size(2))], 1)
        diag_outputs_flat = diag_outputs.view(-1, diag_outputs.size(-1))
        tgt_mask_flat = caption_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_mask_flat]
        outputs = self.output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    @staticmethod
    def load(out_dir: str, tok_dir: str, sep_decoder: bool):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            config, checkpoint = pickle.load(fp)
            lm = LM(text_processor=text_processor, config=config)
            decoder = copy.deepcopy(lm.encoder) if sep_decoder else lm.encoder
            mt_model = ImageSeq2Seq(config=config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                    text_processor=lm.text_processor, checkpoint=checkpoint)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict")))
            return mt_model, lm


class AlbertImageTransformer(AlbertDecoderTransformer):
    def __init__(self, albert_transformer: AlbertTransformer):
        super(AlbertImageTransformer, self).__init__(albert_transformer)

    def forward(self, encoder_states, image_embeddings, src_attention_mask, **kwargs):
        extended_mask = src_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_mask = extended_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_mask = (1.0 - extended_mask) * -10000.0

        hidden_states = image_embeddings
        for i in range(self.config.num_hidden_layers):
            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                encoder_states,
                hidden_states,
                src_attention_mask=extended_mask,
                tgt_attention_mask=None,
            )
            hidden_states = layer_group_output[0]

        return hidden_states
