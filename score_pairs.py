import marshal
import math
import operator
from optparse import OptionParser

import torch
import torch.nn.functional as F
from apex import amp
from torch.nn.utils.rnn import pad_sequence

from seq2seq import Seq2Seq, future_mask
from textprocessor import TextProcessor


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model", metavar="FILE", default=None)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capacity", type="int", default=2000)
    parser.add_option("--data", dest="data", metavar="FILE", default=None)
    parser.add_option("--output", dest="output", metavar="FILE", default=None)
    parser.add_option("--resume", dest="resume_index", type="int", default=0)
    parser.add_option("--end", dest="end_index", type="int", default=-1)
    parser.set_default("model_size", 6)
    return parser


tok_sen = lambda s: text_processor.tokenize_one_sentence(s)[:512]


def create_batches(sen_ids, sentences, src2dst_dict, dst2src_dict, text_processor: TextProcessor, resume_index=0,
                   end_index=-1):
    print(len(sen_ids), len(src2dst_dict), len(dst2src_dict))

    print("Getting batches...")
    index = 0

    for dct in [src2dst_dict, dst2src_dict]:
        for sid in dct.keys():
            index += 1
            if index >= end_index and end_index > 0:
                break
            if index <= resume_index:
                continue
            tids = list(dct[sid])
            source_tokenized = torch.LongTensor(tok_sen(sentences[sid]))
            trans_cands = list(map(lambda i: torch.LongTensor(tok_sen(sentences[i])), tids))
            candidates = pad_sequence(trans_cands, batch_first=True, padding_value=text_processor.pad_token_id())
            target_langs = list(map(lambda i: text_processor.lang_id(sentences[i].strip().split(" ")[0]), tids))
            src_lang = torch.LongTensor([text_processor.lang_id(sentences[sid].strip().split(" ")[0])])
            yield sid, source_tokenized, torch.LongTensor(tids), candidates, src_lang, torch.LongTensor(target_langs)


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()

    print("Loading text processor...")
    text_processor = TextProcessor(options.tokenizer_path)
    num_processors = max(torch.cuda.device_count(), 1)

    print("Loading model...")
    model = Seq2Seq.load(options.model, tok_dir=options.tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_gpu = torch.cuda.device_count()

    assert num_gpu <= 1
    if options.fp16:
        model = amp.initialize(model, opt_level="O2")

    max_capacity = options.total_capacity * 1000000
    with torch.no_grad(), open(options.output, "w") as writer:
        print("Loading data...")
        with open(options.data, "rb") as fp:
            sen_ids, src2dst_dict, dst2src_dict = marshal.load(fp)
        sentences = list(sen_ids.keys())

        print("Scoring candidates")
        for i, batch in enumerate(
                create_batches(sen_ids, sentences, src2dst_dict, dst2src_dict, text_processor, options.resume_index,
                               options.end_index)):
            try:
                sid, src_input, tids_all, tgt_inputs_all, src_lang, dst_langs_all = batch
                cur_capacity = 2 * (max(int(src_input.size(0)), int(tgt_inputs_all.size(1))) ** 3) * int(
                    tgt_inputs_all.size(0))
                split_size = int(math.ceil(cur_capacity / max_capacity))
                split_size = max(1, int(math.floor(len(tids_all) / split_size)))

                tgt_inputs_spl = torch.split(tgt_inputs_all, split_size)
                tids_spl = torch.split(tids_all, split_size)
                dst_langs_spl = torch.split(dst_langs_all, split_size)

                trans_score = dict()
                for spl_i in range(len(tgt_inputs_spl)):
                    src_input = src_input.view(-1, src_input.size(0)).to(device)
                    src_mask = (src_input != text_processor.pad_token_id())
                    src_lang = src_lang.to(device)
                    encoder_states = model.encode(src_input, src_mask, src_lang.expand(src_input.size()))[0]

                    tgt_inputs, tids, dst_langs = tgt_inputs_spl[spl_i], tids_spl[spl_i], dst_langs_spl[spl_i]
                    tgt_mask = (tgt_inputs != text_processor.pad_token_id()).to(device)

                    tgt_inputs = tgt_inputs.to(device)
                    dst_langs = dst_langs.to(device)
                    batch_lang = int(dst_langs[0])
                    subseq_mask = future_mask(tgt_mask[:, :-1])
                    if subseq_mask.device != tgt_inputs.device:
                        subseq_mask = subseq_mask.to(device)

                    decoder = model.decoder if not model.lang_dec else model.decoder[batch_lang]
                    output_layer = model.output_layer if not model.lang_dec else model.output_layer[batch_lang]

                    enc_states = encoder_states.expand(len(tgt_inputs), encoder_states.size(1), encoder_states.size(2))
                    src_mask_spl = src_mask.expand(len(tgt_inputs), src_mask.size(1))
                    dst_langs = dst_langs.unsqueeze(1).expand(tgt_inputs.size())
                    decoder_output = decoder(enc_states, tgt_inputs[:, :-1], tgt_mask[:, :-1], src_mask_spl,
                                             subseq_mask,
                                             token_type_ids=dst_langs[:, :-1])
                    predictions = F.log_softmax(output_layer(decoder_output), dim=-1)

                    predictions = predictions.view(-1, predictions.size(-1))
                    targets = tgt_inputs[:, 1:].contiguous().view(-1)
                    w_losses = tgt_mask[:, 1:] * predictions.gather(1, targets.view(-1, 1)).squeeze(-1).view(
                        len(tgt_mask),
                        -1)
                    loss = torch.sum(w_losses, dim=1)
                    loss = torch.div(loss, torch.sum(tgt_mask[:, 1:], dim=-1)).cpu().numpy()
                    for j, l in enumerate(loss):
                        tid = tids[j]
                        trans_score[tid] = l
                sorted_dict = sorted(trans_score.items(), key=operator.itemgetter(1), reverse=True)
                tid, score = sorted_dict[0]
                writer.write(sentences[sid] + "\t" + sentences[tid] + "\t" + str(score))
                writer.write("\n")

                print(options.resume_index + i + 1, len(src2dst_dict) + len(dst2src_dict), end="\r")
            except RuntimeError:
                pass

    print("\nDone!")
