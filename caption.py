import datetime
from optparse import OptionParser

import torch
import torch.utils.data as data_utils
from apex import amp

import dataset
from image_model import ImageCaptioning
from parallel import DataParallelModel
from seq2seq import Seq2Seq
from seq_gen import BeamDecoder


def get_lm_option_parser():
    parser = OptionParser()
    parser.add_option("--input", dest="input_path", metavar="FILE", default=None)
    parser.add_option("--target", dest="target_lang", type="str", default=None)
    parser.add_option("--output", dest="output_path", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=16)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None)
    parser.add_option("--beam", dest="beam_width", type="int", default=4)
    parser.add_option("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type="float", default=1.3)
    parser.add_option("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type="int", default=5)
    parser.add_option("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type="float", default=0.8)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    return parser


def caption_batch(batch, generator, text_processor):
    proposals = [b["proposal"] for b in batch]
    dst_langs = [b["langs"] for b in batch]
    first_tokens = [b["first_tokens"] for b in batch]
    images = [b["images"] for b in batch]
    max_len = [b["max_len"] for b in batch][0]
    outputs = generator(images=images, first_tokens=first_tokens, tgt_langs=dst_langs,
                                         pad_idx=text_processor.pad_token_id(), proposals=proposals,
                                         max_len=max_len)
    if torch.cuda.device_count() > 1:
        new_outputs = []
        for output in outputs:
            new_outputs += output
        outputs = new_outputs

    mt_output = list(map(lambda x: text_processor.tokenizer.decode(x[1:].numpy()), outputs))
    img_ids = [b["img_ids"] for b in batch][0]
    return mt_output, img_ids


def build_data_loader(options, text_processor):
    print(datetime.datetime.now(), "Binarizing test data")
    image_data = dataset.ImageCaptionTestDataset(root_img_dir="", data_bin_file=options.input_path, max_capacity=10000,
                                                 text_processor=text_processor, max_img_per_batch=options.batch)
    collator = dataset.ImageTextCollator()
    pin_memory = torch.cuda.is_available()
    return data_utils.DataLoader(image_data, batch_size=1, shuffle=False, pin_memory=pin_memory, collate_fn=collator)


def build_model(options):
    model = Seq2Seq.load(ImageCaptioning, options.model_path, tok_dir=options.tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_gpu = torch.cuda.device_count()
    generator = BeamDecoder(model, beam_width=options.beam_width, max_len_a=options.max_len_a,
                            max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio)
    if options.fp16:
        generator = amp.initialize(generator, opt_level="O2")
    if num_gpu > 1:
        generator = DataParallelModel(generator)
    return generator, model.text_processor


if __name__ == "__main__":
    parser = get_lm_option_parser()
    (options, args) = parser.parse_args()
    generator, text_processor = build_model(options)
    test_loader = build_data_loader(options, text_processor)

    sen_count = 0
    with open(options.output_path, "w") as writer:
        with torch.no_grad():
            for batch in test_loader:
                mt_output, paths = caption_batch(batch, generator, text_processor)
                sen_count += len(mt_output)
                print(datetime.datetime.now(), "Captioned", sen_count, "images!", end="\r")
                writer.write("\n".join([x[0] + "\t" + y for x, y in zip(paths, mt_output)]))
                writer.write("\n")

    print(datetime.datetime.now(), "Translated", sen_count, "sentences")
    print(datetime.datetime.now(), "Done!")
