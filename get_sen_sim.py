import sys
from typing import List

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb

from option_parser import get_img_options_parser
from sen_sim import SenSim
from seq_gen import get_outputs_until_eos
from train_image_mt import ImageMTTrainer

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class SenSimEval(ImageMTTrainer):
    def eval(self, saving_path: str = None,
             mt_dev_iter: List[data_utils.DataLoader] = None):
        batch_zip, shortest = self.get_batch_zip(None, None, mt_dev_iter)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        with open(saving_path, "w") as w:
            for i, batches in enumerate(mt_dev_iter):
                for batch in batches:
                    try:
                        with torch.no_grad():
                            src_inputs = batch["src_texts"].squeeze(0)
                            src_mask = batch["src_pad_mask"].squeeze(0)
                            tgt_inputs = batch["dst_texts"].squeeze(0)
                            tgt_mask = batch["dst_pad_mask"].squeeze(0)
                            src_langs = batch["src_langs"].squeeze(0)
                            dst_langs = batch["dst_langs"].squeeze(0)
                            if src_inputs.size(0) < self.num_gpu:
                                continue
                            sims = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                              src_mask=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                              tgt_langs=dst_langs, normalize=False)
                            srcs = get_outputs_until_eos(model.text_processor.sep_token_id(), src_inputs,
                                                         remove_first_token=True)
                            targets = get_outputs_until_eos(model.text_processor.sep_token_id(), tgt_inputs,
                                                            remove_first_token=True)
                            src_txts = list(map(lambda src: model.text_processor.tokenizer.decode(src.numpy()), srcs))
                            target_txts = list(
                                map(lambda tgt: model.text_processor.tokenizer.decode(tgt.numpy()), targets))
                            for s in range(len(sims)):
                                w.write(src_txts[s] + "\t" + target_txts[s] + "\t" + str(float(sims[s])) + "\n")
                            print(i, "/", len(batches), end="\r")


                    except RuntimeError as err:
                        print(repr(err))
                        torch.cuda.empty_cache()
                print("\n")

    @staticmethod
    def sim(options):
        mt_model, text_processor = SenSim.load(options.model_path, tok_dir=options.tokenizer_path)

        print("Model initialization done!")

        trainer = SenSimEval(model=mt_model, mask_prob=options.mask_prob, optimizer=None, clip=options.clip,
                             fp16=options.fp16)

        pin_memory = torch.cuda.is_available()
        mt_dev_loader = SenSimEval.get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer, )
        trainer.eval(mt_dev_iter=mt_dev_loader, saving_path=options.output)


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    SenSimEval.sim(options=options)
    print("Finished Training!")
