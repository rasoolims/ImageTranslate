import copy
import datetime
import os
import pickle
import sys
import time

import torch
import torch.utils.data as data_utils
from IPython.core import ultratb
from torchvision import transforms

import dataset
from image_doc_model import ImageSeq2Seq
from lm import LM
from option_parser import get_img_options_parser
from seq_gen import get_outputs_until_eos
from textprocessor import TextProcessor
from train_mass import MassTrainer
from utils import build_optimizer

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class ImageDocTrainer(MassTrainer):
    def train_epoch(self, data_iter: data_utils.DataLoader, step: int, dev_data_iter: data_utils.DataLoader = None,
                    saving_path: str = None, mt_dev_iter: data_utils.DataLoader = None, **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        sentences = 0
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()

            try:
                predictions = self.model(device=self.device, batch=batch, log_softmax=True)
                targets = [b["captions"][:, 1:].contiguous().view(-1) for b in batch]
                tgt_mask_flat = [b["caption_mask"][:, 1:].contiguous().view(-1) for b in batch]
                targets = torch.cat([targets[i][tgt_mask_flat[i]] for i in range(len(batch))])

                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean()
                loss.backward()

                loss = float(loss.data) * ntokens
                total_loss += loss
                cur_loss += loss
                total_tokens += ntokens
                tokens += ntokens
                sentences += sum([int(b["docs"].size(0)) + int(b["captions"].size(0)) for b in batch])

                if self.optimizer is not None:
                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1
            except RuntimeError as err:
                print("Error processing")
                for b in batch:
                    if isinstance(b, list):
                        b = b[0]
                    print(b["images"].size(), b["captions"].size(), b["docs"].size())
                print("****")

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model_to_save.save_checkpoint(saving_path)

                if step % 500 == 0 and dev_data_iter is not None:
                    self.validate(dev_data_iter)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model_to_save.save(saving_path + ".latest")

        if dev_data_iter is not None:
            self.validate(dev_data_iter)

        if mt_dev_iter is not None:
            bleu = self.eval_bleu(mt_dev_iter, saving_path)
            print("Pretraining BLEU:", bleu)

        return step

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is not None:
            mt_model, lm = ImageSeq2Seq.load(options.pretrained_path)
        else:
            if options.lm_path is None:
                lm = LM(text_processor=text_processor, size=options.model_size)
            else:
                lm = LM.load(options.lm_path)

            decoder = copy.deepcopy(lm.encoder) if options.sep_encoder else lm.encoder
            mt_model = ImageSeq2Seq(config=lm.config, encoder=lm.encoder, decoder=decoder, output_layer=lm.masked_lm,
                                    text_processor=lm.text_processor, checkpoint=options.checkpoint)

        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])

        print("Model initialization done!")

        train_data = dataset.ImageDocDataset(root_img_dir=options.image_dir,
                                             data_bin_file=options.train_path, transform=transform,
                                             max_doc_batch_capacity=options.img_capacity,
                                             text_processor=mt_model.text_processor)

        pin_memory = torch.cuda.is_available()

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        train_loader = data_utils.DataLoader(train_data, batch_size=num_batches, shuffle=False, pin_memory=pin_memory,
                                             collate_fn=collator)
        dev_loader = None
        if options.dev_path is not None:
            dev_data = dataset.MassDataset(batch_pickle_dir=options.dev_path,
                                           max_batch_capacity=options.total_capacity,
                                           max_batch=options.batch,
                                           pad_idx=mt_model.text_processor.pad_token_id(),
                                           max_seq_len=options.max_seq_len)
            dev_loader = data_utils.DataLoader(dev_data, batch_size=num_batches, shuffle=False,
                                               pin_memory=pin_memory, collate_fn=collator)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer, last_epoch = pickle.load(fp)
        else:
            optimizer, last_epoch = build_optimizer(mt_model, options.learning_rate, options.weight_decay), 0
        trainer = ImageDocTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                                  warmup=options.warmup, step=options.step,
                                  beam_width=options.beam_width, max_len_a=options.max_len_a,
                                  max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                                  last_epoch=last_epoch)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=options.mt_dev_path,
                                            max_batch_capacity=options.total_capacity,
                                            max_batch=int(options.batch / (options.beam_width * 2)),
                                            pad_idx=mt_model.text_processor.pad_token_id())
            mt_dev_loader = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

            print("creating reference")
            trainer.reference = []

            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )

            for batch in mt_dev_loader:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, dev_data_iter=dev_loader, mt_dev_iter=mt_dev_loader,
                                       saving_path=options.model_path, step=step)
            train_epoch += 1


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    ImageDocTrainer.train(options=options)
    print("Finished Training!")
