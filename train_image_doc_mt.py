import copy
import datetime
import os
import sys
import time

import torch
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb
from torchvision import transforms

import dataset
import train_lm
import train_mt
from image_doc_model import ImageSeq2Seq
from lm import LM
from loss import SmoothedNLLLoss
from parallel import DataParallelModel, DataParallelCriterion
from textprocessor import TextProcessor

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class Trainer:
    def __init__(self, model: ImageSeq2Seq, clip: int = 1, optimizer=None, warmup: int = 12500, step: int = 125000):
        self.model: ImageSeq2Seq = model
        self.clip = clip
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if self.optimizer is not None:
            self.scheduler = optim.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup,
                                                                   num_training_steps=step)

        self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.num_gpu = torch.cuda.device_count()
        if self.num_gpu > 1:
            print("Let's use", self.num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

    def train_epoch(self, data_iter: data_utils.DataLoader, step: int, max_grad_norm: float = 1.0,
                    dev_data_iter: data_utils.DataLoader = None, saving_path: str = None, ):
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    step += 1
            except:
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
        return step

    def validate(self, dev_data_iter):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_dev_loss, total_dev_tokens = 0, 0
            for batch in dev_data_iter:
                predictions = self.model(device=self.device, batch=batch, log_softmax=True)
                targets = [b["captions"][:, 1:].contiguous().view(-1) for b in batch]
                tgt_mask_flat = [b["caption_mask"][:, 1:].contiguous().view(-1) for b in batch]
                targets = torch.cat([targets[i][tgt_mask_flat[i]] for i in range(len(batch))])
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean().data * ntokens
                total_dev_loss += float(loss)
                total_dev_tokens += ntokens

            dev_loss = total_dev_loss / total_dev_tokens
            print("Current dev loss", dev_loss)
            model.train()

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

        mt_model.save_config_and_tok(options.model_path)

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
                                             max_doc_batch_capacity=options.total_capacity,
                                             pad_index=mt_model.text_processor.pad_token_id())

        pin_memory = torch.cuda.is_available()

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        train_loader = data_utils.DataLoader(train_data, batch_size=num_batches, shuffle=True, pin_memory=pin_memory,
                                             collate_fn=collator)
        dev_loader = None
        if options.dev_path is not None:
            dev_data = dataset.ImageDocDataset(root_img_dir=options.image_dir, data_bin_file=options.dev_path,
                                               transform=transform,
                                               max_doc_batch_capacity=options.total_capacity,
                                               pad_index=mt_model.text_processor.pad_token_id())
            dev_loader = data_utils.DataLoader(dev_data, batch_size=num_batches, shuffle=False,
                                               pin_memory=pin_memory, collate_fn=collator)

        trainer = Trainer(model=mt_model,
                          optimizer=train_lm.LMTrainer.build_optimizer(mt_model, options.learning_rate,
                                                                       options.weight_decay),
                          clip=options.clip, warmup=options.warmup, step=options.step)

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, dev_data_iter=dev_loader,
                                       saving_path=options.model_path,
                                       step=step)
            train_epoch += 1


def get_options_parser():
    parser = train_mt.get_option_parser()
    parser.add_option("--image", dest="image_dir", help="Path to the image files", metavar="FILE", default=None)
    return parser


if __name__ == "__main__":
    parser = get_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    Trainer.train(options=options)
    print("Finished Training!")
