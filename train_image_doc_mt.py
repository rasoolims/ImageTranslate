import copy
import datetime
import os
import sys
import time
from optparse import OptionParser

import torch
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb
from torchvision import transforms

import dataset
from image_doc_model import ImageSeq2Seq
from lm import LM
from loss import SmoothedNLLLoss
from parallel import DataParallelModel, DataParallelCriterion
from pytorch_lamb.pytorch_lamb import Lamb
from textprocessor import TextProcessor

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class Trainer:
    def __init__(self, model: ImageSeq2Seq, clip: int = 1, optimizer=None,
                 warmup: int = 12500, step: int = 125000, fp16: bool = False, fp16_opt_level: str = "01"):
        self.model: ImageSeq2Seq = model

        self.clip = clip
        self.optimizer = optimizer
        self.fp16 = fp16

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        if fp16:
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level=fp16_opt_level)

        if self.optimizer is not None:
            self.scheduler = optim.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup,
                                                                   num_training_steps=step)

        self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.num_gpu = torch.cuda.device_count()
        if self.num_gpu > 1:
            print("Let's use", self.num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

    @staticmethod
    def build_optimizer(model, learning_rate, weight_decay):
        return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)

    def train_epoch(self, data_iter: data_utils.DataLoader, step: int, max_grad_norm: float = 1.0,
                    valid_data_iter: data_utils.DataLoader = None, saving_path: str = None, ):
        if self.fp16:
            try:
                import apex
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

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

            predictions = self.model(device=self.device, batch=batch, log_softmax=True)
            targets = [b["captions"][:, 1:].contiguous().view(-1) for b in batch]
            tgt_mask_flat = [b["caption_mask"][:, 1:].contiguous().view(-1) for b in batch]
            targets = torch.cat([targets[i][tgt_mask_flat[i]] for i in range(len(batch))])

            ntokens = targets.size(0)

            if ntokens == 0:  # Nothing to predict!
                continue

            loss = self.criterion(predictions, targets).mean()
            if self.fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            loss = float(loss.data) * ntokens
            total_loss += loss
            cur_loss += loss
            total_tokens += ntokens
            tokens += ntokens
            sentences += sum([int(b["docs"].size(0)) + int(b["captions"].size(0)) for b in batch])

            if self.optimizer is not None:
                # We accumulate the gradients for both tasks!
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()
                step += 1

            if step % 50 == 0 and tokens > 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          step, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if step % 1000 == 0:
                    # Save every 1000 steps!
                    model_to_save.save_checkpoint(saving_path)

                if step % 500 == 0 and valid_data_iter is not None:
                    self.validate(valid_data_iter)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model_to_save.save(saving_path + ".latest")

        if valid_data_iter is not None:
            self.validate(valid_data_iter)
        return step

    def validate(self, valid_data_iter):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_valid_loss, total_valid_tokens = 0, 0
            for batch in valid_data_iter:
                predictions = self.model(device=self.device, batch=batch, log_softmax=True)
                targets = [b["captions"][:, 1:].contiguous().view(-1) for b in batch]
                tgt_mask_flat = [b["caption_mask"][:, 1:].contiguous().view(-1) for b in batch]
                targets = torch.cat([targets[i][tgt_mask_flat[i]] for i in range(len(batch))])
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean().data * ntokens
                total_valid_loss += float(loss)
                total_valid_tokens += ntokens

            valid_loss = total_valid_loss / total_valid_tokens
            print("Current valid loss", valid_loss)
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

            decoder = copy.DeepCopy(lm.encoder) if options.sep_encoder else lm.encoder
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
        valid_loader = None
        if options.valid_path is not None:
            valid_data = dataset.ImageDocDataset(root_img_dir=options.image_dir, data_bin_file=options.valid_path,
                                                 transform=transform,
                                                 max_doc_batch_capacity=options.total_capacity,
                                                 pad_index=mt_model.text_processor.pad_token_id())
            valid_loader = data_utils.DataLoader(valid_data, batch_size=num_batches, shuffle=False,
                                                 pin_memory=pin_memory,
                                                 collate_fn=collator)

        trainer = Trainer(model=mt_model,
                          optimizer=Trainer.build_optimizer(mt_model, options.learning_rate, options.weight_decay),
                          clip=options.clip, warmup=options.warmup, step=options.step, fp16=options.fp16,
                          fp16_opt_level=options.fp16_opt_level)

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=train_loader, valid_data_iter=valid_loader,
                                       saving_path=options.model_path,
                                       step=step)
            train_epoch += 1


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--train", dest="train_path", help="Path to the train data pickle files", metavar="FILE",
                      default=None)
    parser.add_option("--valid", dest="valid_path",
                      help="Path to the train data pickle files", metavar="FILE", default=None)
    parser.add_option("--image", dest="image_dir", help="Path to the image files", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--lm", dest="lm_path", help="LM pretrained model", metavar="FILE", default=None)
    parser.add_option("--pretrained", dest="pretrained_path", help="MT pretrained model", metavar="FILE", default=None)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capcity", type="int", default=40)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--lr", dest="learning_rate", help="Learning rate", type="float", default=0.002)
    parser.add_option("--warmup", dest="warmup", help="Number of warmup steps", type="int", default=12500)
    parser.add_option("--step", dest="step", help="Number of training steps", type="int", default=125000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    parser.add_option("--fp16", action="store_true", dest="fp16", help="use fp16; should be compatible", default=False)
    parser.add_option("--sep", action="store_true", dest="sep_encoder", help="Disjoint encoder/decoder", default=False)
    parser.add_option("--size", dest="model_size", help="1 base, 2 medium, 3 small, 4 toy", type="int", default=3)
    parser.add_option("--checkpoint", dest="checkpoint", help="Number of checkpoints to average", type="int", default=5)
    parser.add_option(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    options = get_options()
    print(options)
    Trainer.train(options=options)
    print("Finished Training!")
