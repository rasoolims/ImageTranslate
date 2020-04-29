import datetime
import os
import sys
import time
from optparse import OptionParser

import sacrebleu
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import transformers.optimization as optim
from IPython.core import ultratb

import dataset
from albert_seq2seq import AlbertSeq2Seq
from lm import LM
from parallel import DataParallelModel, DataParallelCriterion
from pytorch_lamb.pytorch_lamb import Lamb
from seq_gen import BeamDecoder, get_outputs_until_eos
from textprocessor import TextProcessor

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class Trainer:
    def __init__(self, model: AlbertSeq2Seq, mask_prob: float = 0.15, clip: int = 1, optimizer=None,
                 warmup: float = 0.1,
                 warmup_steps: int = 125000, fp16: bool = False, fp16_opt_level: str = "01"):
        self.model = model

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
            self.scheduler = optim.get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(warmup * warmup_steps), num_training_steps=warmup_steps
            )
        self.mask_prob = mask_prob
        self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.num_gpu = torch.cuda.device_count()
        if self.num_gpu > 1:
            print("Let's use", self.num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

        self.generator = BeamDecoder(model)
        self.reference = None

    @staticmethod
    def build_optimizer(model, learning_rate, weight_decay):
        return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)

    def train_epoch(self, data_iter: data_utils.DataLoader, valid_data_iter: data_utils.DataLoader,
                    best_valid_loss: float, saving_path: str, max_grad_norm: float = 1.0):
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

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            src_inputs = batch["src_texts"].squeeze(0)
            src_mask = batch["src_pad_mask"].squeeze(0)
            tgt_inputs = batch["dst_texts"].squeeze(0)
            tgt_mask = batch["dst_pad_mask"].squeeze(0)

            if src_inputs.size(0) < self.num_gpu:
                continue

            predictions = self.model(device=self.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                     src_mask=src_mask, tgt_mask=tgt_mask, log_softmax=True, flatten=True)
            targets = tgt_inputs[:, 1:].contiguous().view(-1)
            ntokens = targets.size(0)

            if ntokens == 0:  # Nothing to predict!
                continue

            loss = self.criterion(predictions, targets).mean()
            if self.fp16:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if self.optimizer is not None:
                if self.fp16:
                    torch.nn.utils.clip_grad_norm_(apex.amp.master_params(self.optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

            loss = float(loss.data) * ntokens
            total_loss += loss
            cur_loss += loss
            total_tokens += ntokens
            tokens += ntokens
            sentences += int(src_inputs.size(0))

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                print(datetime.datetime.now(),
                      "Epoch Step: %d Loss: %f Tokens per Sec: %f Sentences per Sec: %f" % (
                          i + 1, cur_loss / tokens, tokens / elapsed, sentences / elapsed))

                if (i + 1) % 5000 == 0:
                    best_valid_loss = self.validate_and_save(best_valid_loss, saving_path, valid_data_iter)
                    if best_valid_loss < 1:
                        bleu = self.eval_bleu(valid_data_iter)
                        print("BLEU:", bleu)

                start, tokens, cur_loss, sentences = time.time(), 0, 0, 0

        print("Total loss in this epoch: %f" % (total_loss / total_tokens))
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save(saving_path + ".latest")

        best_valid_loss = self.validate_and_save(best_valid_loss, saving_path, valid_data_iter)
        if best_valid_loss < 1:
            bleu = self.eval_bleu(valid_data_iter)
            print("BLEU:", bleu)
        return total_loss / total_tokens, best_valid_loss

    def eval_bleu(self, valid_data_iter):
        mt_output = []
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            for batch in valid_data_iter:
                src_inputs = batch["src_texts"].squeeze()
                src_mask = batch["src_pad_mask"].squeeze()
                tgt_inputs = batch["dst_texts"].squeeze()
                tgt_mask = batch["dst_pad_mask"].squeeze()
                outputs = self.generator(device=self.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                         src_mask=src_mask, tgt_mask=tgt_mask)
                for output in outputs:
                    mt_output.append(self.generator.seq2seq_model.text_processor.tokenizer.decode(output.numpy()))

            model.train()
        bleu = sacrebleu.corpus_bleu(mt_output, [self.reference[:len(mt_output)]])
        return bleu.score

    def validate_and_save(self, best_valid_loss, saving_path, valid_data_iter):
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()
        with torch.no_grad():
            total_valid_loss, total_valid_tokens = 0, 0
            for batch in valid_data_iter:
                src_inputs = batch["src_texts"].squeeze()
                src_mask = batch["src_pad_mask"].squeeze()
                tgt_inputs = batch["dst_texts"].squeeze()
                tgt_mask = batch["dst_pad_mask"].squeeze()
                predictions = self.model(device=self.device, src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                         src_mask=src_mask, tgt_mask=tgt_mask, log_softmax=True, flatten=True)

                targets = tgt_inputs[:, 1:].contiguous().view(-1)
                ntokens = targets.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, targets).mean().data * ntokens
                total_valid_loss += float(loss)
                total_valid_tokens += ntokens

            valid_loss = total_valid_loss / total_valid_tokens
            print("Current valid loss", valid_loss)
            if best_valid_loss > float(valid_loss):
                best_valid_loss = float(valid_loss)
                print("saving best valid loss", best_valid_loss)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )

                model_to_save.save(saving_path)
            model.train()
        return best_valid_loss

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        if options.pretrained_path is None:
            lm = LM(text_processor=text_processor, size=options.model_size)
        else:
            lm = LM.load(options.pretrained_path)

        mt_model = AlbertSeq2Seq(lm=lm, sep_encoder_decoder=options.sep_encoder)

        train_data = dataset.MTDataset(batch_pickle_dir=options.train__path,
                                       max_batch_total_capcity=options.total_capacity, max_batch=options.batch,
                                       pad_idx=lm.text_processor.pad_token_id())
        valid_data = dataset.MTDataset(batch_pickle_dir=options.valid__path,
                                       max_batch_total_capcity=options.total_capacity, max_batch=options.batch,
                                       pad_idx=lm.text_processor.pad_token_id())

        pin_memory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
        valid_loader = data_utils.DataLoader(valid_data, batch_size=1, shuffle=False, pin_memory=pin_memory)

        trainer = Trainer(model=mt_model, mask_prob=options.mask_prob,
                          optimizer=Trainer.build_optimizer(lm.encoder, options.learning_rate, options.weight_decay),
                          clip=options.clip, warmup=options.warmup, warmup_steps=options.warmup_steps,
                          fp16=options.fp16, fp16_opt_level=options.fp16_opt_level)

        print("creating referece")
        trainer.reference = []
        for batch in valid_loader:
            tgt_inputs = batch["dst_texts"].squeeze()
            refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs)
            ref = [trainer.generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
            trainer.reference += ref

        best_valid_loss = float("inf")
        for i in range(options.num_epochs):
            print("train epoch", i)
            _, best_valid_loss = trainer.train_epoch(data_iter=loader, valid_data_iter=valid_loader,
                                                     best_valid_loss=best_valid_loss, saving_path=options.model_path)


def get_options():
    global options
    parser = OptionParser()
    parser.add_option("--train", dest="train__path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--valid", dest="valid__path",
                      help="Path to the train data pickle files for large data", metavar="FILE", default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--pretrained", dest="pretrained_path", help="Directory of pretrained model", metavar="FILE",
                      default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=100)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capcity", type="int", default=100)
    parser.add_option("--batch", dest="batch", help="Batch num_tokens", type="int", default=25000)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    parser.add_option("--embed", dest="d_model", help="Embedding of contextual word vectors", type="int", default=768)
    parser.add_option("--lr", dest="learning_rate", help="Learning rate", type="float", default=0.002)
    parser.add_option("--warmup", dest="warmup", help="Warm up rate", type="float", default=0.00001)
    parser.add_option("--steps", dest="warmup_steps", help="Number of warmup steps", type="int", default=125000)
    parser.add_option("--decay", dest="weight_decay", help="Weight decay", type="float", default=0.01)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--layer", dest="num_layers", help="Number of Layers in cross-attention", type="int", default=2)
    parser.add_option("--heads", dest="num_heads", help="Number of attention heads", type="int", default=8)
    parser.add_option("--fp16", action="store_true", dest="fp16", help="use fp16; should be compatible", default=False)
    parser.add_option("--sep", action="store_true", dest="sep_encoder", help="Don't share encoder and decoder",
                      default=False)
    parser.add_option("--size", dest="model_size", help="Model size: 1 (base), 2 (medium), 3 (small)", type="int",
                      default=3)
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
