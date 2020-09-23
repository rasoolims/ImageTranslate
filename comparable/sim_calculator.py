from optparse import OptionParser

import torch
import torch.nn as nn
from apex import amp
from torch.nn.utils.rnn import pad_sequence


def get_option_parser():
    parser = OptionParser()
    parser.add_option("--src", dest="src_file", metavar="FILE", default=None)
    parser.add_option("--dst", dest="dst_file", metavar="FILE", default=None)
    parser.add_option("--src-embed", dest="src_embed", metavar="FILE", default=None)
    parser.add_option("--dst-embed", dest="dst_embed", metavar="FILE", default=None)
    parser.add_option("--output", dest="output_file", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=40000)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    return parser


class SimModel(nn.Module):
    def __init__(self, src_vectors, dst_vectors):
        super(SimModel, self).__init__()
        self.src_embed = nn.Embedding(src_vectors.size(0), src_vectors.size(1), _weight=src_vectors)
        self.dst_embed = nn.Embedding(dst_vectors.size(0), dst_vectors.size(1), _weight=dst_vectors)

    def forward(self, src_batch, dst_batch):
        src_embed = self.src_embed(src_batch)
        dst_embed = self.dst_embed(dst_batch)

        src_pad = (src_batch == 0).unsqueeze(-1).float()
        dst_pad = (dst_batch == 0).unsqueeze(-1).float()

        mm = torch.bmm(src_embed, dst_embed.transpose(1, 2))
        pad_mm = (torch.bmm(src_pad, dst_pad.transpose(1, 2)) == 1)
        mm[pad_mm].fill_(-10000.0)

        max_cos = torch.max(mm, dim=-1)[0]
        avg_cos = torch.div(torch.sum(max_cos, dim=-1), int(max_cos.size(-1)))
        return avg_cos


get_id = lambda x, dct: dct[x] if x in dct else (dct[x.lower()] if x.lower() in dct else None)
get_ids = lambda line, dct: list(filter(lambda x: x is not None, map(lambda x: get_id(x, dct), line.split(" "))))


def build_batches(src_file, dst_file, src_embed_dict, dst_embed_dict, batch=40000):
    current_src_batch, current_dst_batch, num_tok = [], [], 0
    with open(src_file, "r") as src_r, open(dst_file, "r") as dst_r:
        for src_line, dst_line in zip(src_r, dst_r):
            src_ids = torch.LongTensor(get_ids(src_line.strip(), src_embed_dict))
            dst_ids = torch.LongTensor(get_ids(dst_line.strip(), dst_embed_dict))
            current_src_batch.append(src_ids)
            current_dst_batch.append(dst_ids)
            num_tok += len(src_ids) + len(dst_line)
            if num_tok >= batch:
                src_batch = pad_sequence(current_src_batch, batch_first=True, padding_value=0)
                dst_batch = pad_sequence(current_dst_batch, batch_first=True, padding_value=0)
                yield src_batch, dst_batch
                current_src_batch, current_dst_batch, num_tok = [], [], 0
    if num_tok > 0:
        src_batch = pad_sequence(current_src_batch, batch_first=True, padding_value=0)
        dst_batch = pad_sequence(current_dst_batch, batch_first=True, padding_value=0)
        yield src_batch, dst_batch


if __name__ == "__main__":
    parser = get_option_parser()
    (options, args) = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading src embedding")
    src_vectors = []
    src_embed_dict = {}
    vec_length = 150
    with open(options.src_embed, "r") as r:
        for line in r:
            spl = line.strip().split(" ")
            if len(spl) < 3: continue
            src_vectors.append(torch.Tensor(list(map(lambda x: float(x), spl[1:]))))
            vec_length = len(src_vectors[-1])
            src_embed_dict[spl[0]] = len(src_embed_dict) + 1
    src_vectors.insert(0, torch.Tensor([1e-4] * vec_length))
    src_vectors = torch.stack(src_vectors)
    src_norm = torch.norm(src_vectors, dim=-1, p=2).unsqueeze(-1) + 1e-4
    src_embed = torch.div(src_vectors, src_norm)

    print("Reading dst embedding")
    dst_vectors = []
    dst_embed_dict = {}
    with open(options.dst_embed, "r") as r:
        for line in r:
            spl = line.strip().split(" ")
            if len(spl) < 3: continue
            dst_vectors.append(torch.Tensor(list(map(lambda x: float(x), spl[1:]))))
            dst_embed_dict[spl[0]] = len(dst_embed_dict) + 1

    dst_vectors.insert(0, torch.Tensor([1e-4] * vec_length))
    dst_vectors = torch.stack(dst_vectors)
    dst_norm = torch.norm(dst_vectors, dim=-1, p=2).unsqueeze(-1) + 1e-4
    dst_vectors = torch.div(dst_vectors, dst_norm)
    sim_model = SimModel(src_vectors=src_vectors, dst_vectors=dst_vectors)

    if options.fp16:
        sim_model.to(device)
        sim_model = amp.initialize(sim_model, opt_level="O2")

    with torch.no_grad(), open(options.output_file, "w") as ow:
        for i, (src_batch, dst_batch) in enumerate(
                build_batches(options.src_file, options.dst_file, src_embed_dict, dst_embed_dict, options.batch)):
            src_batch = src_batch.to(device)
            dst_batch = dst_batch.to(device)
            sims = sim_model(src_batch, dst_batch)
            sims_txt = "\n".join(list(map(lambda x: str(float(x)), sims)))
            ow.write(sims_txt)
            ow.write("\n")
            print(i, end="\r")
    print("\nDone!")
