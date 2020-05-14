from typing import List, Optional

import numpy as np
from tokenizers import Encoding
from tokenizers import SentencePieceBPETokenizer


class TextProcessor:
    def __init__(self, tok_model_path: Optional[str] = None):
        self.init_properties()

        if tok_model_path is not None:
            self.tokenizer = SentencePieceBPETokenizer(
                tok_model_path + "/vocab.json",
                tok_model_path + "/merges.txt",
            )

    def init_properties(self):
        self.max_len = 512
        self.languages = ["<ab>", "<ast>", "<bg>", "<cbk_zam>", "<cv>", "<en>", "<frp>", "<got>", "<hy>", "<jam>",
                          "<kn>", "<lbe>", "<mai>", "<mt>", "<nn>", "<pag>", "<pt>", "<sat>", "<so>", "<te>", "<tum>",
                          "<vls>", "<zh>", "<ace>", "<atj>", "<bh>", "<cdo>", "<cy>", "<eo>", "<frr>", "<gu>", "<ia>",
                          "<jbo>", "<ko>", "<lez>", "<map_bms>", "<mwl>", "<no>", "<pam>", "<qu>", "<sc>", "<sq>",
                          "<tet>", "<tw>", "<vo>", "<zh_classical>", "<ady>", "<av>", "<bi>", "<ce>", "<da>", "<es>",
                          "<fur>", "<gv>", "<id>", "<jv>", "<koi>", "<lfn>", "<mdf>", "<my>", "<nov>", "<pap>", "<rm>",
                          "<scn>", "<sr>", "<tg>", "<ty>", "<wa>", "<zh_min_nan>", "<af>", "<ay>", "<bjn>", "<ch>",
                          "<de>", "<et>", "<fy>", "<ha>", "<ie>", "<ka>", "<krc>", "<lg>", "<mg>", "<myv>", "<nrm>",
                          "<pcd>", "<rmy>", "<sco>", "<srn>", "<th>", "<tyv>", "<war>", "<zh_yue>", "<ak>", "<az>",
                          "<bm>", "<chr>", "<din>", "<eu>", "<ga>", "<hak>", "<ig>", "<kaa>", "<ks>", "<li>", "<mhr>",
                          "<mzn>", "<nso>", "<pdc>", "<rn>", "<sd>", "<ss>", "<ti>", "<udm>", "<wo>", "<zu>", "<als>",
                          "<azb>", "<bn>", "<chy>", "<diq>", "<ext>", "<gag>", "<haw>", "<ik>", "<kab>", "<ksh>",
                          "<lij>", "<mi>", "<na>", "<nv>", "<pfl>", "<ro>", "<se>", "<st>", "<tk>", "<ug>", "<wuu>",
                          "<am>", "<ba>", "<bo>", "<ckb>", "<dsb>", "<fa>", "<gan>", "<he>", "<ilo>", "<kbd>", "<ku>",
                          "<lmo>", "<min>", "<nah>", "<ny>", "<pi>", "<roa_rup>", "<sg>", "<stq>", "<tl>", "<uk>",
                          "<xal>", "<an>", "<ban>", "<bpy>", "<co>", "<dty>", "<ff>", "<gd>", "<hi>", "<inh>", "<kbp>",
                          "<kv>", "<ln>", "<mk>", "<nap>", "<oc>", "<pih>", "<roa_tara>", "<sh>", "<su>", "<tn>",
                          "<ur>", "<xh>", "<ang>", "<bar>", "<br>", "<cr>", "<dv>", "<fi>", "<gl>", "<hif>", "<io>",
                          "<kg>", "<kw>", "<lo>", "<ml>", "<nds>", "<olo>", "<pl>", "<ru>", "<si>", "<sv>", "<to>",
                          "<uz>", "<xmf>", "<ar>", "<bat_smg>", "<bs>", "<crh>", "<dz>", "<fiu_vro>", "<glk>", "<hr>",
                          "<is>", "<ki>", "<ky>", "<lrc>", "<mn>", "<nds_nl>", "<om>", "<pms>", "<rue>", "<sk>", "<sw>",
                          "<tpi>", "<ve>", "<yi>", "<arc>", "<bcl>", "<bug>", "<cs>", "<ee>", "<fj>", "<gn>", "<hsb>",
                          "<it>", "<kk>", "<la>", "<lt>", "<mr>", "<ne>", "<or>", "<pnb>", "<rw>", "<sl>", "<szl>",
                          "<tr>", "<vec>", "<yo>", "<arz>", "<be>", "<bxr>", "<csb>", "<el>", "<fo>", "<gom>", "<ht>",
                          "<iu>", "<kl>", "<lad>", "<ltg>", "<mrj>", "<new>", "<os>", "<pnt>", "<sa>", "<sm>", "<ta>",
                          "<ts>", "<vep>", "<za>", "<as>", "<be_x_old>", "<ca>", "<cu>", "<eml>", "<fr>", "<gor>",
                          "<hu>", "<ja>", "<km>", "<lb>", "<lv>", "<ms>", "<nl>", "<pa>", "<ps>", "<sah>", "<sn>",
                          "<tcy>", "<tt>", "<vi>", "<zea>"]
        self.language_set = set(self.languages)
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.unk_token = "<unk>"
        self.sep_token = "</s>"
        self.bos = "<s>"
        self.special_tokens = [self.bos, self.pad_token, self.unk_token, self.mask_token,
                               self.sep_token] + self.languages

    def train_tokenizer(self, paths: List[str], vocab_size: int, to_save_dir: str):
        self.tokenizer = SentencePieceBPETokenizer()
        self.init_properties()
        self.tokenizer.train(files=paths, vocab_size=vocab_size, min_frequency=5, special_tokens=self.special_tokens)
        self.tokenizer.save(directory=to_save_dir)

    def _tokenize(self, line) -> Encoding:
        return self.tokenizer.encode(line)

    def tokenize_one_line(self, line, ignore_middle_eos: bool = False) -> List[int]:
        tokenized = []
        spl = [sen for sen in line.split("</s>") if len(sen.strip()) > 0]
        if spl[0].startswith("<"):
            words = spl[0].strip().split(" ")
            spl[0] = " ".join(words[1:])
            tokenized += [self.token_id(words[0])]

        for sen in spl:
            tokenized += self._tokenize(sen).ids
            if not ignore_middle_eos:
                tokenized += [self.sep_token_id()]
        if ignore_middle_eos:
            tokenized += [self.sep_token_id()]
        return tokenized

    def tokenize_lines(self, line, blind_split: bool = False, split_len: int = 512) -> List[List[int]]:
        """

        :param line:
        :param blind_split: If True, just splits the tokenized data into chunks without considering that every vector
        should start with a first word in sentence.
        :return:
        """
        tokenized = []
        spl = [sen for sen in line.split("</s>") if len(sen.strip()) > 0]
        if spl[0].startswith("<"):
            words = spl[0].strip().split(" ")
            spl[0] = " ".join(words[1:])
            tokenized += [self.token_id(words[0])]

        max_len = 0
        for sen in spl:
            toks = self._tokenize(sen).ids
            tokenized += toks + [self.sep_token_id()]
            max_len = max(max_len, len(toks) + 1)
        if blind_split:
            num_pads = (split_len - (len(tokenized) % split_len))
            pad_arr = [self.pad_token_id()] * num_pads
            tokenized = np.array(tokenized + pad_arr)
            reshaped = tokenized.reshape((-1, split_len))
            return reshaped
        else:
            return self.split_tokenized(tokenized, min(max_len, self.max_len))

    def tokenize(self, lines) -> List[List[int]]:
        lines = [line.strip() for line in lines.strip().split("\n") if len(line.strip()) > 0]
        tokenized = self.tokenizer.encode_batch(lines)
        return [tok.ids for tok in tokenized]

    def pad_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.pad_token)

    def mask_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.mask_token)

    def unk_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.unk_token)

    def bos_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.bos)

    def sep_token_id(self) -> int:
        return self.tokenizer.token_to_id(self.sep_token)

    def token_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id2token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def is_lang(self, id) -> bool:
        return self.tokenizer.id_to_token(id) in self.languages

    def split_tokenized(self, tokenized: List[int], max_length: int = 512) -> List[List[int]]:
        """
        Based on self.max_len, splits very long sequences to smaller ones.
        Here we assume to not have any overlapping sequences.
        If the first token is a language, we add it to every new sequence.
        :return:
        """
        if len(tokenized) <= max_length:
            sequences = [tokenized]
            sequences[-1] = sequences[-1] + (max_length - len(sequences[-1])) * [self.pad_token_id()]
            return sequences

        has_lang = self.is_lang(tokenized[0])
        sequence = tokenized[0:] if has_lang else tokenized

        seq_len = len(sequence)
        sep_id = self.sep_token_id()
        max_len = max_length - 1 if has_lang else max_length

        cur_start = 0
        sequences = []
        built_seq = []
        truncated = False  # Shows if previous sequence is truncated due to its length.
        used_ends = set()
        while cur_start < seq_len:
            if not truncated or not has_lang:
                cur_end = min(seq_len, cur_start + max_len)
            else:
                cur_end = min(seq_len, cur_start + max_len + 1)
            subseq = sequence[cur_start:cur_end]

            built_seq += subseq
            sep_positions = [i for i, id in enumerate(built_seq) if id == sep_id]
            if len(sep_positions) > 0:
                if sep_positions[-1] in used_ends:
                    truncated = True
                else:
                    built_seq = built_seq[:sep_positions[-1] + 1]
                    truncated = False
            else:
                truncated = True

            assert built_seq[-1] == sequence[len(built_seq) - 1]

            if has_lang and len(subseq) < max_len + 1:
                subseq = [tokenized[0]] + subseq

            sequences.append(subseq)

            cur_start = len(built_seq)
            used_ends.add(cur_start - 1)
        if len(sequences[-1]) < max_length:
            sequences[-1] = sequences[-1] + (max_length - len(sequences[-1])) * [self.pad_token_id()]
        assert built_seq[-1] == sequence[len(built_seq) - 1]
        return sequences
