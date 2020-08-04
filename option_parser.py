from optparse import OptionParser


def get_lm_option_parser():
    parser = OptionParser()
    parser.add_option("--train", dest="train_path", help="Path to the train data pickle files for large data",
                      metavar="FILE", default=None)
    parser.add_option("--dev", dest="dev_path", help="Path to the dev data pickle files for large data", metavar="FILE",
                      default=None)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--cache_size", dest="cache_size", help="Number of blocks in cache", type="int", default=300)
    parser.add_option("--model", dest="model_path", help="Directory path to save the best model", metavar="FILE",
                      default=None)
    parser.add_option("--pretrained", dest="pretrained_path", help="Directory of pretrained model", metavar="FILE",
                      default=None)
    parser.add_option("--epoch", dest="num_epochs", help="Number of training epochs", type="int", default=100)
    parser.add_option("--clip", dest="clip", help="For gradient clipping", type="int", default=1)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=6000)
    parser.add_option("--mask", dest="mask_prob", help="Random masking probability", type="float", default=0.15)
    parser.add_option("--lr", dest="learning_rate", help="Learning rate", type="float", default=0.0001)
    parser.add_option("--warmup", dest="warmup", help="Number of warmup steps", type="int", default=12500)
    parser.add_option("--step", dest="step", help="Number of training steps", type="int", default=125000)
    parser.add_option("--max_grad_norm", dest="max_grad_norm", help="Max grad norm", type="float", default=1.0)
    parser.add_option("--cont", action="store_true", dest="continue_train",
                      help="Continue training from pretrained model", default=False)
    parser.add_option("--dropout", dest="dropout", help="Dropout probability", type="float", default=0.1)
    parser.add_option("--dff", dest="d_ff", help="Position-wise feed-forward dimensions", type="int", default=2048)
    parser.add_option("--reformer", action="store_true", dest="reformer", help="Use Reformer instead of BERT",
                      default=False)
    parser.add_option("--enc", dest="encoder_layer", help="# encoder layers", type="int", default=6)
    parser.add_option("--embed", dest="embed_dim", help="Embedding dimension", type="int", default=768)
    parser.add_option("--intermediate", dest="intermediate_layer_dim", type="int", default=3072)
    return parser


def get_img_options_parser():
    parser = get_lm_option_parser()
    parser.add_option("--capacity", dest="total_capacity", help="Batch capacity", type="int", default=600)
    parser.add_option("--lm", dest="lm_path", help="LM pretrained model", metavar="FILE", default=None)
    parser.add_option("--dict", dest="dict_path", help="External lexical dictionary", metavar="FILE", default=None)
    parser.add_option("--beam", dest="beam_width", help="Beam width", type="int", default=5)
    parser.add_option("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type="float", default=1.3)
    parser.add_option("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type="int", default=5)
    parser.add_option("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type="float", default=0.8)
    parser.add_option("--max_seq_len", dest="max_seq_len", help="Max sequence length", type="int", default=175)
    parser.add_option("--ldec", action="store_true", dest="lang_decoder", help="Lang-specific decoder", default=False)
    parser.add_option("--nll", action="store_true", dest="nll_loss", help="Use NLL loss instead of smoothed NLL loss",
                      default=False)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    parser.set_default("batch", 20000)
    parser.add_option("--dev_mt", dest="mt_dev_path",
                      help="Path to the MT dev data pickle files (SHOULD NOT BE USED IN UNSUPERVISED SETTING)",
                      metavar="FILE", default=None)
    parser.add_option("--train_mt", dest="mt_train_path",
                      help="Path to the MT train data pickle files (SHOULD NOT BE USED IN PURELY UNSUPERVISED SETTING)",
                      metavar="FILE", default=None)
    parser.add_option("--fstep", dest="finetune_step", help="Number of finetuneing steps", type="int", default=125000)
    parser.set_default("mask_prob", 0.5)
    parser.add_option("--mass_train", dest="mass_train_path", metavar="FILE", default=None)
    parser.add_option("--image", dest="image_dir", help="Path to the image files", metavar="FILE", default=None)
    parser.add_option("--img_capacity", dest="img_capacity", help="Batch capacity", type="int", default=50)
    parser.add_option("--max-image", dest="max_image", help="Maximum number of images in batch", type="int", default=32)
    parser.add_option("--img-depth", dest="resnet_depth", help="1 (18), 2 (34), 3 (50), 4 (101), 5 (152)", type="int",
                      default=1)
    parser.add_option("--langs", dest="bt_langs",
                      help="Languages for back-translation (should be two, sepearated by comma)", type="str",
                      default="")
    parser.add_option("--mmode", dest="mm_mode", help="Option: mixed, masked, contrastive", type="str", default="mixed")
    parser.add_option("--dec", dest="decoder_layer", help="# decoder layers", type="int", default=6)
    parser.add_option("--ignore-mt-mass", action="store_true", dest="ignore_mt_mass",
                      help="Ignore MT data in backtranslation loss of MASS model", default=False)
    parser.add_option("--tie", action="store_true", dest="tie_embed", help="Tie encoder and decoder embeddings",
                      default=False)
    return parser
