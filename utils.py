from pytorch_lamb.pytorch_lamb import Lamb


def build_optimizer(model, learning_rate, weight_decay):
    return Lamb(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(.9, .999), adam=True)
