import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ModifiedResnet(models.ResNet):
    def _forward_impl(self, x):
        batch_size = x.size(0)
        input = x
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        grid_hidden = self.layer4(x)
        grid_hidden = grid_hidden.view(grid_hidden.size(0), grid_hidden.size(1), -1)
        grid_hidden = grid_hidden.permute((0, 2, 1))
        if self.dropout > 0:
            grid_hidden = F.dropout(grid_hidden, p=self.dropout)
        grid_outputs = self.fc(grid_hidden)

        location_indices = torch.stack(batch_size * [torch.tensor([i for i in range(49)])])
        if torch.cuda.is_available():
            location_indices = location_indices.cuda(grid_outputs.get_device())
        location_embedding = self.location_embedding(location_indices)

        out = grid_outputs + location_embedding
        out_norm = self.layer_norm(out)
        if self.dropout > 0:
            out_norm = F.dropout(out_norm, p=self.dropout)
        return out_norm


def init_net(embed_dim: int, dropout: float = 0.1, freeze: bool = False):
    model = models.resnet50(pretrained=True)
    model.__class__ = ModifiedResnet
    model.dropout = dropout
    model.layer_norm = torch.nn.LayerNorm(embed_dim, eps=1e-12)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    current_weight = model.state_dict()["fc.weight"]
    model.fc = torch.nn.Linear(in_features=current_weight.size()[1], out_features=embed_dim, bias=False)
    model.fc.train()

    # Learning embedding of each CNN region.
    model.location_embedding = nn.Embedding(49, embed_dim)
    model.location_embedding.train(True)

    return model
