import torch
import torch.nn as nn

from interaction_network.relational_model import RelationalModel
from interaction_network.object_model import ObjectModel


class InteractionNetwork(nn.Module):
    def __init__(self, n_objects, object_dim, n_relations, relation_dim, effect_dim):
        super(InteractionNetwork, self).__init__()

        self.relational_model = RelationalModel(2*object_dim + relation_dim, effect_dim, 150)
        self.object_model = ObjectModel(object_dim + effect_dim, 100)

    def forward(self, objects, sender_relations, receiver_relations, relation_info):
        senders = sender_relations.permute(0, 2, 1).bmm(objects)
        receivers = receiver_relations.permute(0, 2, 1).bmm(objects)
        effects = self.relational_model(torch.cat([senders, receivers, relation_info], 2))
        effect_receivers = receiver_relations.bmm(effects)
        predicted = self.object_model(torch.cat([objects, effect_receivers], 2))
        return predicted
