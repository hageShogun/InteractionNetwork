import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from physics_engine import gen
from interaction_network.interaction_network import InteractionNetwork


USE_CUDA = torch.cuda.is_available()


def get_batch(data, batch_size):
    rand_idx = [random.randint(0, len(data) - 2) for _ in range(batch_size)]
    label_idx = [idx + 1 for idx in rand_idx]

    batch_data = data[rand_idx]
    label_data = data[label_idx]

    objects = batch_data[:,:,:5]

    receiver_relations = np.zeros((batch_size, n_objects, n_relations), dtype=float)
    sender_relations = np.zeros((batch_size, n_objects, n_relations), dtype=float)

    cnt = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if(i != j):
                receiver_relations[:, i, cnt] = 1.0
                sender_relations[:, j, cnt] = 1.0
                cnt += 1

    # There is no relation info in solar system task, just fill with zeros
    relation_info = np.zeros((batch_size, n_relations, relation_dim))
    target = label_data[:,:,3:]

    objects = Variable(torch.FloatTensor(objects))
    sender_relations = Variable(torch.FloatTensor(sender_relations))
    receiver_relations = Variable(torch.FloatTensor(receiver_relations))
    relation_info = Variable(torch.FloatTensor(relation_info))
    target = Variable(torch.FloatTensor(target))
    target.contiguous()
    target = target.view(-1, 2)

    if USE_CUDA:
        objects = objects.cuda()
        sender_relations = sender_relations.cuda()
        receiver_relations = receiver_relations.cuda()
        relation_info = relation_info.cuda()
        target = target.cuda()

    return objects, sender_relations, receiver_relations, relation_info, target


# Data generation
n_objects = 6  # number of planets(nodes)
object_dim = 5  # features: mass, pos_x, pos_y, v_x, v_y
n_relations = n_objects * (n_objects - 1)  # number of edges in fully connected graph
relation_dim = 1
effect_dim = 100  # effect's vector size

data = gen(n_objects, True)


interaction_network = InteractionNetwork(n_objects, object_dim, n_relations, relation_dim, effect_dim)

if USE_CUDA:
    interaction_network = interaction_network.cuda()

optimizer = optim.Adam(interaction_network.parameters())
criterion = nn.MSELoss()


n_epoch = 100
batches_per_epoch = 100

losses = []
for epoch in tqdm(range(n_epoch)):
    for _ in range(batches_per_epoch):
        objects, sender_relations, receiver_relations, relation_info, target = get_batch(data, 30)
        predicted = interaction_network(objects, sender_relations, receiver_relations, relation_info)
        loss = criterion(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(np.sqrt(loss.data[0]))

plt.figure(figsize=(20,5))
plt.subplot(131)
plt.title('Epoch %s RMS Error %s' % (epoch, np.sqrt(np.mean(losses[-100:]))))
plt.plot(losses)
plt.show()
