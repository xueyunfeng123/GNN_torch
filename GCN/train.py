import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj
from    model import GCN
from    config import  args
from    utils import masked_loss, masked_acc


seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)


# load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
print('adj:', adj.shape)
print('features:', features.shape)
print('y:', y_train.shape, y_val.shape, y_test.shape)
print('mask:', train_mask.shape, val_mask.shape, test_mask.shape)

# 讲特征归一化，并转化为稀疏矩阵表示
features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
coordinates_, data_, shape_ = features
# 对邻接阵作自环添加，并且对称归一中心化
supports = preprocess_adj(adj)

print('coordinates_:', coordinates_.shape)  # 每个非零点的坐标
print('data_:', data_.shape)  # 每个非零点的特征值
print('shape_:', shape_)  # 原始特征的shape

device = torch.device('cuda')
train_label = torch.from_numpy(y_train).long().to(device)
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim=1)
train_mask = torch.from_numpy(train_mask.astype(np.int_)).to(device)
val_label = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim=1)
val_mask = torch.from_numpy(val_mask.astype(np.int_)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim=1)
test_mask = torch.from_numpy(test_mask.astype(np.int_)).to(device)

print(train_label.shape)
print(test_label.shape)
print(val_label.shape)

i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse_coo_tensor(i.t(), v, features[2]).to(device)

i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse_coo_tensor(i.t(), v, supports[2]).float().to(device)

print('x :', feature)
print('sp:', support)
num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]


net = GCN(feat_dim, num_classes, num_features_nonzero)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

net.train()
for epoch in range(args.epochs):

    out = net((feature, support))
    out = out[0]
    # 交叉熵损失
    loss = masked_loss(out, train_label, train_mask)
    loss += args.weight_decay * net.l2_loss()

    acc = masked_acc(out, train_label, train_mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:

        print(epoch, loss.item(), acc.item())

net.eval()

out = net((feature, support))
out = out[0]
acc = masked_acc(out, test_label, test_mask)
print('test:', acc.item())