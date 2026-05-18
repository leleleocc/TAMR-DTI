import torch
import torch.nn as nn


class DCNN(nn.Module):
    def __init__(self, batch_size, sentence_length, num_filters, embed_size, top_k, k1):
        super(DCNN, self).__init__()
        self.batch_size = batch_size
        self.sentence_length = sentence_length
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.top_k = top_k
        self.k1 = k1

        # Initialize weights and biases
        self.W1 = nn.Parameter(torch.randn(num_filters[0], embed_size,5 ), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(num_filters[1], embed_size,5 ), requires_grad=True)
        self.b1 = nn.Parameter(torch.randn(num_filters[0]), requires_grad=True)
        self.b2 = nn.Parameter(torch.randn(num_filters[1]), requires_grad=True)
        self.Wh = nn.Parameter(torch.randn(top_k * 100 * 14 // 4, 256), requires_grad=True)
        self.bh = nn.Parameter(torch.randn(256), requires_grad=True)
        self.Wo = nn.Parameter(torch.randn(256, 1), requires_grad=True)

    def per_dim_conv_k_max_pooling_layer(self, x, w, b, k):
        # x shape: [batch_size, sentence_length, embed_size]
        x_unstacked = x.permute(0, 2, 1)  # [batch_size, embed_size, sentence_length]
        convs = []
        for i in range(self.embed_size):
            conv = nn.functional.relu(
                nn.functional.conv1d(x_unstacked[:, i:i + 1, :], w[i:i + 1], stride=1, padding="same") + b[i])
            # conv shape: [batch_size, num_filters[0], sentence_length]
            top_k_values, _ = torch.topk(conv, k, dim=2)
            convs.append(top_k_values)
        conv = torch.stack(convs, dim=2)  # [batch_size, k, embed_size, num_filters[0]]
        return conv

    def per_dim_conv_layer(self, x, w, b):
        x_unstacked = x.permute(0, 2,1)  # [batch_size, embed_size, sentence_length]
        convs = []
        for i in range(len(x_unstacked)):
            conv = nn.functional.relu(
                nn.functional.conv1d(x_unstacked[:, i:i + 1, :], w[i:i + 1], stride=1, padding="same") + b[i])
            convs.append(conv)
        conv = torch.stack(convs, dim=1)  # [batch_size, k1+ws-1, embed_size, num_filters[1]]
        return conv

    def fold_k_max_pooling(self, x, k):
        x_unstacked = x.permute(0, 2, 1, 3)  # [batch_size, num_filters, k1, embed_size]
        out = []
        for i in range(0, x_unstacked.size(1), 2):  # fold every two
            fold = x_unstacked[:, i:i + 2, :, :]  # [batch_size, 2, k1, embed_size]
            fold_sum = fold.sum(dim=1)  # [batch_size, k1, embed_size]
            top_k_values, _ = torch.topk(fold_sum, k, dim=2)  # [batch_size, k2, top_k]
            out.append(top_k_values)
        fold = torch.stack(out, dim=2)  # [batch_size, k2, num_filters, top_k]
        return fold

    def full_connect_layer(self, x, w, b, wo, dropout_keep_prob):
        h = torch.tanh(torch.matmul(x, w) + b)
        h = nn.functional.dropout(h, p=1 - dropout_keep_prob, training=self.training)
        o = torch.matmul(h, wo)
        return o

    def forward(self, sent, dropout_keep_prob):
        conv1 = self.per_dim_conv_layer(sent, self.W1, self.b1)
        conv1 = self.fold_k_max_pooling(conv1, self.k1)
        conv2 = self.per_dim_conv_layer(conv1, self.W2, self.b2)
        fold = self.fold_k_max_pooling(conv2, self.top_k)
        fold_flatten = fold.view(-1, self.top_k * 100 * 14 // 4)  # Flatten for fully connected layer
        out = self.full_connect_layer(fold_flatten, self.Wh, self.bh, self.Wo, dropout_keep_prob)
        return out