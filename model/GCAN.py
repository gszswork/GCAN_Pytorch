import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 0. Co-Attention Module [Done]
class CoAttentionNetwork(nn.Module):

    def __init__(self, hidden_v_dim, hidden_q_dim, co_attention_dim):
        super(CoAttentionNetwork, self).__init__()

        self.hidden_v_dim = hidden_v_dim
        self.hidden_q_dim = hidden_q_dim
        self.co_attention_dim = co_attention_dim

        self.W_b = nn.Parameter(torch.randn(self.hidden_q_dim, self.hidden_v_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_v_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_q_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))

    def forward(self, V, Q):
        V = V.permute([0, 2, 1]) # We have to permute V to fit the shape for matmul().
        # print('W_b:', self.W_b.shape, ', V:', V.shape, ', Q:', Q.shape)
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
        # (batch_size, hidden_v_dim)
        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
        # (batch_size, hidden_q_dim)
        q = torch.squeeze(torch.matmul(a_q, Q))

        return a_v, a_q, v, q


# The GCAN model

# 1. GCN part [Not Yet]
#  TODO: generate edge weight based on cosine similarity.
class GCN(torch.nn.Module):
    def __init__(self, in_feat_dim=12, hid_feat_dim=256, out_feat_dim=128):
        super(GCN, self).__init__()
        self.GCN_l1 = GCNConv(in_feat_dim, hid_feat_dim)
        self.GCN_l2 = GCNConv(hid_feat_dim, out_feat_dim)

    def cal_edge_index(self, user_data):
        # user_data, shape [25, 12]
        # Give the user_batch generate the adjacent matrix for GCN
        # (1) This graph should be fully-connected
        source_node = []
        target_node = []

        len_user_data = user_data.shape[0]
        for i in range(len_user_data):
            for j in range(len_user_data):
                if i != j:
                    source_node.append(i)
                    target_node.append(j)

        return torch.LongTensor([source_node, target_node])

    def forward(self, user_batch):
        # user_batch, shape [Batch, 25, 12]

        batch_size = user_batch.shape[0]
        edge_index = self.cal_edge_index(user_batch)

        gcn_outputs = []
        for i in range(batch_size):
            # We need to take user_batch element 1 by 1 cause GCNConv doesnt support batch training.

            gcn_output1 = self.GCN_l1(user_batch[i], edge_index)
            gcn_output2 = self.GCN_l2(gcn_output1, edge_index)

            gcn_outputs.append(gcn_output2.unsqueeze(dim=0))

        return torch.cat(gcn_outputs, dim=0)


# 2. Source Encoder [Done]
class Source_Encoder(torch.nn.Module):
    def __init__(self, in_dim=3347, hid_dim=32):
        super(Source_Encoder, self).__init__()
        self.gru_encoder = torch.nn.GRU(input_size=in_dim, hidden_size=hid_dim, batch_first=True)

    def forward(self, source_text):
        # source_text: one hot, size [B, length of words, vec_dim]
        return self.gru_encoder(source_text)[0]


# 3. CNN Propagation Encoder [Done]
# Note: I double-checked their paper and official code, their 1-D CNN is not what we usually handle with.
# It's dot production, flatten and then Linear.
class CNN_Encoder(torch.nn.Module):
    def __init__(self, filter_size, input_dim, kernel_size):
        super(CNN_Encoder, self).__init__()
        self.filter_size = filter_size
        self.input_dim = input_dim
        self.kernel_size = kernel_size

        self.filter = torch.nn.Parameter(torch.rand(self.filter_size, self.input_dim))

        self.W_k = torch.nn.Parameter(torch.rand(self.filter_size * input_dim, self.kernel_size))
        self.b_k = torch.nn.Parameter(torch.rand(self.kernel_size))

    def forward(self, user_batch):
        # user_batch: (B, len=25 dim=12)
        batch_size, len_user_batch, _ = user_batch.shape

        res = []
        for i in range(0, len_user_batch - self.filter_size):
            sub_user_batch = user_batch[:, i:i + self.filter_size, :]  # shape: [B, 3, 12]
            multiply_res = torch.mul(self.filter, sub_user_batch)  # shape: [B, 3, 12]
            res.append(torch.reshape(multiply_res, (batch_size, 1, -1)))  # shape: [B, 1, 36]

        res = torch.cat(res, dim=1)  # shape: [B, len_user_batch-filter_size +1, 36]
        # print(res.shape)
        return F.relu(torch.matmul(res, self.W_k) + self.b_k)


# 4. GRU Propagation Encoder [Done]
class GRU_Encoder(torch.nn.Module):
    def __init__(self, in_dim=12, hid_dim=64):
        super(GRU_Encoder, self).__init__()
        self.gru_encoder = torch.nn.GRU(input_size=in_dim, hidden_size=hid_dim)
        pass

    def forward(self, user_batch):
        # user_batch: UsrInfo, size [B, length_of_usrs=25, dim=12]
        user_batch = user_batch.permute([1, 0, 2])
        return self.gru_encoder(user_batch)


# 5. Integrator
class GCAN(torch.nn.Module):
    def __init__(self,
                 gcn_in_dim,
                 gcn_hid_dim,
                 gcn_out_dim,
                 source_gru_in_dim,
                 source_gru_hid_dim,
                 cnn_filter_size,
                 cnn_in_dim,
                 cnn_kernel_size,
                 propagation_gru_in_dim,
                 propagation_gru_hid_dim,
                 source_gcn_coattn_dim,
                 source_cnn_coattn_dim,
                 fc_in_dim,
                 fc_out_dim
                 ):
        super(GCAN, self).__init__()
        self.gcn_module = GCN(gcn_in_dim, gcn_hid_dim, gcn_out_dim)
        self.source_gru = Source_Encoder(source_gru_in_dim, source_gru_hid_dim)
        self.cnn_module = CNN_Encoder(cnn_filter_size, cnn_in_dim, cnn_kernel_size)
        self.user_gru = GRU_Encoder(propagation_gru_in_dim, propagation_gru_hid_dim)
        # source_gcn_coattn, V is the source, Q is the gcn
        self.source_gcn_coattn = CoAttentionNetwork(source_gru_hid_dim, gcn_out_dim, source_gcn_coattn_dim)
        # source_cnn_coattn, V is the source, Q is the cnn
        self.source_cnn_coattn = CoAttentionNetwork(source_gru_hid_dim, propagation_gru_hid_dim, source_cnn_coattn_dim)
        self.fc_layer = torch.nn.Linear(fc_in_dim, fc_out_dim)

    def forward(self, source_text, user_batch):
        # The shape of user_batch: (batch_size, length_of_user_under_post, user_feat_dim)
        gcn_output = self.gcn_module(user_batch)
        source_output = self.source_gru(source_text)
        cnn_output = self.cnn_module(user_batch)
        gru_output = self.user_gru(user_batch)
        co_attn_output1 = self.source_gcn_coattn(source_output, gcn_output)
        co_attn_output2 = self.source_cnn_coattn(source_output, cnn_output)

        #whole_rep = torch.concat(co_attn_output1, co_attn_output2, pool(gru_output1))
        return None

if __name__ == '__main__':
    # Play with our model with a mini-demo
    source_batch = torch.rand(4, 40, 3347)
    user_batch = torch.rand(4, 25, 12)
    gcan = GCAN(gcn_in_dim=12,
                gcn_hid_dim=64,
                gcn_out_dim=128,
                source_gru_in_dim=3347,
                source_gru_hid_dim=32,
                cnn_filter_size=3,
                cnn_in_dim=12,
                cnn_kernel_size=32,
                propagation_gru_in_dim=12,
                propagation_gru_hid_dim=32,
                source_gcn_coattn_dim=64,
                source_cnn_coattn_dim=64,
                fc_in_dim=128 + 32 + 32 + 32,
                fc_out_dim=2
                )
    gcan(source_batch, user_batch)