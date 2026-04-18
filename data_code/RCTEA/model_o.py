#the general model.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())

class icl_loss(nn.Module):

    def __init__\
    (self, tau=0.05, ab_weight=0.5, n_view=2, intra_weight=1.0, inversion=False, replay=False, neg_cross_kg=False):
        super(icl_loss, self).__init__()
        self.tau = tau
        self.sim = cosine_sim
        self.weight = ab_weight  # the factor of a->b and b<-a
        self.n_view = n_view
        self.intra_weight = intra_weight  # the factor of aa and bb
        self.inversion = inversion
        self.replay = replay
        self.neg_cross_kg = neg_cross_kg

    def softXEnt(self, target, logits, replay=False, neg_cross_kg=False):
        # torch.Size([2239, 4478])

        logprobs = F.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        if replay:
            logits = logits
            idx = torch.arange(start=0, end=logprobs.shape[0], dtype=torch.int64).cuda()
            stg_neg = logits.argmax(dim=1)
            new_value = torch.zeros(logprobs.shape[0]).cuda()
            index = (
                idx,
                stg_neg,
            )
            logits = logits.index_put(index, new_value)
            stg_neg_2 = logits.argmax(dim=1)
            tmp = idx.eq_(stg_neg)
            neg_idx = stg_neg - stg_neg * tmp + stg_neg_2 * tmp
            return loss, neg_idx

        return loss

    # train_links[:, 0]: shape: (2239,)
    # array([11303,  2910,  2072, ..., 10504, 13555,  8416], dtype=int32)

    def forward(self, emb, train_links, neg_l=None, neg_r=None, norm=True):
        if norm:
            emb = F.normalize(emb, dim=1)
        num_ent = emb.shape[0]
        # Get (normalized) hidden1 and hidden2.
        zis = emb[train_links[:, 0]]
        zjs = emb[train_links[:, 1]]

        temperature = self.tau
        alpha = self.weight
        # 2
        n_view = self.n_view
        LARGE_NUM = 1e9
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]
        hidden1_large = hidden1
        hidden2_large = hidden2

        if neg_l is None:
            num_classes = batch_size * n_view
        else:
            num_classes = batch_size * n_view + neg_l.shape[0]
            num_classes_2 = batch_size * n_view + neg_r.shape[0]

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes).float()
        labels = labels.cuda()
        if neg_l is not None:
            labels_2 = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=num_classes_2).float()
            labels_2 = labels_2.cuda()

        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        masks = masks.cuda().float()
        logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM

        logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM

        if neg_l is not None:
            zins = emb[neg_l]
            zjns = emb[neg_r]
            logits_ana = torch.matmul(hidden1, torch.transpose(zins, 0, 1)) / temperature
            logits_bnb = torch.matmul(hidden2, torch.transpose(zjns, 0, 1)) / temperature

        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature

        # logits_a = torch.cat([logits_ab, self.intra_weight*logits_aa], dim=1)
        # logits_b = torch.cat([logits_ba, self.intra_weight*logits_bb], dim=1)
        if self.inversion:
            logits_a = torch.cat([logits_ab, logits_bb], dim=1)
            logits_b = torch.cat([logits_ba, logits_aa], dim=1)
        else:
            if neg_l is None:
                logits_a = torch.cat([logits_ab, logits_aa], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb], dim=1)
            else:
                logits_a = torch.cat([logits_ab, logits_aa, logits_ana], dim=1)
                logits_b = torch.cat([logits_ba, logits_bb, logits_bnb], dim=1)

        if self.replay:
            loss_a, a_neg_idx = self.softXEnt(labels, logits_a, replay=True, neg_cross_kg=self.neg_cross_kg)
            if neg_l is not None:
                loss_b, b_neg_idx = self.softXEnt(labels_2, logits_b, replay=True, neg_cross_kg=self.neg_cross_kg)
                #
                a_ea_cand = torch.cat([train_links[:, 1], train_links[:, 0], neg_l]).cuda()
                b_ea_cand = torch.cat([train_links[:, 0], train_links[:, 1], neg_r]).cuda()
            else:
                loss_b, b_neg_idx = self.softXEnt(labels, logits_b, replay=True, neg_cross_kg=self.neg_cross_kg)
                a_ea_cand = torch.cat([train_links[:, 1], train_links[:, 0]]).cuda()
                b_ea_cand = torch.cat([train_links[:, 0], train_links[:, 1]]).cuda()

            a_neg = a_ea_cand[a_neg_idx]
            b_neg = b_ea_cand[b_neg_idx]
            return alpha * loss_a + (1 - alpha) * loss_b, a_neg, b_neg

        else:
            loss_a = self.softXEnt(labels, logits_a)
            loss_b = self.softXEnt(labels, logits_b)
            return alpha * loss_a + (1 - alpha) * loss_b

class WeightMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WeightMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs weights in [0, 1]
        )

    def forward(self, input_array):
        # Concatenate inputs
        #combined_input = torch.cat((x1, x2), dim=-1)
        weight = self.mlp(input_array)
        return weight

class Embedding_init(nn.Module):
    @staticmethod
    def init_emb(row, col):
        w = torch.empty(row, col)
        torch.nn.init.normal_(w)
        w = torch.nn.functional.normalize(w)
        entities_emb = nn.Parameter(w)
        return entities_emb

class OverAll(nn.Module):
    def __init__(self, node_size, node_hidden, time_hidden,
                 rel_size, rel_hidden,
                 time_size,
                 time_int_size,
                 triple_size,
                 rel_matrix, rel_val,
                 ent_matrix, ent_val,
                 time_matrix, time_val, out_dim,
                 time_int_matrix, time_int_val, args,
                 dropout_rate=0, depth=2, dropout_time=0.5,
                 device='cpu'
                 ):
        super(OverAll, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_time = dropout_time
        self.args = args
        self.return_loss = True
        self.rel_size = rel_size

        self.criterion_cl_joint = \
            icl_loss(tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2, \
                     replay=self.args.replay, neg_cross_kg=self.args.neg_cross_kg)

        # new adding
        # rel_or_time in GraphAttention.forward

        self.e_encoder = GraphAttention(node_size, rel_size, triple_size, depth=depth, device=device,
                                        dim=node_hidden)
        self.r_encoder = GraphAttention(node_size, rel_size, triple_size, depth=depth, device=device,
                                        dim=node_hidden)
        self.t_encoder = GraphAttention(node_size, rel_size, triple_size, depth=depth, device=device,
                                         dim=time_hidden)

        self.ent_adj = self.get_spares_matrix_by_index_value_\
        (ent_matrix, ent_val, (node_size, node_size))
        self.rel_adj = self.get_spares_matrix_by_index_value_\
        (rel_matrix, rel_val, (node_size, rel_size))
        self.time_adj = self.get_spares_matrix_by_index_value_\
        (time_matrix, time_val, (node_size, time_size))
        
        self.time_int_adj = self.get_spares_matrix_by_index_value_\
        (time_int_matrix, time_int_val, (node_size, time_int_size))
        #self.time_adj = self.get_spares_matrix_by_index(time_matrix, (node_size, time_size))
        #self.time_adj = self.get_spares_matrix_by_index\
        #(time_int_matrix, (node_size, time_int_size))

        self.ent_emb = self.init_emb(node_size, node_hidden)
        self.rel_emb = self.init_emb(rel_size, node_hidden)
        self.time_emb = self.init_emb(time_size, time_hidden)
        self.time_int_emb = self.init_emb(time_int_size, time_hidden)

        self.weight_mlp = WeightMLP(4, out_dim)
        self.use_mlp = False

        self.try_emb = self.init_emb(1, node_hidden)
        self.device = device
        self.ent_adj, self.rel_adj, self.time_adj, self.time_int_adj = \
            map(lambda x: x.to(device), [self.ent_adj, self.rel_adj, self.time_adj, self.time_int_adj])

    # get prepared
    @staticmethod
    def get_spares_matrix_by_index(index, size):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        # dim ??
        return torch.sparse.softmax(adj, dim=1)
    
    @staticmethod
    def get_spares_matrix_by_index_value(index, value, size):
        index = torch.LongTensor(index)
        value = torch.tensor(value, dtype=torch.float)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), 
                                       value, size,)
        return torch.sparse.softmax(adj, dim=1)

    @staticmethod
    def get_spares_matrix_by_index_value_(index, value, size):
        index = torch.LongTensor(index)
        value = torch.tensor(value, dtype=torch.float)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), 
                                       value, size)
        return adj

    @staticmethod
    def init_emb(*size):
        entities_emb = nn.Parameter(torch.randn(size))
        torch.nn.init.xavier_normal_(entities_emb)
        return entities_emb
    
    def set_att_gate(self, att, gate):
        self.e_encoder.att = att
        self.r_encoder.att = att
        self.t_encoder.att = att
        self.e_encoder.gate = gate
        self.r_encoder.gate = gate
        self.t_encoder.gate = gate
    
    def set_att_gate_rel(self, att, gate):
        self.e_encoder.att = att
        self.r_encoder.att = att
        
        self.e_encoder.gate = gate
        self.r_encoder.gate = gate
    
    def set_att_gate_time(self, att, gate):
        self.t_encoder.att = att
        self.t_encoder.gate = gate

    def forward(self, inputs):
        # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_pairs]
        ent_feature = torch.matmul(self.ent_adj, self.ent_emb)
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb)
        time_feature = torch.matmul(self.time_adj, self.time_emb)
        time_int_feature = torch.matmul(self.time_int_adj, self.time_int_emb)

        adj_input = inputs[0]
        batch = inputs[1]
        rel_ref = (self.rel_emb[-1] + self.rel_emb[int(self.rel_size/2-1)]) / 2

        opt0 = [self.rel_emb, adj_input, self.ent_emb[-1]]
        opt1 = [self.rel_emb, adj_input, rel_ref]
        opt2 = [self.time_emb, adj_input, self.time_emb[0]]
        opt3 = [self.time_emb, adj_input, self.time_int_emb[2542]]

        # ref_emb = \
        # torch.cat([self.ent_emb[-1], self.ent_emb[-1], self.ent_emb[-1],\
        #            rel_ref, rel_ref, rel_ref,\
        #            self.time_emb[0], self.time_emb[0], self.time_emb[0],\
        #            self.time_int_emb[2542], self.time_int_emb[2542], self.time_int_emb[2542]])
        
        ref_emb_e = torch.cat([self.ent_emb[-1], self.ent_emb[-1], self.ent_emb[-1]])
        ref_emb_r = torch.cat([rel_ref, rel_ref, rel_ref])
        ref_emb_t = torch.cat([self.time_emb[0], self.time_emb[0], self.time_emb[0]])
        ref_emb_it = \
            torch.cat([self.time_int_emb[2542], self.time_int_emb[2542], self.time_int_emb[2542]])

        #opt2 = [self.time_emb, adj_input]
        # attention opt_1 or 2
        out_feature_ent = self.e_encoder([ent_feature] + opt0)
        out_feature_rel = self.r_encoder([rel_feature] + opt1)
        out_feature_time = self.t_encoder([time_feature] + opt2)
        out_feature_int_time = self.t_encoder([time_int_feature] + opt3)

        distance_e = torch.norm(out_feature_ent - ref_emb_e, p=2, dim=1)
        distance_r = torch.norm(out_feature_rel - ref_emb_r, p=2, dim=1)
        distance_t = torch.norm(out_feature_time - ref_emb_t, p=2, dim=1)
        distance_it = torch.norm(out_feature_int_time - ref_emb_it, p=2, dim=1)
        distance_e = distance_e.unsqueeze(-1)
        distance_r = distance_r.unsqueeze(-1)
        distance_t = distance_t.unsqueeze(-1)
        distance_it = distance_it.unsqueeze(-1)

        #out_feature_overall = torch.cat((out_feature_ent, out_feature_rel), dim=-1)
        #out_feature_time = torch.cat((out_feature_time, out_feature_int_time), dim=-1)
        out_feature = torch.cat((out_feature_ent,\
                                  out_feature_rel, out_feature_time, out_feature_int_time), dim=-1)
        out_feature_rel = torch.cat((out_feature_ent, out_feature_rel), dim=-1)
        out_feature_time = torch.cat((out_feature_time, out_feature_int_time), dim=-1)


        #overall_distance = torch.norm(out_feature - ref_emb, p=2, dim=1)
        #overall_distance = overall_distance.unsqueeze(-1)

        feature_vector = torch.cat([distance_e, distance_r, distance_t, distance_it], dim=-1)
        output_array = self.weight_mlp(feature_vector)

        if self.use_mlp:
            out_feature = \
            [output_array * out_feature_rel, (1 - output_array) * out_feature_time]
            out_feature = torch.cat(out_feature, dim=-1)       
        else:
            out_feature = torch.cat([out_feature_rel, out_feature_time], dim=-1)

        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)

        if self.return_loss:
            loss = self.criterion_cl_joint(out_feature, batch)
            return out_feature, loss
        else:
            return out_feature


class GraphAttention(nn.Module):
    def __init__(self, node_size, rel_size, triple_size,
                 activation=torch.tanh, use_bias=True,
                 attn_heads=1, dim=100,
                 depth=1, device='cpu'):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.activation = activation
        self.rel_size = rel_size

        self.triple_size = triple_size
        self.use_bias = use_bias
        self.attn_heads = attn_heads
        self.attn_heads_reduction = 'concat'
        self.depth = depth
        self.device = device
        self.attn_kernels = []
        self.att = True
        self.gate = False

        node_F = dim
        rel_F = dim
        self.ent_F = node_F
        ent_F = self.ent_F

        # gate kernel Eq 9 M
        self.gate_kernel = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.proxy = OverAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = OverAll.init_emb(1, ent_F * (self.depth + 1))
        for d in range(self.depth):
            self.attn_kernels.append([])
            for h in range(self.attn_heads):
                attn_kernel = OverAll.init_emb(node_F, 1)
                self.attn_kernels[d].append(attn_kernel.to(device))

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj_index = inputs[2]  # adj
        none_relation = inputs[3] 
        index = torch.tensor(adj_index, dtype=torch.int64)
        index = index.to(self.device)

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            neighs = features[index[:, 1]]
            #here we implement the similarity between the none temporal and neighbour 
            #and the attention mechanism.
            if self.att:
                sim_score = torch.squeeze(-torch.matmul(F.normalize(neighs, dim=-1), \
                    torch.transpose(F.normalize(none_relation.reshape(1, -1), dim=-1), 0, 1)), dim=-1)
                #print(sim_score.shape)
                sim_score = torch.sparse.FloatTensor\
                (torch.transpose(index, 0, 1), sim_score, (self.node_size, self.node_size))
                
                sim_att = torch.sparse.softmax(sim_score, dim=1)
            else:
                sim_score = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), torch.ones_like(index[:, 0], dtype=torch.float), \
                                            (self.node_size, self.node_size))
                sim_att = torch.sparse.softmax(sim_score, dim=1)

            new_features = torch_scatter.scatter_add(
                torch.transpose(neighs * torch.unsqueeze(sim_att.coalesce().values(), dim=-1), 0, 1),
                index[:, 0])
            new_features = torch.transpose(new_features, 0, 1)

            features = self.activation(new_features)
            outputs.append(features)

        if self.gate:
            outputs = torch.cat(outputs, dim=1)
            proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
                                 torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
            proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
            proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)

            if self.use_bias:
                gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
            else:
                gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
            outputs = gate_rate * outputs + (1 - gate_rate) * proxy_feature
        else:
            outputs = torch.cat(outputs, dim=1)

        return outputs