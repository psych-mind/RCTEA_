import faiss
#we here first arrange with the code for the new setting.
import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.inits import reset
import numpy as np
import torch.nn.functional as F

try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

EPS = 1e-8

def masked_softmax(src, mask, dim=-1):
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)
    return out

def to_sparse(x, mask):
    return x[mask]


def to_dense(x, mask):
    out = x.new_zeros(tuple(mask.size()) + (x.size(-1), ))
    out[mask] = x
    return out

class WeightMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(WeightMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()  # Outputs weights in [0, 1]
        )

    def forward(self, input_array):
        # Concatenate inputs
        #combined_input = torch.cat((x1, x2), dim=-1)
        weight = self.mlp(input_array)
        return weight

class DGMC(torch.nn.Module):
    r"""The *Deep Graph Matching Consensus* module which first matches nodes
    locally via a graph neural network :math:`\Psi_{\theta_1}`, and then
    updates correspondence scores iteratively by reaching for neighborhood
    consensus via a second graph neural network :math:`\Psi_{\theta_2}`.

    .. note::
        See the `PyTorch Geometric introductory tutorial
        <https://pytorch-geometric.readthedocs.io/en/latest/notes/
        introduction.html>`_ for a detailed overview of the used GNN modules
        and the respective data format.

    Args:
        psi_1 (torch.nn.Module): The first GNN :math:`\Psi_{\theta_1}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            computes node embeddings.
        psi_2 (torch.nn.Module): The second GNN :math:`\Psi_{\theta_2}` which
            takes in node features :obj:`x`, edge connectivity
            :obj:`edge_index`, and optional edge features :obj:`edge_attr` and
            validates for neighborhood consensus.
            :obj:`psi_2` needs to hold the attributes :obj:`in_channels` and
            :obj:`out_channels` which indicates the dimensionality of randomly
            drawn node indicator functions and the output dimensionality of
            :obj:`psi_2`, respectively.
        num_steps (int): Number of consensus iterations.
        k (int, optional): Sparsity parameter. If set to :obj:`-1`, will
            not sparsify initial correspondence rankings. (default: :obj:`-1`)
        detach (bool, optional): If set to :obj:`True`, will detach the
            computation of :math:`\Psi_{\theta_1}` from the current computation
            graph. (default: :obj:`False`)
    """
    def __init__(self, psi_1, psi_2, psi_3, num_steps, k=-1, detach=False):
        super(DGMC, self).__init__()

        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.psi_3 = psi_3
        self.num_steps = num_steps
        self.k = k
        self.detach = detach
        self.weight_mlp = WeightMLP(1, 4)
        self.backend = 'auto'

        self.mlp = Seq(
            Lin(psi_3.out_channels, psi_3.out_channels),
            ReLU(),
            Lin(psi_3.out_channels, 1),
        )

    def reset_parameters(self):
        self.psi_1.reset_parameters()
        self.psi_2.reset_parameters()
        self.psi_3.reset_parameters()
        reset(self.mlp)

    def __top_k__(self, x_s, x_t):  # pragma: no cover
        r"""Memory-efficient top-k correspondence computation."""
        if LazyTensor is not None:
            x_s = x_s.unsqueeze(-2)  # [..., n_s, 1, d]
            x_t = x_t.unsqueeze(-3)  # [..., 1, n_t, d]
            x_s, x_t = LazyTensor(x_s), LazyTensor(x_t)
            S_ij = (-x_s * x_t).sum(dim=-1)
            return S_ij.argKmin(self.k, dim=2, backend=self.backend)
        else:
            x_s = x_s  # [..., n_s, d]
            x_t = x_t.transpose(-1, -2)  # [..., d, n_t]
            S_ij = x_s @ x_t
            return S_ij.topk(self.k, dim=2)[1]
    
    def custom_top_k(self, x_s, x_t):
        """top-k correspondence computation for the efficient faiss framework
        for the comparison direction, we do from x_s to x_t, (ent_s, top_k)"""
        index = faiss.IndexFlatIP(x_t.shape[-1])  # Use IndexFlatIP for dot product similarity
        index.add(x_t)  # Add all the entity embeddings to the index

        # Process queries in batches to handle memory efficiently
        batch_size = 2000  # Adjust based on your memory capacity
        all_distances = []
        all_indices = []

        for start in range(0, x_s.shape[0], batch_size):
            end = min(start + batch_size, x_s.shape[0])
            distances, indices = index.search(x_s[start:end], self.k)
            all_distances.append(distances)
            all_indices.append(indices)

        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        all_indices = torch.LongTensor(all_indices)
        return all_indices.unsqueeze(0)
    
    def custom_top_k_gpu(self, x_s, x_t, batch_size=1000):
        """top-k correspondence computation for the efficient faiss framework
        for the comparison direction, we do from x_s to x_t, (ent_s, top_k)
        using the gpu to compute it"""
        res = faiss.StandardGpuResources() 
        #res.setTempMemory(500 * 1024 * 1024)
        index = faiss.IndexFlatIP(x_t.shape[-1])
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
        index_gpu.add(x_t)

        # Process queries in batches to handle memory efficiently
        batch_size = 2000  # Adjust based on your memory capacity
        all_distances = []
        all_indices = []

        for start in range(0, x_s.shape[0], batch_size):
            end = min(start + batch_size, x_s.shape[0])
            distances, indices = index_gpu.search(x_s[start:end], self.k)
            all_distances.append(distances)
            all_indices.append(indices)

        all_distances = np.vstack(all_distances)
        all_indices = np.vstack(all_indices)
        all_indices = torch.LongTensor(all_indices)
        return all_indices.unsqueeze(0)

    def __include_gt__(self, S_idx, s_mask, y):
        r"""Includes the ground-truth values in :obj:`y` to the index tensor
        :obj:`S_idx`."""
        (B, N_s), (row, col), k = s_mask.size(), y, S_idx.size(-1)

        gt_mask = (S_idx[s_mask][row] != col.view(-1, 1)).all(dim=-1)

        sparse_mask = gt_mask.new_zeros((s_mask.sum(), ))
        sparse_mask[row] = gt_mask

        dense_mask = sparse_mask.new_zeros((B, N_s))
        dense_mask[s_mask] = sparse_mask
        last_entry = torch.zeros(k, dtype=torch.bool, device=gt_mask.device)
        last_entry[-1] = 1
        dense_mask = dense_mask.view(B, N_s, 1) * last_entry.view(1, 1, k)

        return S_idx.masked_scatter(dense_mask, col[gt_mask])

    def forward(self, inputs, index_n1, index_n2, x_s, edge_index_s, edge_attr_s, batch_s, x_t,
                edge_index_t, edge_attr_t, batch_t, selection_index, y=None):
        r"""
        Args:
            x_s (Tensor): Source graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_s (LongTensor): Source graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_s (Tensor): Source graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Source graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            x_t (Tensor): Target graph node features of shape
                :obj:`[batch_size * num_nodes, C_in]`.
            edge_index_t (LongTensor): Target graph edge connectivity of shape
                :obj:`[2, num_edges]`.
            edge_attr_t (Tensor): Target graph edge features of shape
                :obj:`[num_edges, D]`. Set to :obj:`None` if the GNNs are not
                taking edge features into account.
            batch_s (LongTensor): Target graph batch vector of shape
                :obj:`[batch_size * num_nodes]` indicating node to graph
                assignment. Set to :obj:`None` if operating on single graphs.
            y (LongTensor, optional): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]` to include ground-truth values
                when training against sparse correspondences. Ground-truths
                are only used in case the model is in training mode.
                (default: :obj:`None`)

        Returns:
            Initial and refined correspondence matrices :obj:`(S_0, S_L)`
            of shapes :obj:`[batch_size * num_nodes, num_nodes]`. The
            correspondence matrix are either given as dense or sparse matrices.
        """
        
        #h_s = self.psi_1(x_s, edge_index_s, edge_attr_s)
        #h_t = self.psi_1(x_t, edge_index_t, edge_attr_t)
        #here we generate wiht the h_s and h_t.
        #so, here we regard psi_1 with the temporal model
        #psi_2 with the relation model.
        #we follow with the previous setting. 
        time_emb = self.psi_1(inputs)
        #time_ref = self.psi_1.time_emb[0]
        #time_int_ref = self.psi_1.time_int_emb[2542]

        #ref_emb = torch.cat([time_ref, time_ref, time_ref,\
        #            time_int_ref, time_int_ref, time_int_ref])

        #select with the correspounding indices for both KGs.
        h_st = time_emb[np.array(list(index_n1.keys()))]
        h_tt = time_emb[np.array(list(index_n2.keys()))]

        h_st, h_tt = (h_st.detach(), h_tt.detach()) if self.detach else (h_st, h_tt)

        h_st, s_mask = to_dense_batch(h_st, batch_s, fill_value=0)
        h_tt, t_mask = to_dense_batch(h_tt, batch_t, fill_value=0)

        #extract with the relation part feature.
        rel_emb = self.psi_2(inputs)
        h_sr = rel_emb[np.array(list(index_n1.keys()))]
        h_tr = rel_emb[np.array(list(index_n2.keys()))]
        h_sr = h_sr.unsqueeze(0)
        h_tr = h_tr.unsqueeze(0)
        C_out_r = h_sr.size(-1)

        #here, we get with the concatenation for the overall part of the embedding.
        h_s = torch.cat((h_st, h_sr), dim=-1)
        h_t = torch.cat((h_tt, h_tr), dim=-1)
        
        assert h_st.size(0) == h_tt.size(0), 'Encountered unequal batch-sizes'
        (B, N_s, C_out), N_t = h_st.size(), h_tt.size(1)
        R_in, R_out = self.psi_3.in_channels, self.psi_3.out_channels

        if self.k < 1:
            # ------ Dense variant ------ #
            S_hat = h_s @ h_t.transpose(-1, -2)  # [B, N_s, N_t, C_out]
            S_mask = s_mask.view(B, N_s, 1) & t_mask.view(B, 1, N_t)
            S_0 = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            for _ in range(self.num_steps):
                S = masked_softmax(S_hat, S_mask, dim=-1)
                r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
                                  device=h_s.device)
                r_t = S.transpose(-1, -2) @ r_s

                r_s, r_t = to_sparse(r_s, s_mask), to_sparse(r_t, t_mask)
                o_s = self.psi_3(r_s, edge_index_s, edge_attr_s)
                o_t = self.psi_3(r_t, edge_index_t, edge_attr_t)
                o_s, o_t = to_dense(o_s, s_mask), to_dense(o_t, t_mask)

                D = o_s.view(B, N_s, 1, R_out) - o_t.view(B, 1, N_t, R_out)
                S_hat = S_hat + self.mlp(D).squeeze(-1).masked_fill(~S_mask, 0)

            S_L = masked_softmax(S_hat, S_mask, dim=-1)[s_mask]

            return S_0, S_L
        else:
            # ------ Sparse variant ------ #
            #here, we rewrite our version of retrieving the top k entities.
            #S_idx = self.custom_top_k_gpu(h_s.squeeze(0).detach().cpu().numpy(), \
            #h_t.squeeze(0).detach().cpu().numpy(), batch_size=10000)

            #here, we get with the top_k relation part of the result.
            S_idx_r = self.custom_top_k_gpu(h_s.squeeze(0).detach().cpu().numpy(), \
            h_t.squeeze(0).detach().cpu().numpy(), batch_size=10000)

            #S_idx = S_idx.to('cuda:0')
            S_idx = S_idx_r.to('cuda:0')
            #S_idx = torch.cat((S_idx, S_idx_r), dim=-1)
            #S_idx = S_idx_r

            # In addition to the top-k, randomly sample negative examples and
            # ensure that the ground-truth is included as a sparse entry.
            if self.training and y is not None:
                #so, here, the k size is with the number of index generated.
                #rnd_size = (B, N_s, min(self.k, N_t - self.k))
                #S_rnd_idx = torch.randint(N_t, rnd_size, dtype=torch.long,
                #                          device=S_idx.device)
                #S_idx = torch.cat([S_idx, S_rnd_idx], dim=-1)
                #S_idx_r = torch.cat([S_idx_r, S_rnd_idx], dim=-1)
                S_idx = self.__include_gt__(S_idx, s_mask, y)
                #S_idx_r = self.__include_gt__(S_idx_r, s_mask, y)
            
            #here is with the score computation.
            #the temporal part of score.
            k = S_idx.size(-1)
            idx = S_idx.view(B, N_s * k, 1).expand(-1, -1, C_out)

            tmp_st = h_st.view(B, N_s, 1, C_out)
            tmp_tt = h_tt[0][idx[0, :, 0]].unsqueeze(0)
            
            S_hat_t = (tmp_st * tmp_tt.view(B, N_s, k, C_out)).sum(dim=-1)

            #the relation part of score.
            tmp_sr = h_sr.view(B, N_s, 1, C_out_r)
            #this part should be with the specific S_idx calculated by aspect.
            #can be trimed.
            #idx_r = S_idx_r.view(B, N_s * k, 1).expand(-1, -1, C_out_r)
            tmp_tr = h_tr[0][idx[0, :, 0]].unsqueeze(0)

            S_hat_r = (tmp_sr * tmp_tr.view(B, N_s, k, C_out_r)).sum(dim=-1)

            #the softmax version of score. 
            #S_0 = S_hat.softmax(dim=-1)[s_mask]

            for _ in range(self.num_steps):
                #here, we have that relation part of sim matrix
                #and temporal part of sim matrix and basically we
                #process two sim matrix accordingly.
                S_t = S_hat_t.softmax(dim=-1)
                S_r = S_hat_r.softmax(dim=-1)

                #the random embedding definition. 
                r_s = torch.randn((B, N_s, R_in), dtype=h_s.dtype,
                                  device=h_s.device)
                                  
                #first we process with the temporal part.
                tmp_tt = r_s.view(B, N_s, 1, R_in) * S_t.view(B, N_s, k, 1)
                tmp_tt = tmp_tt.view(B, N_s * k, R_in)
                idx = S_idx.view(B, N_s * k, 1)
                #the temporal part reconstruction
                r_tt = scatter_add(tmp_tt, idx, dim=1, dim_size=N_t)

                #process with the relation part. 
                tmp_tr = r_s.view(B, N_s, 1, R_in) * S_r.view(B, N_s, k, 1)
                tmp_tr = tmp_tr.view(B, N_s * k, R_in)
                r_tr = scatter_add(tmp_tr, idx, dim=1, dim_size=N_t)
                
                r_s, r_tt, r_tr = to_sparse(r_s, s_mask), \
                    to_sparse(r_tt, t_mask), to_sparse(r_tr, t_mask)
                
                o_s = self.psi_3(r_s, edge_index_s, edge_attr_s)
                #the temporal part GNN embedding.
                o_tt = self.psi_3(r_tt, edge_index_t, edge_attr_t)
                #the relation part GNN embedding.
                o_tr = self.psi_3(r_tr, edge_index_t, edge_attr_t)
                o_s, o_tt, o_tr = to_dense(o_s, s_mask),\
                      to_dense(o_tt, t_mask), to_dense(o_tr, t_mask)

                o_s = o_s.view(B, N_s, 1, R_out).expand(-1, -1, k, -1)
                idx = S_idx.view(B, N_s * k, 1).expand(-1, -1, R_out)
                #the temporal part embedding.
                tmp_tt = torch.gather(o_tt.view(B, N_t, R_out), -2, idx)
                #the relation part embedding.
                tmp_tr = torch.gather(o_tr.view(B, N_t, R_out), -2, idx)

                D_tt = o_s - tmp_tt.view(B, N_s, k, R_out)
                D_tr = o_s - tmp_tr.view(B, N_s, k, R_out)
                S_hat_t = S_hat_t + self.mlp(D_tt).squeeze(-1)
                S_hat_r = S_hat_r + self.mlp(D_tr).squeeze(-1)

            #here, we combine with the two parts of sim matrix.
            S_hat = (S_hat_t + S_hat_r) / 2
            #S_0 = S_hat.softmax(dim=-1)[s_mask]
            S_L = S_hat.softmax(dim=-1)[s_mask]
            S_idx = S_idx[s_mask]

            # Convert sparse layout to `torch.sparse_coo_tensor`.
            row = torch.arange(h_s.size(1), device=S_idx.device)
            row = row.view(-1, 1).repeat(1, k)
            idx = torch.stack([row.view(-1), S_idx.view(-1)], dim=0)
            size = torch.Size([h_s.size(1), N_t])

            #S_sparse_0 = torch.sparse_coo_tensor(
            #    idx, S_0.view(-1), size, requires_grad=S_0.requires_grad)
            #S_sparse_0.__idx__ = S_idx
            #S_sparse_0.__val__ = S_0

            S_sparse_L = torch.sparse_coo_tensor(
                idx, S_L.view(-1), size, requires_grad=S_L.requires_grad)
            S_sparse_L.__idx__ = S_idx
            S_sparse_L.__val__ = S_L

            return '', S_sparse_L

    def loss(self, S, y, reduction='mean'):
        r"""Computes the negative log-likelihood loss on the correspondence
        matrix.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'none'|'mean'|'sum'`.
                (default: :obj:`'mean'`)
        """
        assert reduction in ['none', 'mean', 'sum']
        if not S.is_sparse:
            val = S[y[0], y[1]]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            mask = S.__idx__[y[0]] == y[1].view(-1, 1)
            val = S.__val__[[y[0]]][mask]
        nll = -torch.log(val + EPS)
        return nll if reduction == 'none' else getattr(torch, reduction)(nll)

    def acc(self, S, y, reduction='mean'):
        r"""Computes the accuracy of correspondence predictions.

        Args:
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argmax(dim=-1)
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            pred = S.__idx__[y[0], S.__val__[y[0]].argmax(dim=-1)]

        correct = (pred == y[1]).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    def hits_at_k(self, k, S, y, reduction='mean'):
        r"""Computes the hits@k of correspondence predictions.

        Args:
            k (int): The :math:`\mathrm{top}_k` predictions to consider.
            S (Tensor): Sparse or dense correspondence matrix of shape
                :obj:`[batch_size * num_nodes, num_nodes]`.
            y (LongTensor): Ground-truth matchings of shape
                :obj:`[2, num_ground_truths]`.
            reduction (string, optional): Specifies the reduction to apply to
                the output: :obj:`'mean'|'sum'`. (default: :obj:`'mean'`)
        """
        assert reduction in ['mean', 'sum']
        if not S.is_sparse:
            pred = S[y[0]].argsort(dim=-1, descending=True)[:, :k]
        else:
            assert S.__idx__ is not None and S.__val__ is not None
            perm = S.__val__[y[0]].argsort(dim=-1, descending=True)[:, :k]
            pred = torch.gather(S.__idx__[y[0]], -1, perm)

        correct = (pred == y[1].view(-1, 1)).sum().item()
        return correct / y.size(1) if reduction == 'mean' else correct

    #def __repr__(self):
    #    return ('{}(\n'
    #            '    psi_1={},\n'
    #            '    psi_2={},\n'
    #            '    num_steps={}, k={}\n)').format(self.__class__.__name__,
    #                                                self.psi_1, self.psi_2,
    #                                                self.num_steps, self.k)
