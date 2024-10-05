import torch
import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange
import core.model_utils.gMHA_wrapper as gMHA_wrapper

from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN


class GraphFractal(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type=None,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 alpha_type = 'sc',
                 hpf = True,
                 fn_only = False,
                 alpha_init = 'zero'):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0

        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        
        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)
        
        self.n_patches = n_patches
        self.alpha_type = alpha_type
        self.hpf = hpf
        self.fn_only = fn_only

        if alpha_type == 'sc':
            if alpha_init == 'zero':
                self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(nlayer_gnn)])
            elif alpha_init == 'one':
                self.alpha = nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=True) for _ in range(nlayer_gnn)])
        elif alpha_type == 'vc':
            if alpha_init == 'zero':
                self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(n_patches), requires_grad=True) for _ in range(nlayer_gnn)])
            elif alpha_init == 'one':
                self.alpha = nn.ParameterList([nn.Parameter(torch.ones(n_patches), requires_grad=True) for _ in range(nlayer_gnn)])

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())

        # Node PE
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        # Subgraph GNN
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        
        ## Fractal nodes
        for i, gnn in enumerate(self.gnns):
            x = gnn(x, edge_index, e)
            lpf_x = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
            
            if self.hpf:
                hpf_x = x - lpf_x
                if self.alpha_type == 'sc':
                    fractal_x = lpf_x + self.alpha[i] * hpf_x
                elif self.alpha_type == 'vc':
                    alpha_per_patch = self.alpha[i][data.subgraphs_nodes_mapper % self.n_patches]
                    fractal_x = lpf_x + alpha_per_patch.unsqueeze(-1) * hpf_x
            else:
                fractal_x = lpf_x            
            x = x + fractal_x
            x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]

        if self.fn_only:
            subgraph_x = scatter(fractal_x, batch_x, dim=0, reduce=self.pooling)
        else:
            subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        subgraph_x = self.reshape(subgraph_x)

        # Global Average Pooling
        x = (subgraph_x * data.mask.unsqueeze(-1)).sum(1) / \
            data.mask.sum(1, keepdim=True)

        # Readout
        x = self.output_decoder(x)
        return x

class GraphFractalMixer(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 alpha_type = 'sc',
                 hpf = True,
                 fn_only = False,
                 alpha_init = 'zero'):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0

        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.transformer_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)
        
        self.n_patches = n_patches
        self.alpha_type = alpha_type
        self.hpf = hpf
        self.fn_only = fn_only

        if alpha_type == 'sc':
            if alpha_init == 'zero':
                self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(nlayer_gnn)])
            elif alpha_init == 'one':
                self.alpha = nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=True) for _ in range(nlayer_gnn)])
        elif alpha_type == 'vc':
            if alpha_init == 'zero':
                self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(n_patches), requires_grad=True) for _ in range(nlayer_gnn)])
            elif alpha_init == 'one':
                self.alpha = nn.ParameterList([nn.Parameter(torch.ones(n_patches), requires_grad=True) for _ in range(nlayer_gnn)])


    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())

        # Node PE
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        # Subgraph GNN
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        ## Fractal nodes
        for i, gnn in enumerate(self.gnns):
            x = gnn(x, edge_index, e)
            lpf_x = scatter(x, batch_x, dim=0, reduce=self.pooling)[batch_x]
            
            if self.hpf:
                hpf_x = x - lpf_x
                if self.alpha_type == 'sc':
                    fractal_x = lpf_x + self.alpha[i] * hpf_x
                elif self.alpha_type == 'vc':
                    alpha_per_patch = self.alpha[i][data.subgraphs_nodes_mapper % self.n_patches]
                    fractal_x = lpf_x + alpha_per_patch.unsqueeze(-1) * hpf_x
            else:
                fractal_x = lpf_x            
            x = x + fractal_x
            x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')[data.subgraphs_nodes_mapper]

        if self.fn_only:
            subgraph_x = scatter(fractal_x, batch_x, dim=0, reduce=self.pooling)
        else:
            subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        # Patch PE
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)
        mixer_x = self.reshape(subgraph_x)

        # MLPMixer
        mixer_x = self.transformer_encoder(mixer_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~data.mask)

        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / \
            data.mask.sum(1, keepdim=True)

        # Readout
        x = self.output_decoder(x)
        return x