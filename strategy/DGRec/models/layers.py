import torch.nn as nn
import torch as th
import os
import sys

parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(parent)
from utility import get_jaccard_matrix


class DGRecLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.k = args.k
        self.device = args.gpu
        self.sigma = args.sigma
        self.gamma = args.gamma
        self.folder = args.folder
        self.distance = args.distance

        jaccard_path = f"{self.folder}/jaccard_{self.distance}_distances_{args.dataset}.npy"
        items_items_distance = get_jaccard_matrix(None, jaccard_path)
        self.items_items_sims = th.tensor(1 - items_items_distance, device=self.device)


    def similarity_matrix(self, items_id):

        sims = []
        for items_row in items_id:
            b = th.cartesian_prod(items_row, items_row)
            row_sims = self.items_items_sims[b[:, 0], b[:, 1]]
            sims.append(row_sims)

        batch_size, row_size = items_id.shape
        sims = th.cat(sims).reshape(batch_size, row_size, row_size)

        return sims

    def submodular_selection_feature(self, nodes):
        device = nodes.mailbox['m'].device
        feature = nodes.mailbox['m']

        items_id = nodes.mailbox["items_id"]

        sims = self.similarity_matrix(items_id)

        batch_num, neighbor_num, feature_size = feature.shape
        nodes_selected = []
        cache = th.zeros((batch_num, 1, neighbor_num), device=device)

        for i in range(self.k):
            gain = th.sum(th.maximum(sims, cache) - cache, dim=-1)

            selected = th.argmax(gain, dim=1)
            cache = th.maximum(sims[th.arange(batch_num, device=device), selected].unsqueeze(1), cache)

            nodes_selected.append(selected)

        return th.stack(nodes_selected).t()

    def sub_reduction(self, nodes):
        # -1 indicate user -> node, which does not include category information
        mail = nodes.mailbox['m']
        batch_size, neighbor_size, feature_size = mail.shape

        if (-1 in nodes.mailbox['c']) or nodes.mailbox['m'].shape[1] <= self.k:
            mail = mail.sum(dim=1)
        else:
            neighbors = self.submodular_selection_feature(nodes)
            mail = mail[th.arange(batch_size, dtype=th.long, device=mail.device).unsqueeze(-1), neighbors]
            mail = mail.sum(dim=1)
        return {'h': mail}

    def category_aggregation(self, edges):
        return {'c': edges.src['category'], 'm': edges.src['h'], 'items_id': edges.edges()[0]}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _, dst = etype
            feat_src = h[src]
            feat_dst = h[dst]

            degs = graph.out_degrees(etype=etype).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = th.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(self.category_aggregation, self.sub_reduction, etype=etype)

            rst = graph.nodes[dst].data['h']
            degs = graph.in_degrees(etype=etype).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm
            return rst
