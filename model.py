import torch_geometric as tg
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
import torch_geometric.datasets as datasets


def build_mlp(in_channels = 33,num_layers=2, hidden_channels=128, out_channels=128):
    return tg.nn.MLP(
        in_channels = in_channels, 
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers
    )

def layerNorm(latentSize = 128):
    return tg.nn.norm.LayerNorm(in_channels = latentSize)

# encoder
# Update information of the global, node, and edge features without message passing scheme



class encoder(nn.Module):
    # get a graph and spit out a latent graph!
    def __init__(self, numberHiddenLayers, 
                 numOfHidden,
                 latentSize) -> None:
        super().__init__()
        self._nodes_mlp = build_mlp(num_layers=numberHiddenLayers,
                              hidden_channels = numOfHidden,
                              out_channels = latentSize,
                              )
        self._edges_mlp = build_mlp(
            in_channels=72
        )
        self._normLayer = tg.nn.norm.LayerNorm(in_channels = latentSize)

    def apply_mlp_norm(self, input, nodes):
        print(f'This is input: {input}')
        if nodes:
            g1 = self._nodes_mlp(input.float())
        else: 
            g1 = self._edges_mlp(input.float())
        latent_graph = self._normLayer(g1)
        return latent_graph

    def forward(self, data):
        print(f'This is the data: {data}')
        node_features = self.apply_mlp_norm(data.x, nodes=True)
        edge_features = self.apply_mlp_norm(data.edge_attr.unsqueeze(dim=0), nodes=False)
        return Data(x=node_features, edge_index= data.edge_index, edge_attr=edge_features)


# Process
# This is where the message passing is happening
# Recive a graph, return a graph

class interactionNetwork(MessagePassing):
    def __init__(self, aggr = 'add', *, aggr_kwargs = None, flow = "source_to_target", node_dim = -2, decomposed_layers = 1):
        super().__init__(aggr, aggr_kwargs=aggr_kwargs, flow=flow, node_dim=node_dim, decomposed_layers=decomposed_layers)
        self.mlp = build_mlp(in_channels=216, out_channels=72)
        self.layernorm = layerNorm()

    def forward(self, data):
        print(data.num_nodes)
        node_messages = self.propagate(data.edge_index, x=data.x, edge_attr=data.edge_attr)
        new_x = self.layernorm(self.mlp(torch.cat([data.x, node_messages])))
        data.x = new_x
        data.edge_attr = node_messages
        return data
    
    def message(self, x_i, x_j, edge_attr):
        dd = torch.concat([x_i.float(), x_j.float(), edge_attr.float()]).squeeze()
        up_edge = self.mlp(dd.unsqueeze(dim=0))
        return torch.permute(up_edge, [1, 0])




# Decoder
# A simple mlp to aggregate information on the nodes

class decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mlp = build_mlp(in_channels=128, out_channels=2)
    def forward(self, data):
        return self._mlp(data.x)


class network(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder = encoder()
        self._process = interactionNetwork()
        self._decoder = decoder()
    
    def processing_network(self, x):
        return self._process(x)

    def forward(self, input):
        latent_1 = self._encoder(input)
        processed_latent = self.processing_network(latent_1)
        return self._decoder(processed_latent)



model = encoder(3, 128, 128)
model_2 = interactionNetwork()

dataset = datasets.ZINC(root='./data/zink')
data = dataset[0]
print(f'raw data {data}')
# data.x = torch.permute(data.x, (1, 0))
model_2.eval()
# print(model(data))
print(model_2(data))