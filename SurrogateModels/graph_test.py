from torchview import draw_graph
from torch import nn
import torch
import graphviz

graphviz.set_jupyter_format('png')

class MLP_test(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
    
model_graph_1 = draw_graph(
    MLP_test(), input_size=(2, 128),
    graph_name='MLP',
    hide_inner_tensors=False,
    hide_module_functions=False,
)

model_graph_1.visual_graph()