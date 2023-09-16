from functools import lru_cache
from typing import Callable

from requests import options
import torch
import dgl
from dgl import function as fn
import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    NaturalVariationalDistribution,
    TrilNaturalVariationalDistribution,
    MeanFieldVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
)
from gpytorch.kernels import (
    ScaleKernel, RBFKernel, LinearKernel, CosineKernel, MaternKernel,
    PolynomialKernel,
    GridInterpolationKernel, SpectralMixtureKernel, GaussianSymmetrizedKLKernel
)
from .variational import GraphVariationalStrategy
from dgl.nn.functional import edge_softmax
from torchdiffeq import odeint_adjoint as odeint
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP

# from torchdiffeq import odeint

class ODEFunc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._g = None
        self._e = None
        self.alpha = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def g(self):
        return self._g

    @property
    def e(self):
        return self._e
    
    @g.setter
    def g(self, g):
        self._g = g

    @e.setter
    def e(self, e):
        self._e = e

    def forward(self, t, x):
        h, e = x[:self.node_shape.numel()], x[self.node_shape.numel():]
        h, e = h.reshape(*self.node_shape), e.reshape(*self.edge_shape)
        h0 = h
        g = self.g.local_var()
        g.edata["e"] = e
        g.ndata["h"] = h
        g.update_all(fn.u_mul_e("h", "e", "m"), fn.sum("m", "h"))
        h = g.ndata["h"]
        h = self.alpha.sigmoid() * (h - h0)
        h, e = h.flatten(), e.flatten()
        x = torch.cat([h, torch.zeros_like(e)])
        return x

class Rewire(torch.nn.Module):
    def __init__(self, in_features, out_features, t=1.0, n:int=8):
        super().__init__()
        self.fc_k = torch.nn.Linear(in_features, out_features, bias=False)
        self.fc_q = torch.nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("t", torch.tensor(t))
        self.n = n
        torch.nn.init.constant_(self.fc_k.weight, 1e-5)
        torch.nn.init.constant_(self.fc_q.weight, 1e-5)
        self.odefunc = ODEFunc()

    def forward(self, h, g):
        g = g.local_var()
        h = h - h.mean(-1, keepdims=True)
        h = torch.nn.functional.normalize(h, dim=-1)
        k = self.fc_k(h)
        q = self.fc_q(h)
        g.ndata["k"] = k
        g.ndata["q"] = q
        g.apply_edges(dgl.function.u_dot_v("k", "q", "e"))
        e = g.edata["e"] / k.shape[-1] ** 0.5
        e = edge_softmax(g, e)

        node_shape = h.shape
        self.odefunc.node_shape = node_shape
        self.odefunc.edge_shape = e.shape
        self.odefunc.g = g
        t = torch.linspace(0, self.t, self.n, device=h.device, dtype=h.dtype)
        x = torch.cat([h.flatten(), e.flatten()])
        x = odeint(self.odefunc, x, t, method="dopri5")# [-1]
        h, e = x[:, :h.numel()], x[:, h.numel():]
        h = h.reshape(-1, *node_shape)
        return h

@lru_cache(maxsize=1)
def graph_exp(graph):
    a = torch.zeros(
        graph.number_of_nodes(),
        graph.number_of_nodes(),
        dtype=torch.float32,
        device=graph.device,
    )
    src, dst = graph.edges()
    a[src, dst] = 1
    d = a.sum(-1, keepdims=True).clamp(min=1)
    a = a / d
    a = a - torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    a = torch.linalg.matrix_exp(a)
    return a

class EmbeddingLayer(DeepGPLayer):
    def __init__(
            self,
            features,
            hidden_features,
            out_features,
    ):
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=features.shape[0],
            # batch_shape=torch.Size([out_features]),
        )

        variational_strategy = VariationalStrategy(
            self,
            features,
            variational_distribution,
            learn_inducing_locations=False,
        )

        super().__init__(variational_strategy, features.shape[-1], out_features)
        self.out_features = out_features
        self.register_buffer("features", features)
        self.fc = torch.nn.Linear(features.shape[-1], hidden_features)
        self.mean_module = gpytorch.means.LinearMean(hidden_features)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([out_features]), ard_num_dims=hidden_features),
            batch_shape=torch.Size([out_features]),
        )

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def __call__(self, *args, **kwargs):
        x = self.features
        return super().__call__(x, *args, **kwargs)




class BklynModel(DeepGP):
    def __init__(self, features, hidden_features, out_features):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(features, hidden_features, out_features)

    def forward(self):
        x = self.embedding_layer()
        return x
    


    