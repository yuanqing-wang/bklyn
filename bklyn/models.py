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
        g.apply_edges(dgl.function.u_dot_v("k", "q", "e+"))
        g.apply_edges(dgl.function.u_dot_v("q", "k", "e-"))
        g.edata["e"] = g.edata["e+"] + g.edata["e-"]
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

class ExactBklynModel(ExactGP):
    def __init__(
            self, train_x, train_y, likelihood, num_classes, 
            features, graph, in_features, hidden_features, t,
            activation, n=8,
        ):
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size((num_classes,)),
        )

        self.covar_module = ScaleKernel(
            LinearKernel(
                batch_shape=torch.Size((num_classes,)),
            ),
            batch_shape=torch.Size((num_classes,)),
        )

        self.rewire = Rewire(
            hidden_features, hidden_features, t=t, n=n,
        )
        self.likelihood = likelihood
        self.num_classes = num_classes
        self.register_buffer("features", features)
        self.graph = graph
        self.fc = torch.nn.Linear(in_features, hidden_features, bias=False)
        # self.norm = torch.nn.LayerNorm(hidden_features, elementwise_affine=False)
        self.activation = activation

    def forward(self, x):
        h = self.fc(self.features)
        # h = self.norm(h)
        h = self.activation(h)
        mean = self.mean_module(h)
        h = self.rewire(h, self.graph)
        covar = self.covar_module(h).mean(0)
        x = x.squeeze().long()
        mean = mean[..., x]
        covar = covar[..., x, :][..., :, x]
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class ApproximateBklynModel(ApproximateGP):
    def __init__(
            self,
            features,
            inducing_points,
            graph: dgl.DGLGraph,
            in_features: int,
            hidden_features: int,
            num_classes: int,
            learn_inducing_locations: bool = False,
            t: float=1.0,
            activation: Callable=torch.nn.functional.silu,
            n: int=8,
    ):

        batch_shape = torch.Size([num_classes])
        variational_distribution = TrilNaturalVariationalDistribution(
            inducing_points.size(-1),
            batch_shape=batch_shape,
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        
        variational_strategy = IndependentMultitaskVariationalStrategy(
            variational_strategy, 
            num_tasks=num_classes,
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size((num_classes,)),
        )
        self.covar_module = LinearKernel()
        self.rewire = Rewire(
            hidden_features, hidden_features, t=t, n=n,
        )
        self.num_classes = num_classes
        self.register_buffer("features", features)
        self.graph = graph
        self.fc = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.activation = activation
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        h = self.fc(self.features)
        h = self.activation(h)
        h = self.dropout(h)
        mean = self.mean_module(h)
        h = self.rewire(h, self.graph)[-1]
        covar = self.covar_module(h)
        x = x.squeeze().long()
        mean = mean[..., x]
        covar = covar[..., x, :][..., :, x]
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    