import math
from abc import ABC, abstractmethod
from operator import mul
from functools import reduce
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


class Hypercube:
    """
    Hypercube for sampling of the input data.
    """

    def __init__(self, interval, dims=(1,)):
        self.interval = interval
        self.dims = dims

    @property
    def interval(self):
        return self.__interval

    @interval.setter
    def interval(self, value):
        if not len(value) == 2:
            raise ValueError(f"interval {value} must be of the form [a, b]")
        self.__interval = value

    @property
    def dims(self):
        return self.__dims

    @dims.setter
    def dims(self, value):
        if not isinstance(value, tuple):
            raise TypeError(f"dims {value} must be a tuple")
        self.__dims = value

    @property
    def mean(self):
        return sum(self.__interval) / 2

    @property
    def std(self):
        return (self.__interval[1] - self.__interval[0]) / math.sqrt(12)

    @property
    def dim_flat(self):
        return reduce(mul, self.__dims)

    def sample(self, batch_size):
        return torch.DoubleTensor(batch_size, *self.__dims).uniform_(*self.__interval) # only when use the finite difference for benchmark of greeks
        return torch.FloatTensor(batch_size, *self.__dims).uniform_(*self.__interval) # normally used

    def __repr__(self):
        return f'hypercube {self.__interval}^({"x".join(map(str,self.__dims))})'


class Data(Dataset):
    """
    Uniformly distributed input data as a PyTorch (infinite) dataset.
    """

    def __init__(self, hypercubes, batch_size, n_batches, get_X, get_K, get_r, get_sigma, t):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.hypercubes = hypercubes
        self.get_X = get_X
        self.get_K = get_K
        self.get_r = get_r
        self.get_sigma = get_sigma
        self.t = t

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        batch = {
            key: cube.sample(self.batch_size) for key, cube in self.hypercubes.items()
        }
        if self.get_X is not None:
            batch["x"] = self.get_X(batch, self.t)
        if self.get_K is not None:
            batch["K"] = self.get_K(batch)
        if self.get_r is not None:
            batch["r"] = self.get_r(batch)
        if self.get_sigma is not None:
            batch["sigma"] = self.get_sigma(batch)
        if "x" not in batch:
            batch["x"] = batch["s"].clone()
        return batch

class Pde(ABC):
    """
    Base class for different parametrized PDEs.
    """

    def __init__(self, hypercubes):
        super().__init__()
        self.hypercubes = hypercubes

    @property
    def hypercubes(self):
        return self.__hypercubes

    @hypercubes.setter
    def hypercubes(self, value):
        if not (
            isinstance(value, dict)
            and all(isinstance(cube, Hypercube) for cube in value.values())
        ):
            raise TypeError(f"{value} must be a dictionary consisting of hypercubes")
        self.__hypercubes = value

    @property
    def dim_flat(self):
        return sum([cube.dim_flat for cube in self.__hypercubes.values()])

    def dataloader(self, batch_size, n_batches, data_type, Testset_narrow_x=False, frezed_params={}):
        if not Testset_narrow_x:
            return DataLoader(
                    Data(self.__hypercubes, batch_size, n_batches, self.get_X, self.get_K, self.get_r, self.get_sigma, self.t), batch_size=None
                )
        ### Use it when want the testset x be in [9,10] 
        if data_type == 'train': 
            return DataLoader(
                Data(self.__hypercubes, batch_size, n_batches, self.get_X, self.get_K, self.get_r, self.get_sigma), batch_size=None
            )
        else:
            return DataLoader(
                Data(self.__hypercubes, batch_size, n_batches, None, self.get_K, self.get_r, self.get_sigma), batch_size=None
            )
        ### only use for compare with berner's results ####
        # return DataLoader(
        #     Data(self.__hypercubes, batch_size, n_batches, None, None, self.get_r, None), batch_size=None
        # )

    def naf(self, batch, param):
        raise NotImplementedError

    def normalize_and_flatten(self, batch, No_normalization_and_flatten=False):
        # batch = [
        #     (batch[param] - self.__hypercubes[param].mean) / self.hypercubes[param].std
        #     for param in self.params
        # ]
        batch = [
            self.naf(batch, param, No_normalization_and_flatten) for param in self.params
        ]
        return torch.cat([tensor.flatten(start_dim=1) for tensor in batch], dim=1)

    @property
    @abstractmethod
    def params(self):
        pass

    @staticmethod
    @abstractmethod
    def _check_dims(hypercubes):
        pass

    @staticmethod
    @abstractmethod
    def sde(batch):
        pass


    def __repr__(self):
        return f"Parametrized {self.__class__.__name__} PDE with hypercubes {self.__hypercubes}"

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    def get_greeks(self, batch, greeks=["delta"], d=1e-6):
        """
        Outputs the delta of the given samples with a dict by finit difference
        """
        dict_greek_param = {"delta": "x", "vega": "sigma", "c-delta": "rho"}
        res = {}
        for _g in greeks:
            _param = dict_greek_param[_g]
            original_param = batch[_param].clone()
            res[_g] = original_param.clone()
            # calculaate by central difference
            for _i in range(original_param.shape[-1]):
                batch[_param] = original_param.clone()
                batch[_param][:,_i] = original_param.clone()[:,_i]*(1 + d)
                y_p = self.solution(batch)
                batch[_param][:,_i] = original_param.clone()[:,_i]*(1 - d)
                y_m = self.solution(batch)
                res[_g][:,_i] = (y_p - y_m).flatten()/2/(original_param.clone()[:,_i]*d)
            
            # calculate use five grid point central difference methods
            # for _i in range(original_param.shape[-1]):
            #     batch[_param] = original_param.clone()
            #     batch[_param][:,_i] = original_param.clone()[:,_i]*(1 + d)
            #     y_p = self.solution(batch)
            #     batch[_param][:,_i] = original_param.clone()[:,_i]*(1 + 2*d)
            #     y_p2 = self.solution(batch)
            #     batch[_param][:,_i] = original_param.clone()[:,_i]*(1 - d)
            #     y_m = self.solution(batch)
            #     batch[_param][:,_i] = original_param.clone()[:,_i]*(1 - 2*d)
            #     y_m2 = self.solution(batch)
            #     res[_g][:,_i] = (-y_p2 + 8*y_p - 8*y_m + y_m2).flatten()/(original_param.clone()[:,_i]*12*d)
        return res


HYPERCUBES = {
    f"basket_{d_basket}d": {
        "t": Hypercube(interval=[0.0, 1.0]),
        "x": Hypercube(interval=[9.0, 10.0], dims=(d_basket,)),
        "sigma": Hypercube(
            interval=[0.1, 0.6], dims=(d_basket, d_basket, d_basket + 1)
        ),
        "mu": Hypercube(interval=[0.1, 0.6], dims=(d_basket, d_basket + 1)),
        "K": Hypercube(interval=[10.0, 12.0]),
    }
    for d_basket in range(1, 6)
}


class Basket(Pde):
    params = ("t", "x", "sigma", "mu", "K")

    def __init__(self, hypercubes=HYPERCUBES["basket_3d"]):
        super().__init__(hypercubes)

    @staticmethod
    def _check_dims(hypercubes):
        d = hypercubes["x"].dims[0]
        return all(
            [
                hypercubes["t"].dims == (1,),
                hypercubes["x"].dims == (d,),
                hypercubes["sigma"].dims == (d, d, d + 1),
                hypercubes["mu"].dims == (d, d + 1),
                hypercubes["K"].dims == (1,),
            ]
        )

    @staticmethod
    def sde(batch, steps=25):
        """
        Outputs batched realizations of the SDE.
        """
        batch_size, d = batch["x"].shape
        steplen = (batch["t"] / steps).flatten()
        std = torch.sqrt(steplen)
        outputs = batch["x"].clone()
        for _ in range(steps):
            dw = (
                torch.randn(
                    d, batch_size, dtype=batch["x"].dtype, device=batch["x"].device
                )
                * std
            )
            sigma_x = (
                torch.einsum("iklj, il -> ikj", batch["sigma"][:, :, :, :d], outputs)
                + batch["sigma"][:, :, :, d]
            )
            mu_x = (
                torch.einsum("ikj, ij -> ik", batch["mu"][:, :, :d], outputs)
                + batch["mu"][:, :, d]
            )
            outputs += torch.einsum("ij, i -> ij", mu_x, steplen) + torch.einsum(
                "ijk, ki -> ij", sigma_x, dw
            )
        return torch.nn.ReLU()(batch["K"] - outputs.mean(dim=1, keepdims=True))

    @staticmethod
    def solution(batch, steps=25, mc_rounds=1048576):
        """
        Outputs the MC approximated solution.
        """
        ys = []
        for t, x, sigma, mu, K in zip(
            batch["t"], batch["x"], batch["sigma"], batch["mu"], batch["K"]
        ):
            mu_t = mu[:, :-1].T
            steplen = t / steps
            std = torch.sqrt(steplen)
            outputs = x.expand(mc_rounds, -1).clone()
            for _ in range(steps):
                dw = (
                    torch.randn(mc_rounds, len(x), dtype=x.dtype, device=x.device) * std
                )
                sigma_x = (
                    torch.einsum("ijk, lj -> lik", sigma[:, :, :-1], outputs)
                    + sigma[:, :, -1]
                )
                mu_x = outputs @ mu_t + mu[:, -1]
                outputs += mu_x * steplen + torch.einsum("ijk, ik -> ij", sigma_x, dw)
            y = (torch.nn.ReLU()(K - outputs.mean(dim=1, keepdims=True))).mean(
                dim=0, keepdims=True
            )
            ys.append(y)
        return torch.cat(ys, dim=0)


def n_dist(x):
    """
    Cumulative distribution function of the standard normal distribution.
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def n_density(x):
    """
    Density function of the standard normal distribution.
    """
    return torch.exp(-(x ** 2) / 2.0) / math.sqrt(2.0 * math.pi)


HYPERCUBES["black_scholes_r"] = {
    # "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0]),
    "r": Hypercube(interval=[0.005, 0.08]),
    "q": Hypercube(interval=[0.00,0.1]),
    "sigma": Hypercube(interval=[0.1, 0.6]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}


class BSr(Pde):
    params = ("x", "r", "q", "sigma", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_r"], payoff=None):
        super().__init__(hypercubes)
        self.payoff = payoff


    @staticmethod
    def _check_dims(hypercubes):
        return all(cube.dims == (1,) for cube in hypercubes.values())

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.dt
        dw = t**0.5 * torch.randn(
            batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device
        )
        sde = batch["x"] * torch.exp(
             (batch["r"] - batch["q"]) * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        if not self.payoff:
            return torch.exp(-batch["r"] * t) * torch.nn.ReLU()(batch["K"]-sde)
        else:
            roam_batch = batch.copy()
            roam_batch["x"] = sde
            return torch.exp(-batch["r"] * t) * self.payoff(roam_batch, onlynet=True)

    @staticmethod
    def get_X(batch, t):
        """
        get the X from S_0
        """
        dw = t**0.5 * torch.randn(
            batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device
        )
        sde = batch["s"] * torch.exp(
            (batch["r"] - batch["q"]) * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the strike K from kappa and S_0, kappa means the moneyness, i.e., K/S_0.
        """
        return batch["kappa"] * batch["s"]
    
    get_r, get_sigma = None, None
    
    def naf(self, batch, param, No_normalization_and_flatten=False):
        # normalization for the input of NN.
        if No_normalization_and_flatten:
            return batch[param]
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
            # return (batch[param] - self.hypercubes["s"].mean * 1) / (1 ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * (0.4 / math.sqrt(12)) ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std

HYPERCUBES["black_scholes_basket"] = {
    # "t": Hypercube(interval=[0.0, 1.0]),
    "s": Hypercube(interval=[9.0, 10.0], dims=(10,)),
    "r": Hypercube(interval=[0.005, 0.08]),
    "q": Hypercube(interval=[0.00,0.1], dims=(10,)),
    "sigma": Hypercube(interval=[0.1, 0.6], dims=(10,)),
    "rho": Hypercube(interval=[-0.1, 0.8]),
    "kappa": Hypercube(interval=[0.8, 1.2]),
}

class BSbasket(Pde):
    params = ("x", "r", "q", "sigma", "rho", "K")

    def __init__(self, hypercubes=HYPERCUBES["black_scholes_basket"], payoff=None):
        super().__init__(hypercubes)
        self.payoff = payoff

    @staticmethod
    def _check_dims(hypercubes):
        return True

    def sde(self, batch):
        """
        Outputs batched realizations of the SDE.
        """
        t = self.dt
        n = batch["sigma"].shape[-1]
        batch_size = batch["sigma"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        dw = t**0.5 * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["x"].shape, dtype=batch["x"].dtype, device=batch["x"].device).unsqueeze(2)
        ).squeeze(2)
        sde = batch["x"] * torch.exp(
             (batch["r"] - batch["q"]) * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        if not self.payoff:
            return torch.exp(-batch["r"] * t) * torch.nn.ReLU()(batch["K"] - torch.exp(torch.mean(torch.log(sde), dim=1, keepdim=True)))
        else:
            roam_batch = batch.copy()
            roam_batch["x"] = sde
            return torch.exp(-batch["r"] * t) * self.payoff(roam_batch, onlynet=True)

    @staticmethod
    def get_X(batch, t):
        """
        get the X from S_0
        """
        n = batch["sigma"].shape[-1]
        batch_size = batch["sigma"].shape[0]
        RHO = batch["rho"].view(batch_size, 1, 1).expand(batch_size, n, n).clone()
        RHO.as_strided((batch_size, n), (n ** 2, n + 1)).fill_(1)
        sqrt_cov = torch.linalg.cholesky(RHO)
        dw = t**0.5 * torch.matmul(sqrt_cov, 
                                                torch.randn(batch["s"].shape, dtype=batch["s"].dtype, device=batch["s"].device).unsqueeze(2)
        ).squeeze(2)
        sde = batch["s"] * torch.exp(
            (batch["r"] - batch["q"]) * t - 0.5 * t * batch["sigma"] ** 2 + batch["sigma"] * dw
        )
        return sde

    @staticmethod
    def get_K(batch):
        """
        Get the K from kappa and S_0
        """
        return batch["kappa"] * torch.exp(torch.mean(torch.log(batch["s"]), dim=1, keepdim=True))
    
    get_r, get_sigma = None, None
    
    def naf(self, batch, param, No_normalization_and_flatten=False):
        if param == "x":
            return (batch[param] - self.hypercubes["s"].mean) /  self.hypercubes["s"].std
        elif param == 'K':
            return (batch[param] - self.hypercubes["s"].mean * self.hypercubes["kappa"].mean) / (self.hypercubes["kappa"].mean ** 2 * self.hypercubes["s"].std ** 2 + self.hypercubes["s"].mean ** 2 * self.hypercubes["kappa"].std ** 2) ** 0.5
        else:
            return (batch[param] - self.hypercubes[param].mean) / self.hypercubes[param].std


PDES = {pde.__name__: pde for pde in Pde.get_subclasses()}