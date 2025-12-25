r"""A module that is consist of a dataset loading function and a PyTorch dataloader loading function.
"""

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from functools import partial

from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    to_networkx,
    to_dense_adj,
)
from torch_scatter import scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.data.expander_edges import generate_random_expander


def read_meta_info(meta_info, config: Union[CommonArgs, Munch]):
    config.dataset.dataset_type = meta_info.dataset_type
    config.model.model_level = meta_info.model_level
    config.dataset.dim_node = meta_info.dim_node
    config.dataset.dim_edge = meta_info.dim_edge
    config.dataset.num_envs = meta_info.num_envs
    config.dataset.num_classes = meta_info.num_classes
    config.dataset.num_train_nodes = meta_info.get("num_train_nodes")
    config.dataset.num_domains = meta_info.get("num_domains")
    config.dataset.feat_dims = meta_info.get("feat_dims")
    config.dataset.edge_feat_dims = meta_info.get("edge_feat_dims")



def pre_transform_in_memory(datasets, transform_func):
    """Pre-transform already loaded PyG dataset object.

    Note: This implementation is adapted from GraphGPS (MIT License).
        Original source: https://github.com/rampasek/GraphGPS

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return datasets

    for split in ["train", "id_val", "id_test", "val", "test"]:
        if split in datasets:
            data_list = [
                transform_func(datasets[split].get(i)) for i in range(len(datasets[split]))
            ]
            data_list = list(filter(None, data_list))

            datasets[split]._indices = None
            datasets[split]._data_list = data_list
            datasets[split].data, datasets[split].slices = datasets[split].collate(
                data_list
            )


def compute_posenc_stats(data, is_undirected, config):
    """Precompute positional encodings for the given graph.

    Note: This implementation is adapted from GraphGPS (MIT License).
        Original source: https://github.com/rampasek/GraphGPS
    """
    # Basic preprocessing of the input graph.
    if hasattr(data, "num_nodes"):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = config.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == "none":
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if config.model.model_name in ["GOODFormer"]:
        L = to_scipy_sparse_matrix(
            *get_laplacian(
                undir_edge_index, normalization=laplacian_norm_type, num_nodes=N
            )
        )
        evals, evects = np.linalg.eigh(L.toarray())

        max_freqs = config.posenc_LapPE.eigen.max_freqs
        eigvec_norm = config.posenc_LapPE.eigen.eigvec_norm

        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects, max_freqs=max_freqs, eigvec_norm=eigvec_norm
        )

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm="L2"):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Note: This implementation is adapted from GraphGPS (MIT License).
        Original source: https://github.com/rampasek/GraphGPS

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float("nan"))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float("nan")).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.

    Note: This implementation is adapted from GraphGPS (MIT License).
        Original source: https://github.com/rampasek/GraphGPS
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = (
            torch.max(EigVecs.abs(), dim=0, keepdim=True)
            .values.clamp_min(eps)
            .expand_as(EigVecs)
        )
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(
            dim=0, keepdim=True
        )
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def load_dataset(name: str, config: Union[CommonArgs, Munch]) -> dir:
    r"""
    Load a dataset given the dataset name.

    Args:
        name (str): Dataset name.
        config (Union[CommonArgs, Munch]): Required configs:
            ``config.dataset.dataset_root``
            ``config.dataset.domain``
            ``config.dataset.shift_type``
            ``config.dataset.generate``

    Returns:
        A dataset object and new configs
            - config.dataset.dataset_type
            - config.model.model_level
            - config.dataset.dim_node
            - config.dataset.dim_edge
            - config.dataset.num_envs
            - config.dataset.num_classes

    """
    try:
        reset_random_seed(config)
        dataset, meta_info = register.datasets[name].load(
            dataset_root=config.dataset.dataset_root,
            domain=config.dataset.domain,
            shift=config.dataset.shift_type,
            generate=config.dataset.generate,
        )
    except KeyError as e:
        print("#IN#Dataset not found.")
        raise e
    
    if config.model.model_name in ["GOODFormer"]:
        is_undirected = all(d.is_undirected() for d in dataset["train"][:10])
        pre_transform_in_memory(
            dataset,
            partial(compute_posenc_stats, is_undirected=is_undirected, config=config),
        )
        if config.model.model_name == "Exphormer":
            pre_transform_in_memory(
                dataset,
                partial(
                    generate_random_expander,
                    degree=5,
                    algorithm="Hamiltonian",
                    rng=None,
                    max_num_iters=100,
                    exp_index=0,
                ),
            )

    read_meta_info(meta_info, config)

    config.metric.set_score_func(
        dataset["metric"] if type(dataset) is dict else getattr(dataset, "metric")
    )
    config.metric.set_loss_func(
        dataset["task"] if type(dataset) is dict else getattr(dataset, "task")
    )

    return dataset


def create_dataloader(dataset, config: Union[CommonArgs, Munch]):
    r"""
    Create a PyG data loader.

    Args:
        loader_name:
        dataset: A GOOD dataset.
        config: Required configs:
            ``config.train.train_bs``
            ``config.train.val_bs``
            ``config.train.test_bs``
            ``config.model.model_layer``
            ``config.train.num_steps(for node prediction)``

    Returns:
        A PyG dataset loader.

    """
    loader_name = config.dataset.dataloader_name
    try:
        reset_random_seed(config)
        loader = register.dataloader[loader_name].setup(dataset, config)
    except KeyError as e:
        print(f"DataLoader {loader_name} not found.")
        raise e

    return loader


def domain_pair_dataloader(dataset, config: Union[CommonArgs, Munch]):
    r"""
    Create a PyG domain_pair data loader.

    Args:
        dataset: A GOOD dataset.
        config: Required configs:
            ``config.train.train_bs``
            ``config.train.val_bs``
            ``config.train.test_bs``
            ``config.model.model_layer``
            ``config.train.num_steps(for node prediction)``

    Returns:
        A PyG domain_pair dataset loader.

    """
    reset_random_seed(config)
    if config.model.model_level == "node":
        graph = dataset[0]
        loader = GraphSAINTRandomWalkSampler(
            graph,
            batch_size=config.train.train_bs,
            walk_length=config.model.model_layer,
            num_steps=config.train.num_steps,
            sample_coverage=100,
            save_dir=dataset.processed_dir,
        )
        loader = {
            "train": loader,
            "eval_train": [graph],
            "id_val": [graph],
            "id_test": [graph],
            "val": [graph],
            "test": [graph],
        }
    else:
        loader = {
            "train": DataLoader(
                dataset["train"], batch_size=config.train.train_bs, shuffle=True
            ),
            "eval_train": DataLoader(
                dataset["train"], batch_size=config.train.val_bs, shuffle=False
            ),
            "id_val": (
                DataLoader(
                    dataset["id_val"], batch_size=config.train.val_bs, shuffle=False
                )
                if dataset.get("id_val")
                else None
            ),
            "id_test": (
                DataLoader(
                    dataset["id_test"], batch_size=config.train.test_bs, shuffle=False
                )
                if dataset.get("id_test")
                else None
            ),
            "val": DataLoader(
                dataset["val"], batch_size=config.train.val_bs, shuffle=False
            ),
            "test": DataLoader(
                dataset["test"], batch_size=config.train.test_bs, shuffle=False
            ),
        }

    return loader
