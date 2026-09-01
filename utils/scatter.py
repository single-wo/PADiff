try:
    from torch_scatter import scatter_add, scatter_mean, scatter_softmax, scatter_sum
except (ImportError, OSError):
    from torch_geometric.utils import scatter as _scatter
    from torch_geometric.utils import softmax as _softmax

    def scatter_sum(src, index, dim=0, out=None, dim_size=None):
        if out is not None:
            raise NotImplementedError("fallback scatter_sum does not accept out")
        return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")

    scatter_add = scatter_sum

    def scatter_mean(src, index, dim=0, out=None, dim_size=None):
        if out is not None:
            raise NotImplementedError("fallback scatter_mean does not accept out")
        return _scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")

    def scatter_softmax(src, index, dim=-1, dim_size=None):
        return _softmax(src, index=index, num_nodes=dim_size, dim=dim)


__all__ = ["scatter_add", "scatter_mean", "scatter_softmax", "scatter_sum"]
