"""Core dictionary-like structures used to mirror MATLAB structs."""


class AttrDict(dict):
    """Dictionary with MATLAB-like attribute access.

    The original MATLAB code passes around nested structs and adds fields
    dynamically. This class keeps the same style while remaining a plain
    dict for JSON and NumPy persistence.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def as_attrdict(value):
    """Recursively convert mappings to AttrDict instances."""
    if isinstance(value, dict):
        out = AttrDict()
        for key, item in value.items():
            out[key] = as_attrdict(item)
        return out
    if isinstance(value, list):
        return [as_attrdict(item) for item in value]
    return value


def to_plain(value):
    """Recursively convert AttrDict instances to plain Python objects."""
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value
