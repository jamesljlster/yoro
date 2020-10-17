from pydoc import locate


__all__ = ['load_object']


def load_object(modPath):
    mod = locate(modPath)
    if mod is None:
        raise ModuleNotFoundError(
            'Failed to load object with module path: %s' % modPath)
    else:
        return mod
