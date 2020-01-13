import os
from getdist.inifile import IniFile


def file_root_to_root(root):
    return (os.path.basename(root) if not root.endswith((os.sep, "/"))
            else os.path.basename(root[:-1]) + os.sep)


class ChainItem(object):
    # Basic object for chain on disk, basic duck-type compatibility with JobItem in more complex BatchJob grids
    def __init__(self, batchPath, chainRoot, paramtag, name=None):
        self.batchPath = batchPath  # root directory of grid structure
        self.chainRoot = chainRoot  # full path and root name of chain
        self.paramtag = paramtag  # lowest level folder name under batchPath
        self.name = name or os.path.basename(chainRoot)
        self.chainPath = os.path.dirname(chainRoot)


class ChainDirGrid(object):
    # Basic grid, just all chains under a given folder (including nested subfolders)
    # getdist.ini in the base directory can specify default getdist analysis settings for all chains in the folders.
    # Chains are indexed by their root name, which includes as many leading subdirectories as needed to be unique
    # getdist.plots an getdist.MCSamples compatible with the BatchJob complex grid objects
    def __init__(self, base):
        self.batchPath = base
        self.roots = {}
        self.base_dir_names = set()
        option_file = os.path.join(base, 'getdist.ini')
        if os.path.exists(option_file):
            self.getdist_options = IniFile(option_file).params
        else:
            self.getdist_options = {}

    def add(self, dir_tag, dirname, roots):
        self.base_dir_names.add(dir_tag)
        for root in roots:
            root = file_root_to_root(root)
            self.roots[root] = self.roots.get(root, []) + [
                ChainItem(self.batchPath, os.path.join(dirname, root), dir_tag, root)]

    def make_unique(self):
        for root in list(self.roots):
            if len(self.roots[root]) > 1:
                paths = [item.chainRoot.split(os.sep) for item in self.roots[root]]
                i = -2
                while all(s[i] == paths[0][i] for s in paths[1:]):
                    i -= 1
                for parts, item in zip(paths, self.roots[root]):
                    item.name = '/'.join(parts[i:])
                    item.chainPath = os.sep.join(parts[:i])
                    self.roots[item.name] = item
                self.roots.pop(root)
            else:
                self.roots[root] = self.roots[root][0]

    def roots_for_dir(self, paramtag):
        roots = []
        for item in self.roots.values():
            if item.paramtag == paramtag:
                roots.append(item.name)
        return roots

    def resolveRoot(self, root):
        return self.roots.get(root)
