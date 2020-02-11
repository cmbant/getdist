import os
import glob
from getdist.inifile import IniFile


def file_root_to_root(root):
    return (os.path.basename(root) if not root.endswith((os.sep, "/"))
            else os.path.basename(root[:-1]) + os.sep)


def get_chain_root_files(rootdir):
    """
    Gets the root names of all chain files in a directory.

    :param rootdir: The root directory to check
    :return:  The root names
    """
    from getdist.chains import hasChainFiles
    pattern = os.path.join(rootdir, '*.paramnames')
    files = [os.path.splitext(f)[0] for f in glob.glob(pattern)]
    ending = 'updated.yaml'
    pattern = os.path.join(rootdir, "*" + ending)
    files += [f[:-len(ending)].rstrip("_.") for f in glob.glob(pattern)]
    files = [f for f in files if hasChainFiles(os.path.join(rootdir, f))]
    files.sort()
    return files


def is_grid_object(obj):
    return hasattr(obj, "resolve_root") or hasattr(obj, "resolveRoot")


# noinspection PyUnresolvedReferences
def load_supported_grid(chain_dir):
    if is_grid_object(chain_dir):
        return chain_dir
    try:
        # If cosmomc is installed, e.g. to allow use of old Planck grids
        # The 2018 final Planck grid should be OK with getdist default chain grid below
        from paramgrid import gridconfig, batchjob
        if gridconfig.pathIsGrid(chain_dir):
            return batchjob.readobject(chain_dir)
    except ImportError:
        pass
    return None


class ChainItem:
    # Basic object for chain on disk, basic duck-type compatibility with JobItem in more complex BatchJob grids
    def __init__(self, batchPath, chainRoot, paramtag, name=None):
        self.batchPath = batchPath  # root directory of grid structure
        self.chainRoot = chainRoot  # full path and root name of chain
        self.paramtag = paramtag  # lowest level folder name under batchPath
        self.name = name or os.path.basename(chainRoot)
        self.chainPath = os.path.dirname(chainRoot)


class ChainDirGrid:
    # Basic grid, just all chains under a given folder (including nested subfolders)
    # getdist.ini in the base directory can specify default getdist analysis settings for all chains in the folders.
    # Chains are indexed by their root name, which includes as many leading subdirectories as needed to be unique
    # getdist.plots and getdist.MCSamples compatible with the paramgrid.BatchJob complex grid objects of cosmomc
    def __init__(self, base):
        self.batchPath = base
        self.roots = {}
        self.base_dir_names = set()
        self._sorted_names = {}
        option_file = os.path.join(base, 'getdist.ini')
        if os.path.exists(option_file):
            self.getdist_options = IniFile(option_file).params
        else:
            self.getdist_options = {}
        for base, dirs, files in os.walk(base):
            for _dir in dirs:
                files = get_chain_root_files(os.path.join(base, _dir))
                if files:
                    self._add(_dir, os.path.join(base, _dir), files)
                for base_rel, dirs_rel, files_rel in os.walk(os.path.join(base, _dir)):
                    for _subdir in dirs_rel:
                        files = get_chain_root_files(os.path.join(base_rel, _subdir))
                        if files:
                            self._add(_dir, os.path.join(base_rel, _subdir), files)
            break
        self._make_unique()

    def normed_name(self, root):
        return '_'.join(sorted(root.replace('__', '_').replace('_post', '').split('_')))

    def _add(self, dir_tag, dirname, roots):
        self.base_dir_names.add(dir_tag)
        for root in roots:
            root = file_root_to_root(root)
            self.roots[root] = self.roots.get(root, []) + [
                ChainItem(self.batchPath, os.path.join(dirname, root), dir_tag, root)]

    def _make_unique(self):
        for root in list(self.roots):
            normed_name = self.normed_name(root)
            self._sorted_names[normed_name] = self._sorted_names.get(normed_name, []) + self.roots[root]
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
        return self.resolve_root(root)

    def resolve_root(self, root):
        item = self.roots.get(root)
        if not item:
            normed_name = self.normed_name(root)
            items = self._sorted_names.get(normed_name)
            if items:
                if len(items) == 1:
                    return items[0]
                raise ValueError('No exact march for %s and normalized name %s is ambiguous: %r' %
                                 (root, normed_name, [i.chainRoot for i in items]))
        return item
