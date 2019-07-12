# JT 2017-19

from __future__ import division
import re
from collections import OrderedDict as odict
import six

if six.PY2:
    ModuleNotFoundError = ImportError
try:
    import yaml
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "You need to install 'PyYAML' in order to load Cobaya samples.")


# Exceptions
class InputSyntaxError(Exception):
    """Syntax error in YAML input."""


# Better loader for YAML
# 1. Matches 1e2 as 100 (no need for dot, or sign after e),
#    from http://stackoverflow.com/a/30462009
# 2. Wrapper to load mappings as OrderedDict (for likelihoods and params),
#    from http://stackoverflow.com/a/21912744
def yaml_load(text_stream, Loader=yaml.Loader, object_pairs_hook=odict, file_name=None):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    OrderedLoader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    # Ignore python objects
    def dummy_object_loader(loader, suffix, node):
        return None

    OrderedLoader.add_multi_constructor(
        u'tag:yaml.org,2002:python/name:', dummy_object_loader)
    try:
        return yaml.load(text_stream, OrderedLoader)
    # Redefining the general exception to give more user-friendly information
    except yaml.YAMLError as exception:
        errstr = "Error in your input file " + ("'" + file_name + "'" if file_name else "")
        if hasattr(exception, "problem_mark"):
            line = 1 + exception.problem_mark.line
            column = 1 + exception.problem_mark.column
            signal = " --> "
            signal_right = "    <---- "
            sep = "|"
            context = 4
            lines = text_stream.split("\n")
            pre = ((("\n" + " " * len(signal) + sep).join(
                [""] + lines[max(line - 1 - context, 0):line - 1]))) + "\n"
            errorline = (signal + sep + lines[line - 1] +
                         signal_right + "column %s" % column)
            post = ((("\n" + " " * len(signal) + sep).join(
                [""] + lines[line + 1 - 1:min(line + 1 + context - 1, len(lines))]))) + "\n"
            raise InputSyntaxError(
                errstr + " at line %d, column %d." % (line, column) +
                pre + errorline + post +
                "Maybe inconsistent indentation, '=' instead of ':', "
                "no space after ':', or a missing ':' on an empty group?")
        else:
            raise InputSyntaxError(errstr)


def yaml_load_file(input_file):
    """Wrapper to load a yaml file."""
    with open(input_file, "r") as f:
        lines = "".join(f.readlines())
    return yaml_load(lines, file_name=input_file)
