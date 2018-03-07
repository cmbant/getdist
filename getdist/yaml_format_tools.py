# JT 2017

from __future__ import division
from six import string_types, integer_types
from numbers import Number
import re
from collections import OrderedDict as odict

# Conventions
_prior = "prior"


# Exceptions
class InputSyntaxError(Exception):
    """Syntax error in YAML input."""


# Better loader for YAML
# 1. Matches 1e2 as 100 (no need for dot, or sign after e),
#    from http://stackoverflow.com/a/30462009
# 2. Wrapper to load mappings as OrderedDict (for likelihoods and params),
#    from http://stackoverflow.com/a/21912744
def yaml_custom_load(text_stream, Loader=None, object_pairs_hook=odict, file_name=None):
    import yaml
    if Loader is None: Loader = yaml.Loader

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
            pre = ((("\n" + " " * len(signal) + sep).
                join([""] + lines[max(line - 1 - context, 0):line - 1]))) + "\n"
            errorline = (signal + sep + lines[line - 1]
                         + signal_right + "column %s" % column)
            post = ((("\n" + " " * len(signal) + sep).
                join([""] + lines[line + 1 - 1:min(line + 1 + context - 1, len(lines))]))) + "\n"
            raise InputSyntaxError(
                errstr + " at line %d, column %d." % (line, column) + pre + errorline + post +
                "Maybe inconsistent indentation, '=' instead of ':', "
                "no space after ':', or a missing ':' on an empty group?")
        else:
            raise InputSyntaxError(errstr)


def yaml_load_file(input_file):
    """Wrapper to load a yaml file."""
    with open(input_file, "r") as file:
        lines = "".join(file.readlines())
    return yaml_custom_load(lines, file_name=input_file)


# Extracting parameter info from the new yaml format
def load_info_params(fileName):
    info = yaml_load_file(fileName)
    # Flatten theory params, preserving the order, and prune the fixed ones
    info_params = info.get("params")
    info_params_flat = odict()
    for p in info_params:
        if p == "theory":
            for pth in info_params["theory"]:
                info_params_flat[pth] = info_params["theory"][pth]
        else:
            info_params_flat[p] = info_params[p]
        # fixed? discard
        last = info_params_flat.keys()[-1]
        if isinstance(last, float) or isinstance(last, integer_types):
            info_params_flat.pop(last)
    # Now add prior and likelihoods
    info_params_flat["minuslogprior"] = {"latex": r"-\log\pi"}
    info_params_flat["chi2"] = {"latex": r"\chi^2"}
    for lik in info.get("likelihood"):
        info_params_flat["chi2_" + lik] = {
            "latex": r"\chi^2_\mathrm{" + lik.replace("_", "\ ") + r"}"}
    return info_params_flat


def is_fixed_param(info_param):
    """
    Returns True if `info_param` is a number, a string or a function.
    """
    return (isinstance(info_param, Number) or isinstance(info_param, string_types) or
            callable(info_param))


def is_sampled_param(info_param):
    """
    Returns True if `info_param` has a `%s` field.
    """ % _prior
    return _prior in (info_param if hasattr(info_param, "get") else {})


def is_derived_param(info_param):
    """
    Returns False if `info_param` is "fixed" or "sampled".
    """
    return not (is_fixed_param(info_param) or is_sampled_param(info_param))
