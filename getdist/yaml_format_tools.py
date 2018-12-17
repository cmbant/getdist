# JT 2017-18

from __future__ import division
from importlib import import_module
from six import string_types
from copy import deepcopy
import re
from collections import OrderedDict as odict
import numpy as np
import yaml

# Conventions
_prior = "prior"
_theory = "theory"
_params = "params"
_likelihood = "likelihood"
_sampler = "sampler"
_p_label = "latex"
_p_dist = "dist"
_p_value = "value"
_p_derived = "derived"
_p_renames = "renames"
_separator = "__"
_minuslogprior = "minuslogprior"
_prior_1d_name = "0"
_chi2 = "chi2"
_weight = "weight"
_minuslogpost = "minuslogpost"


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


def get_info_params(info):
    """
    Extracts parameter info from the new yaml format.
    """
    # Prune fixed parameters
    info_params = info.get(_params)
    info_params_full = odict()
    for p, pinfo in info_params.items():
        # Discard fixed+non-saved parameters
        if is_fixed_param(pinfo) and not is_derived_param(pinfo):
            continue
        info_params_full[p] = info_params[p]
    # Add prior and likelihoods
    info_params_full[_minuslogprior] = {_p_label: r"-\log\pi"}
    for prior in [_prior_1d_name] + list(info.get(_prior, [])):
        info_params_full[_minuslogprior + _separator + prior] = {
            _p_label: r"-\log\pi_\mathrm{" + prior.replace("_", "\ ") + r"}"}
    info_params_full[_chi2] = {_p_label: r"\chi^2"}
    for lik in info.get(_likelihood):
        info_params_full[_chi2 + _separator + lik] = {
            _p_label: r"\chi^2_\mathrm{" + lik.replace("_", "\ ") + r"}"}
    return info_params_full


def get_range(param_info):
    # Sampled
    if is_sampled_param(param_info):
        info_lims = dict([[l, param_info[_prior].get(l)]
                          for l in ["min", "max", "loc", "scale"]])
        if info_lims["min"] is not None or info_lims["max"] is not None:
            lims = [param_info[_prior].get("min"), param_info[_prior].get("max")]
        elif info_lims["loc"] is not None or info_lims["scale"] is not None:
            dist = param_info[_prior].pop(_p_dist, "uniform")
            pdf_dist = getattr(import_module("scipy.stats", dist), dist)
            lims = pdf_dist.interval(1, **param_info[_prior])
    # Derived
    elif is_derived_param(param_info):
        lims = (lambda i: [i.get("min", -np.inf), i.get("max", np.inf)])(param_info or {})
    # Fixed
    else:
        lims = None
    return lims


def is_fixed_param(info_param):
    """
    Returns True if the parameter has been fixed to a value or through a function.
    """
    return expand_info_param(info_param).get(_p_value, None) is not None


def is_sampled_param(info_param):
    """
    Returns True if the parameter has a prior.
    """
    return _prior in expand_info_param(info_param)


def is_derived_param(info_param):
    """
    Returns True if the parameter is saved as a derived one.
    """
    return expand_info_param(info_param).get(_p_derived, False)


def expand_info_param(info_param):
    """
    Expands the info of a parameter, from the user friendly, shorter format
    to a more unambiguous one.
    """
    info_param = deepcopy(info_param)
    if not hasattr(info_param, "keys"):
        if info_param is None:
            info_param = odict()
        else:
            info_param = odict([[_p_value, info_param]])
    if all([(f not in info_param) for f in [_prior, _p_value, _p_derived]]):
        info_param[_p_derived] = True
    # Dynamical input parameters: save as derived by default
    value = info_param.get(_p_value, None)
    if isinstance(value, string_types) or callable(value):
        info_param[_p_derived] = info_param.get(_p_derived, True)
    return info_param


def get_sampler_type(filename_or_info):
    if isinstance(filename_or_info, string_types):
        filename_or_info = yaml_load_file(filename_or_info)
    default_sampler_for_chain_type = "mcmc"
    sampler = list(filename_or_info.get(_sampler, [default_sampler_for_chain_type]))[0]
    return {"mcmc": "mcmc", "polychord": "nested"}[sampler]
