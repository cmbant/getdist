# JT 2017-19

from __future__ import division
from importlib import import_module
from six import string_types
from copy import deepcopy
from collections import OrderedDict as odict
import numpy as np

# Conventions
_label = "label"
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
_separator_files = "."
_minuslogprior = "minuslogprior"
_prior_1d_name = "0"
_chi2 = "chi2"
_weight = "weight"
_minuslogpost = "minuslogpost"
_post = "post"


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
    priors = [_prior_1d_name] + list(info.get(_prior, []))
    likes = list(info.get(_likelihood))
    # Account for post
    remove = info.get(_post, {}).get("remove", {})
    for param in remove.get(_params, []) or []:
        info_params_full.pop(param, None)
    for like in remove.get(_likelihood, []) or []:
        likes.remove(like)
    for prior in remove.get(_prior, []) or []:
        priors.remove(prior)
    add = info.get(_post, {}).get("add", {})
    # Adding derived params and updating 1d priors
    for param, pinfo in add.get(_params, {}).items():
        pinfo_old = info_params_full.get(param, {})
        pinfo_old.update(pinfo)
        info_params_full[param] = pinfo_old
    likes += list(add.get(_likelihood, []))
    priors += list(add.get(_prior, []))
    # Add the prior and the likelihood as derived parameters
    info_params_full[_minuslogprior] = {_p_label: r"-\log\pi"}
    for prior in priors:
        info_params_full[_minuslogprior + _separator + prior] = {
            _p_label: r"-\log\pi_\mathrm{" + prior.replace("_", r"\ ") + r"}"}
    info_params_full[_chi2] = {_p_label: r"\chi^2"}
    for like in likes:
        info_params_full[_chi2 + _separator + like] = {
            _p_label: r"\chi^2_\mathrm{" + like.replace("_", r"\ ") + r"}"}
    return info_params_full


def get_range(param_info):
    # Sampled
    if is_sampled_param(param_info):
        info_lims = dict([[l, param_info[_prior].get(l)] for l in ["min", "max", "loc", "scale"]])
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
        from getdist.yaml_tools import yaml_load_file
        filename_or_info = yaml_load_file(filename_or_info)
    default_sampler_for_chain_type = "mcmc"
    sampler = list(filename_or_info.get(_sampler, [default_sampler_for_chain_type]))[0]
    return {"mcmc": "mcmc", "polychord": "nested", "minimize": "minimize"}[sampler]


def get_sample_label(filename_or_info):
    if isinstance(filename_or_info, string_types):
        from getdist.yaml_tools import yaml_load_file
        filename_or_info = yaml_load_file(filename_or_info)
    return filename_or_info.get(_label, None)
