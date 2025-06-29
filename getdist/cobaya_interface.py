# JT 2017-19

import logging
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from importlib import import_module
from numbers import Number

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
_minuslogprior = "minuslogprior"
_prior_1d_name = "0"
_chi2 = "chi2"
_weight = "weight"
_minuslogpost = "minuslogpost"
_post = "post"


def cobaya_params_file(root):
    file = root + ("" if root.endswith((os.sep, "/")) else ".") + "updated.yaml"
    if os.path.exists(file):
        return file
    else:
        file = root + ("" if root.endswith((os.sep, "/")) else "__") + "full.yaml"
        if os.path.exists(file):
            return file
    return None


def yaml_file_or_dict(file_or_dict) -> Mapping:
    if isinstance(file_or_dict, str):
        from getdist.yaml_tools import yaml_load_file

        return yaml_load_file(file_or_dict)
    elif isinstance(file_or_dict, Mapping):
        return file_or_dict
    else:
        raise ValueError("Cobya parameter input must be a dictionary or filename")


def MCSamplesFromCobaya(info, collections, name_tag=None, ignore_rows=0, ini=None, settings=None):
    """
    Creates a set of samples from Cobaya's output.
    Parameter names, ranges and labels are taken from the "info" dictionary
    (always use the "updated" one generated by `cobaya.run`).

    For a description of the various analysis settings and default values see
    `analysis_defaults.ini <https://getdist.readthedocs.io/en/latest/analysis_settings.html>`_.

    :param collections: collection(s) of samples from Cobaya
    :param info: info dictionary, common to all collections
                 (use the "updated" one, returned by `cobaya.run`)
    :param name_tag: name for this sample to be shown in the plots' legend
    :param ignore_rows: initial samples to skip, number (`int>=1`) or fraction (`float<1`)
    :param ini: The name of a .ini file with analysis settings to use
    :param settings: dictionary of analysis settings to override defaults
    :return: The :class:`MCSamples` instance
    """

    if hasattr(collections, "data"):
        collections = [collections]
    # Check consistency between collections
    try:
        columns = list(collections[0].data)
    except AttributeError:
        raise TypeError("The second argument does not appear to be a (list of) samples `Collection`.")
    if not all(list(c.data) == columns for c in collections[1:]):
        raise ValueError("The given collections don't have the same columns.")
    # Check consistency with info
    info_params = get_info_params(info)
    # if skip burn in *has already been done*
    skip = info.get(_post, {}).get("skip", 0)
    if ignore_rows != 0 and skip != 0:
        logging.warning(
            "You are asking for rows to be ignored (%r), but some (%r) were already ignored in the original chain.",
            ignore_rows,
            skip,
        )
    var_params = [k for k, v in info_params.items() if is_sampled_param(v) or is_derived_param(v)]
    assert set(columns[2:]) == set(var_params), (
        "Info and collection(s) are not compatible, because their parameters differ: "
        "the collection(s) have %r and the info has %r. "
        % (columns[2:], var_params)
        + "Are you sure that you are using an *updated* info dictionary "
        "(i.e. the output of `cobaya.run`)?"
    )
    # We need to use *collection* sorting, not info sorting!
    names = [p + ("*" if is_derived_param(info_params[p]) else "") for p in columns[2:]]
    labels = [(info_params[p] or {}).get(_p_label, p) for p in columns[2:]]
    ranges = {p: get_range(info_params[p]) for p in info_params}  # include fixed parameters not in columns
    renames = {p: info_params.get(p, {}).get(_p_renames, []) for p in columns[2:]}
    samples = [c[c.data.columns[2:]].values.astype(np.float64) for c in collections]
    weights = [c[_weight].values.astype(np.float64) for c in collections]
    loglikes = [c[_minuslogpost].values.astype(np.float64) for c in collections]
    sampler = get_sampler_type(info)
    temperature = get_sampler_temperature(info)
    label = get_sample_label(info)
    if temperature is not None and temperature != 1:
        logging.warning(
            "You have loaded a sample with non-unit temperature. "
            "Use the 'MCSamples.cool()' method to turn it into a sample from "
            "the original posterior before performing statistical analyses, "
            "but maybe after thinning the sample with method "
            "'MCSamples.thin_indices()'."
        )
    from getdist.mcsamples import MCSamples

    return MCSamples(
        samples=samples,
        weights=weights,
        loglikes=loglikes,
        sampler=sampler,
        names=names,
        labels=labels,
        ranges=ranges,
        renames=renames,
        ignore_rows=ignore_rows,
        name_tag=name_tag,
        label=label,
        ini=ini,
        temperature=temperature,
        settings=settings,
    )


def str_to_list(x):
    return [x] if isinstance(x, str) else x


def get_info_params(info):
    """
    Extracts parameter info from the new yaml format.
    """
    info = yaml_file_or_dict(info)
    # Prune fixed parameters
    info_params = info.get(_params)
    info_params_full = dict()
    for p, pinfo in info_params.items():
        info_params_full[p] = info_params[p]
    # Add prior and likelihoods
    priors = [_prior_1d_name] + list(info.get(_prior) or [])
    likes = list(info.get(_likelihood))
    # Account for post
    remove = info.get(_post, {}).get("remove", {})
    for param in remove.get(_params, []) or []:
        info_params_full.pop(param, None)
    for like in str_to_list(remove.get(_likelihood) or []):
        likes.remove(like)
    for prior in str_to_list(remove.get(_prior)) or []:
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
            _p_label: r"-\log\pi_\mathrm{" + prior.replace("_", r"\ ") + r"}"
        }
    info_params_full[_chi2] = {_p_label: r"\chi^2"}
    for like in likes:
        info_params_full[_chi2 + _separator + like] = {_p_label: r"\chi^2_\mathrm{" + like.replace("_", r"\ ") + r"}"}
    return info_params_full


# noinspection PyUnboundLocalVariable
def get_range(param_info):
    # Sampled
    if is_sampled_param(param_info):
        prior = param_info[_prior]
        if isinstance(prior, Sequence) and len(prior) == 2:
            prior = {lim: n for lim, n in zip(["min", "max"], prior)}
        elif not isinstance(prior, Mapping):
            raise ValueError(
                "Format of prior not recognised: %r. " % prior
                + "Use '[min, max]' or a dictionary following Cobaya's documentation."
            )
        info_lims = {tag: prior.get(tag) for tag in ["min", "max", "loc", "scale"]}
        if info_lims["min"] is not None or info_lims["max"] is not None:
            lims = [prior.get("min"), prior.get("max")]
        elif info_lims["loc"] is not None or info_lims["scale"] is not None:
            args = prior.copy()
            dist = args.pop(_p_dist, "uniform")
            pdf_dist = getattr(import_module("scipy.stats", dist), dist)
            lims = pdf_dist.interval(1, **args)
    # Derived
    elif is_derived_param(param_info):
        lims = (lambda i: [i.get("min", -np.inf), i.get("max", np.inf)])(param_info or {})
    # Fixed
    else:
        value = fixed_value(param_info)
        try:
            value = float(value)
        except (ValueError, TypeError):
            # e.g. lambda function values
            lims = (lambda i: [i.get("min", -np.inf), i.get("max", np.inf)])(param_info or {})
        else:
            lims = (value, value)
    return lims[0] if lims[0] != -np.inf else None, lims[1] if lims[1] != np.inf else None


def fixed_value(info_param):
    """
    Returns True if the parameter has been fixed to a value or through a function.
    """
    return expand_info_param(info_param).get(_p_value, None)


def is_fixed_param(info_param):
    """
    Returns True if the parameter has been fixed to a value or through a function.
    """
    return fixed_value(info_param) is not None


def is_parameter_with_range(info_param):
    value = fixed_value(info_param)
    return value is None or isinstance(value, Number) or is_derived_param(info_param)


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
    Expands the info of a parameter, from the user-friendly, shorter format
    to a more unambiguous one.
    """
    if not isinstance(info_param, Mapping):
        if info_param is None:
            info_param = {}
        else:
            info_param = {_p_value: info_param}
    else:
        info_param = deepcopy(info_param)
    if all((f not in info_param) for f in [_prior, _p_value, _p_derived]):
        info_param[_p_derived] = True
    # Dynamical input parameters: save as derived by default
    value = info_param.get(_p_value, None)
    if isinstance(value, str) or callable(value):
        info_param[_p_derived] = info_param.get(_p_derived, True)
    return info_param


def get_sampler_key(filename_or_info, default_sampler_for_chain_type="mcmc"):
    return list(yaml_file_or_dict(filename_or_info).get(_sampler, [default_sampler_for_chain_type]))[0]


def get_sampler_type(filename_or_info, default_sampler_for_chain_type="mcmc"):
    sampler = get_sampler_key(filename_or_info, default_sampler_for_chain_type)
    return "nested" if sampler == "polychord" else sampler


def get_sampler_temperature(filename_or_info):
    info = yaml_file_or_dict(filename_or_info)
    if _sampler not in info:
        return None
    # post-processed chains have always already been cooled
    if _post in info:
        return 1
    return (info[_sampler][get_sampler_key(info)] or {}).get("temperature")


def get_sample_label(filename_or_info):
    return yaml_file_or_dict(filename_or_info).get(_label)


def get_burn_removed(filename_or_info):
    info = get_info_params(filename_or_info)
    # if skip burn in *has already been done*
    return info.get(_post, {}).get("skip", 0)
