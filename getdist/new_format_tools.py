# Better loader for YAML
# 1. Matches 1e2 as 100 (no need for dot, or sign after e),
#    from http://stackoverflow.com/a/30462009
# 2. Wrapper to load mappings as OrderedDict (for likelihoods and params),
#    from http://stackoverflow.com/a/21912744

import yaml
import re
from collections import OrderedDict as odict
import six

def yaml_custom_load(stream, Loader=yaml.Loader, object_pairs_hook=odict,
                        file_name=None):
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
            return yaml.load(stream, OrderedLoader)
        # Redefining the general exception to give more user-friendly information
        except yaml.YAMLError, exception:
            errstr = "Error in your input file "+("'"+file_name+"'" if file_name else "")
            if hasattr(exception, "problem_mark"):
                raise yaml.YAMLError(errstr + " at line %d, column %d. "%(
                    1+exception.problem_mark.line, 1+exception.problem_mark.column)+
                    "Maybe inconsistent indentation, '=' instead of ':', "+
                    "or a missing ':' on an empty group?")
            else:
                raise yaml.YAMLError(errstr)


# Extracting parameter info from the new yaml format
def load_info_params(fileName):
    with open(fileName) as f:
        info = yaml_custom_load(f)
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
        if isinstance(last, float) or isinstance(last, six.integer_types):
            info_params_flat.pop(last)
    # Now add prior and likelihoods
    info_params_flat["minuslogprior"] = {"latex": r"-\log\pi"}
    info_params_flat["chi2"] = {"latex": r"\chi^2"}
    for lik in info.get("likelihood"):
        info_params_flat["chi2_"+lik] = {
            "latex": r"\chi^2_\mathrm{"+lik.replace("_","\ ")+r"}"}
    return info_params_flat
