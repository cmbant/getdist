# Better loader and dumper for YAML:
# 1. Matches 1e2 as 100 (no need for dot, or sign after e),
#    from http://stackoverflow.com/a/30462009
# 2. Wrapper to load mappings as OrderedDict (for likelihoods and params),
#    from http://stackoverflow.com/a/21912744
import yaml
import re
from collections import OrderedDict as odict

def yaml_odict_sci_load(stream, Loader=yaml.Loader, object_pairs_hook=odict,
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
        except yaml.YAMLError, exception:
            errstr = "Error in your input file "+("'"+file_name+"'" if file_name else "")
            if hasattr(exception, "problem_mark"):
                raise yaml.YAMLError(errstr + " at line %d, column %d. "%(
                    1+exception.problem_mark.line, 1+exception.problem_mark.column)+
                    "Maybe inconsistent indentation or a missing colon on an empty group?")
            else:
                raise yaml.YAMLError(errstr++str(err))
