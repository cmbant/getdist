# This provides base classes for handling backwards compatibility with renamed or removed attributes

import re
import logging


def _convert_camel(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def _map_name(obj, name):
    try:
        return object.__getattribute__(obj, name), name
    except AttributeError:
        pass
    _old = obj.__class__.__dict__.get('_deprecated')
    if _old and name in _old:
        newname = _old.get(name)
        if newname is None:
            return None, None
    else:
        newname = _convert_camel(name)
    try:
        return object.__getattribute__(obj, newname), newname
    except AttributeError:
        return None


class _BaseObject(object):
    # Compatibility of pep_8_style and camelCase for backwards compatibility

    _fail_on_not_exist = False

    def __getattribute__(self, name):
        if name.startswith('__'):
            return object.__getattribute__(self, name)
        res = _map_name(self, name)
        if res is None:
            raise AttributeError('%s is not a valid attribute for class %s' % (name, self.__class__))
        value, newname = res
        if newname is None:
            logging.warning("%s is removed and will be ignored" % name)
            return None
        if newname is not name:
            logging.warning("%s is deprecated, use %s" % (name, newname))
        return value

    def __setattr__(self, name, value):

        res = _map_name(self, name)
        if res is None:
            if object.__getattribute__(self, '_fail_on_not_exist'):
                raise AttributeError('Unknown attribute %s for class %s' % (name, self.__class__))
            newname = name
        else:
            _, newname = res
        if newname is None:
            logging.warning("%s is removed and will be ignored" % name)
            return
        object.__setattr__(self, newname, value)
