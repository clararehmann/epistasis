__description__ = \
"""
Utilities useful for model construction from abstract classes and mixins.
"""
__author__ = "Zach Sailer"

import inspect
from functools import wraps

def arghandler(method):
    """
    Points methods to argument handlers. Assumes each argument has a
    corresponding method attached to the object named "_{argument}". These
    methods given default values to arguments.

    Ignores self and kwargs
    """
    @wraps(method)
    def inner(self, *args, **kwargs):
        # Get method name
        name = method.__name__

        # Inspect function for arguments to update.
        out = inspect.signature(method)

        # Construct kwargs from signature.
        kws = {key: val.default for key, val in out.parameters.items()}
        kws.pop('self')

        # Try to remove kwargs.
        try:
            kws.pop('kwargs')
        except:
            pass

        # Update kwargs with user specified kwargs.
        kws.update(**kwargs)

        # Handle each argument
        for arg in kws:
            # Get handler function.
            handler_name = "_{}".format(arg)
            handler = getattr(self, handler_name)
            kws[arg] = handler(data=kws[arg], method=name)

        return method(self, **kws)
    return inner
