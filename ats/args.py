import argparse
from functools import cache

@cache
def make_forward_option(state):
    class ForwardOption(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            s = state + "_pending"
            # technically can use hasattr here
            # but to match it to InputOption
            if getattr(namespace, s, None) is None:
                setattr(namespace, s, {})
            getattr(namespace, s)[self.dest] =  values
    return ForwardOption


@cache
def make_input_option(state):
    class InputOption(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # can't use hasattr heere as the self.dest is already set to None by argparse
            if getattr(namespace, self.dest, None) is None:
                setattr(namespace, self.dest, [])

            pending = getattr(namespace, state+"_pending", {})
            global_pending = getattr(namespace, "_pending", {})
            getattr(namespace, self.dest).append({
                "file": values,
                **pending,
                **global_pending
                })
            pending.clear()
            global_pending.clear()
    return InputOption
