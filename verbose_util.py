
def init_verbose(verbose=True):
    if verbose:
        def verboseprint(*args):
            for arg in args:
                print(arg, end=' ')
            print()
    else:
        verboseprint = lambda *a: None

    return verboseprint