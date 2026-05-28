class BaseResourceProvider:
    """
    An abstracton class for resource providers, which are responsible for providing access to various types of
    resources (e.g. annotation files, precompute feature files, etc.) associated with samples in a dataset.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample):
        raise NotImplementedError
