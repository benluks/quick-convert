class Indexer:
    def fit(self, dataset):
        raise NotImplementedError


class ResourceIndexer:
    def __init__(self, resource_name: str, provider=None):
        self.resource_name = resource_name
        self.provider = provider

    def fit(self, dataset):
        values = []

        for row in dataset.rows:
            if self.provider is not None:
                ref = self.provider(row)
                values.append(ref.value)
            else:
                ref = row.resources[self.resource_name]
                values.append(ref.value)

        labels = sorted(set(values))

        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

        return self
