from napcas import DataLoader

class DataLoader:
    def __init__(self, dataset_path, batch_size):
        self.loader = DataLoader(dataset_path, batch_size)

    def next(self):
        return self.loader.next()
