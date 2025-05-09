from napcas import SGD, Adam

class SGD:
    def __init__(self, params, lr=0.01):
        self.optimizer = SGD(params, lr)

    def step(self):
        self.optimizer.step()

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999)):
        self.optimizer = Adam(params, lr, betas)

    def step(self):
        self.optimizer.step()
        
