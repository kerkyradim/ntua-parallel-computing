import time

import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from utilities import CustomMatmul

class MLP:

    def __init__(self, din, dout, matMullMode: str = 'numpy', numberOfProcesses: int = 1):
        self.Type = f'MLP {din}->{dout}'
        self.deltab = None
        self.deltaW = None
        self.x = None
        np.random.seed(8)
        self.W = (2 * np.random.rand(dout, din) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        np.random.seed(8)
        self.b = (2 * np.random.rand(dout) - 1) * (np.sqrt(6) / np.sqrt(din + dout))
        self.matMullMode = matMullMode
        self.numberOfProcesses = numberOfProcesses
        self.transfersTime = 0

    def forward(self, x):  # x.shape = (batch_size, din)
        self.x = x  # Storing x for latter (backward pass)
        return x @ self.W.T + self.b

    def backward(self, gradout):
        # print(f"{self.Type}, gradout.T @ self.x: {gradout.T.shape} @ {self.x.shape}\n")
        self.deltaW, tempTransfersTime = CustomMatmul(gradout.T, self.x, mode=self.matMullMode, numberOfProcesses=self.numberOfProcesses)
        self.transfersTime += tempTransfersTime
        self.deltab = gradout.sum(0)
        # print(f"{self.Type}, gradout, self.W: {gradout.shape} @ {self.W.shape}\n")
        out, tempTransfersTime = CustomMatmul(gradout, self.W, mode=self.matMullMode, numberOfProcesses=self.numberOfProcesses)
        self.transfersTime += tempTransfersTime
        return out


class SequentialNN:

    def __init__(self, blocks: list):
        self.blocks = blocks
        self.layerTimers = {}

        for iBlock in blocks:
            self.layerTimers[iBlock.Type] = 0

    def forward(self, x):

        for block in self.blocks:
            x = block.forward(x)

        return x

    def backward(self, gradout):

        for idx, block in enumerate(self.blocks[::-1]):
            tic = time.perf_counter()
            gradout = block.backward(gradout)
            toc = time.perf_counter()
            self.layerTimers[block.Type] += toc - tic

        return gradout

    def getCUDATransferTimes(self):
        totalTransfersTime = 0
        for idx, block in enumerate(self.blocks[::-1]):
            totalTransfersTime += block.transfersTime
        return totalTransfersTime


    def printTimers(self):
        for iLayer, iTime in self.layerTimers.items():
            print(f"{iLayer}:\t\t{iTime}")


class ReLU:

    def __init__(self):
        self.x = None
        self.Type = f'ReLU'
        self.transfersTime = 0

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, gradout):
        new_grad = gradout.copy()
        new_grad[self.x < 0] = 0.
        return new_grad


class LogSoftmax:

    def __init__(self, matMullMode: str = 'numpy'):
        self.Type = f'LogSoftmax'
        self.x = None
        self.matMullMode = matMullMode
        self.transfersTime = 0

    def forward(self, x):
        self.x = x
        return x - logsumexp(x, axis=1)[..., None]

    def backward(self, gradout):
        gradients = np.eye(self.x.shape[1])[None, ...]
        gradients = gradients - (np.exp(self.x) / np.sum(np.exp(self.x), axis=1)[..., None])[..., None]
        # print(f"{self.Type}, gradients @ gradout[..., None]: {gradients.shape} @ {gradout[..., None].shape}\n")
        # out, tempTransfersTime = CustomMatmul(gradients, gradout[..., None], mode=self.matMullMode)
        # self.transfersTime += tempTransfersTime
        out = np.matmul(gradients, gradout[..., None])
        return (out)[:, :, 0]

class NLLLoss:

    def __init__(self):
        self.true = None
        self.pred = None
        self.transfersTime = 0

    def forward(self, pred, true):
        self.pred = pred
        self.true = true

        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b, int(true[b])]
        return loss

    def backward(self):
        din = self.pred.shape[1]
        jacobian = np.zeros((self.pred.shape[0], din))
        for b in range(self.pred.shape[0]):
            jacobian[b, int(self.true[b])] = -1

        return jacobian  # batch_size x din

    def __call__(self, pred, true):
        return self.forward(pred, true)


class Optimizer:

    def __init__(self, lr, compound_nn: SequentialNN):
        self.lr = lr
        self.compound_nn = compound_nn
        self.transfersTime = 0

    def step(self):

        for block in self.compound_nn.blocks:
            if block.__class__ == MLP:
                block.W = block.W - self.lr * block.deltaW
                block.b = block.b - self.lr * block.deltab


def train(model, optimizer, trainX, trainy, loss_fct=NLLLoss(), nb_epochs=5, batch_size=2048):
    training_loss = []
    # for epoch in range(nb_epochs):
    for epoch in range(nb_epochs):
        # Sample batch size
        np.random.seed(8)
        batch_idx = [np.random.randint(0, trainX.shape[0]) for _ in range(batch_size)]
        x = trainX[batch_idx]
        target = trainy[batch_idx]

        prediction = model.forward(x)  # Forward pass
        loss_value = loss_fct(prediction, target)  # Compute the loss
        training_loss.append(loss_value)  # Log loss
        gradout = loss_fct.backward()
        model.backward(gradout)  # Backward pass

        # Update the weights
        optimizer.step()
    return training_loss
