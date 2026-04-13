import numpy as np

class Loss:
    def forward(self, Y_pred, Y_true):
        raise NotImplementedError("ERROR: forward failed")
    
    def backward(self):
        raise NotImplementedError("ERROR: backward failed")
    
class MSELoss(Loss):
    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y = Y_true
        N = self.Y.shape[0]
        loss = 1 / N * np.square(self.Y_pred - self.Y)
        return loss
    
    def backward(self):
        N = self.Y.shape[0]
        d_loss = 2 / N * (self.Y_pred - self.Y)
        return d_loss

class MAELoss(Loss):
    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y = Y_true
        N = self.Y.shape[0]
        loss = 1 / N * np.abs(self.Y - self.Y_pred)
        return loss
    
    def backward(self):
        N = self.Y.shape[0]
        d_loss = 1 / N * np.sign(self.Y - self.Y_pred)
        return d_loss

class HuberLoss(Loss):
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.Y = None
        self.Y_pred = None

    def forward(self, Y_pred, Y_true):
        self.Y_pred = Y_pred 
        self.Y = Y_true
        e = self.Y_pred - self.Y
        if np.abs(e) <= self.threshold:
            loss = 0.5 * np.square(e)
        else:
            loss = self.threshold * (np.abs(e) - 0.5 * self.threshold)
        return loss

    def backward(self):
        e = self.Y_pred - self.Y
        if np.abs(e) <= self.threshold:
            d_loss = -e
        else:
            d_loss = -self.threshold * np.sign(e)
        return d_loss  

class FC:
    def __init__(self, input_size, output_size):
        self.cache_X = None

        self.weight = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size), dtype=np.float64)
    
    def forward(self, X):
        self.cache_X = X
        return np.dot(self.cache_X, self.weight) + self.bias
    
    def backward(self, d_out):
        """
        Input: d_out (Đạo hàm từ lớp phía sau truyền tới)
        Output: d_in (Đạo hàm để truyền ngược tiếp cho lớp phía trước)

        dL / dW = dL / dY * dY / dW = X * d_out

        dL / dY = d_loss (MSELoss)
        dY / dW = X

        dL / dX = dL / dY * dY / dX = d_out * W
        """
        self.dW = np.dot(self.cache_X.T, d_out)
        self.db = np.sum(d_out, axis=0, keepdims=True)
        d_in = np.dot(d_out, self.weight.T)
        return d_in
    
class SGD:
    def __init__(self, trainable_layers, learning_rate=0.01):
        """
        trainable_layers: Một list chứa các lớp CÓ TRỌNG SỐ (như FC). 
                          Không đưa ReLU vào đây vì ReLU không có W, b.
        """
        self.layers = trainable_layers
        self.lr = learning_rate
    
    def step(self):
        for layer in self.layers:
            layer.W = layer.W - self.lr * layer.dW
            layer.b = layer.b - self.lr * layer.db

    def zero_grad(self):
        for layer in self.layers:
            layer.dW = 0
            layer.db = 0

class Adam:
    def __init__(self, trainable_layers, learning_rate=0.01, beta_1=0.9, beta_2=0.99):
        self.layers = trainable_layers
        self.lr = learning_rate
        self.b1 = beta_1
        self.b2 = beta_2
    
    def step(self):
        epsilon = 1e-8
        m, m_old, v, v_old = 0
        for idx, layer in enumerate(self.layers):
            m = self.b1 * m_old + (1-self.b1) * layer.dW
            v = self.b2 * v_old + (1-self.b2) * np.square(layer.dW)
            if idx <= 3:
                m = m / (1 - np.power(self.b1, idx))
                v = v / (1 - np.power(self.b2, idx))
            layer.W = layer.W - self.lr * (m / (np.sqrt(v) + epsilon))
            layer.b = layer.b - self.lr * (m / (np.sqrt(v) + epsilon))


  

