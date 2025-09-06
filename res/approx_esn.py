

import torch
import torch.nn as nn
import torch.optim as optim

class ApproxEuSN2EcSN:
    def __init__(self, W_in, W_h, b, epsilon, gamma, N_samples=1000, N_epochs=500):
        self.W_in = W_in
        self.W_h = W_h
        self.b = b
        self.n = W_in.shape[0]
        self.l = W_in.shape[1]
        self.epsilon = epsilon
        self.gamma = gamma
        self.N_samples = N_samples
        self.N_epochs = N_epochs
        self.W_in_p = None
        self.W_h_p = None
        self.b_p = None
        self.x_data = None
        self.h_data = None
        self.y0 = None
        self.y1 = None
        self.loss_mean = None
        self.correlation_mean = None
        # W_h, b のサイズチェック
        assert W_h.shape == (self.n, self.n), "W_h must be of shape (n, n)"
        assert b.shape == (self.n, 1), "b must be of shape (n, 1)"

    def _correlation_coefficient(self, y_true, y_pred):
        y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
        y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
        corr_num = (y_true_centered * y_pred_centered).sum(dim=0)
        corr_den = torch.sqrt((y_true_centered**2).sum(dim=0) * (y_pred_centered**2).sum(dim=0) + 1e-8)
        correlation = (corr_num / corr_den)
        return correlation.mean()

    def euler_hidden(self, W_in, W_h, b, x, h):
        I = torch.eye(self.n)
        return h + self.epsilon * torch.tanh(W_in @ x + (W_h - self.gamma * I) @ h + b)

    def echo_hidden(self, W_in, W_h, b, x, h):
        return torch.tanh(W_in @ x + W_h @ h + b)

    def fit(self):
        # ダミーデータ生成
        self.x_data = torch.randn(self.N_samples, self.l, 1)
        self.h_data = torch.randn(self.N_samples, self.n, 1)
        # y0計算
        y0_list = [self.euler_hidden(self.W_in, self.W_h, self.b, x, h).squeeze(-1)
                   for x, h in zip(self.x_data, self.h_data)]
        self.y0 = torch.stack(y0_list)
        # 学習パラメータ初期化
        self.W_in_p = nn.Parameter(torch.randn(self.n, self.l))
        self.W_h_p = nn.Parameter(torch.randn(self.n, self.n))
        self.b_p = nn.Parameter(torch.randn(self.n, 1))
        params = [self.W_in_p, self.W_h_p, self.b_p]
        optimizer = optim.Adam(params, lr=1e-2)
        # 学習ループ
        for epoch in range(self.N_epochs):
            optimizer.zero_grad()
            y1_list = [self.echo_hidden(self.W_in_p, self.W_h_p, self.b_p, x, h).squeeze(-1)
                       for x, h in zip(self.x_data, self.h_data)]
            self.y1 = torch.stack(y1_list)
            loss = (-1) * self._correlation_coefficient(self.y0, self.y1)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Corr: {-loss.item():.4f}")
        self.loss_mean = (-1) * self._correlation_coefficient(self.y0, self.y1).item()
        self.correlation_mean = self._correlation_coefficient(self.y0, self.y1).item()
        # print(f"[Training] Loss: {self.loss_mean:.4f}, Corr: {self.correlation_mean:.4f}")

    def evaluate(self):
        # 新しいダミーデータ生成
        x_data_new = torch.randn(self.N_samples, self.l, 1)
        h_data_new = torch.randn(self.N_samples, self.n, 1)
        y1_list = [self.echo_hidden(self.W_in_p, self.W_h_p, self.b_p, x, h).squeeze(-1)
                   for x, h in zip(x_data_new, h_data_new)]
        y1_new = torch.stack(y1_list)
        correlation_mean = self._correlation_coefficient(self.y0, y1_new).item()
        loss_mean = correlation_mean * (-1)
        return {'correlation_mean': correlation_mean, 'loss_mean': loss_mean}

def _example():
    n, l = 10, 5
    W_in = torch.randn(n, l)
    W_h = torch.randn(n, n)
    b = torch.randn(n, 1)
    epsilon = 0.1
    gamma = 0.5
    N_samples = 100
    N_epochs = 100
    model = ApproxEuSN2EcSN(W_in, W_h, b, epsilon, gamma, N_samples, N_epochs)
    model.fit()
    eval_result = model.evaluate()
    print(f"train data: Loss: {model.loss_mean:.4f}, Corr: {model.correlation_mean:.4f}")
    print(f"test data: Loss: {eval_result['loss_mean']:.4f}, Corr: {eval_result['correlation_mean']:.4f}")

if __name__ == "__main__":
    _example()



"""
pythonで、
W_in（nxl行列）、W_h（nxn行列）、b（nx1行列）、epsilon（正実数）、gamma（正実数）
が与えられたときに、
任意のx（lx1行列）、h（nx1行列）に対して、
y0 = h + epsilon * np.tanh(W_in@x + (W_h - gamma@np.eye)@h + b)
y1 = np.tanh(W_in_apx@x + W_h_apx@h + b_apx)
としたとき、y0とy1の相関を正の方向に最大化するような
W_in_p（nxl行列）、W_h_p（nxn行列）、b_p（nx1行列）
を求めたい。
"""