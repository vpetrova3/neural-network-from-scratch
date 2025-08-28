import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse

# -------- helpers --------
def one_hot(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(np.float32)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy(y_true_onehot, y_prob):
    n = y_true_onehot.shape[0]
    eps = 1e-12
    return -np.sum(y_true_onehot * np.log(y_prob + eps)) / n

# -------- model --------
class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim, lr=0.1, seed=1337):
        rng = np.random.default_rng(seed)
        # Xavier/He-ish inits
        self.W1 = rng.normal(0, np.sqrt(2.0/in_dim), size=(in_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0/hidden_dim), size=(hidden_dim, out_dim)).astype(np.float32)
        self.b2 = np.zeros((1, out_dim), dtype=np.float32)
        self.lr = lr

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1        # (N,H)
        self.h1 = relu(self.z1)                # (N,H)
        self.z2 = self.h1 @ self.W2 + self.b2  # (N,C)
        self.p  = softmax(self.z2)             # (N,C)
        return self.p

    def backward(self, X, y_onehot):
        N = X.shape[0]
        dz2 = (self.p - y_onehot) / N               # (N,C)
        dW2 = self.h1.T @ dz2                        # (H,C)
        db2 = dz2.sum(axis=0, keepdims=True)         # (1,C)

        dh1 = dz2 @ self.W2.T                        # (N,H)
        dz1 = dh1 * relu_deriv(self.z1)              # (N,H)
        dW1 = X.T @ dz1                               # (D,H)
        db1 = dz1.sum(axis=0, keepdims=True)         # (1,H)

        # SGD step
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def train(self, X, y, Xv, yv, epochs=30, batch=64, seed=1337):
        rng = np.random.default_rng(seed)
        for ep in range(1, epochs+1):
            idx = np.arange(len(X)); rng.shuffle(idx)
            for s in range(0, len(X), batch):
                b = idx[s:s+batch]
                p = self.forward(X[b])
                self.backward(X[b], y[b])

            # validation
            pv = self.forward(Xv)
            loss = cross_entropy(yv, pv)
            acc  = accuracy_score(yv.argmax(1), pv.argmax(1))
            print(f"Epoch {ep:02d} | val_loss {loss:.4f} | val_acc {acc*100:.2f}%")

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X = StandardScaler().fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    ytr_oh = one_hot(ytr, 10)
    yte_oh = one_hot(yte, 10)

    model = MLP(in_dim=64, hidden_dim=args.hidden, out_dim=10, lr=args.lr, seed=args.seed)
    model.train(Xtr, ytr_oh, Xte, yte_oh, epochs=args.epochs, batch=args.batch, seed=args.seed)

    ypred = model.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"Final validation accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
