# Neural Network From Scratch

Part 1: My 2-layer neural network (NumPy) trained on a toy dataset of four people’s **weight and height → gender**

Part 2: Digits – Neural network (NumPy) trained on sklearn’s **Digits (8×8 pixel images → digit classification)**.

## Quickstart
```bash
pip install -r requirements.txt

# Run Part 1 (Fundamentals)
python fundamentals/main.py

# Run Part 2 (Digits)
python digits/main.py --epochs 30 --hidden 64 --lr 0.1 --batch 64

```

## Notes & Theory (study section)
Part 1: 
Taking notes with all main points from the project (explanation of the code, nn theory and steps to completion):
- Let's look at the basics: neurons (basic unit of a NN), they take input, do math with it and produce one output. It's like taking x1+x2 and then producing y. Each input is multiplied by a weight, then all weighted inputs are added together with a bias b, and the sum is passed through an activation function. Activation function is used to to turn an unbounded input into an output with a nice, predictable form. Commonly used one is the sigmoid function (it only outputs numbers in range (0,1)). It comperesses big numbers to those.
- Example of a neuron: we have a 2- input neuron using a sigmoid function with following params. w = [0,1] (w1 = 0, w2 = 1 in vector form), b = 4. Now neuron's input is x = [2,3]. We use dot product to write thigs concisely: (w . x) +b = ((w1*x1)+(w2*x2)) + b = 0*2 + 1*3 + 4 = 7, then y = f(w . x +b) = f(7) = 0.999, so neuron outputs 0.999 when given the inputs x = [2,3]
- This process is called freeforward: passing inputs forward to get an output
- We will now code the neuron wiht NumPy:
    A neuron takes weighted inputs, adds a bias, and passes the result through an activation function.  
    Example from above:
    ```python
    import numpy as np

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    class Neuron:
        def __init__(self, weights, bias):
            self.weights = weights
            self.bias = bias

        def feedforward(self, inputs):
            total = np.dot(self.weights, inputs) + self.bias
            return sigmoid(total)

    weights = np.array([0, 1])
    bias = 4
    n = Neuron(weights, bias)
    print(n.feedforward(np.array([2, 3])))  # ≈ 0.999 same as from above
    ```
- Combining neurons into a nn: nn is a bunch of interconnected neurons. It can looks like input layer -> hidden layer -> output layer (2 input neurons to a hidden layer with 2 neurons to a single output neuron). There can be multiple hidden layers
- An example: feedforward -> let's use the example of a nn above and assume all neurons have the same weights = [0,1] and same bias = 0, and same sigmoid activation function. Let h1,h1,o1 denote the outputs of neurons they represent. We pass in the input x = [2,3]: h1 = h2 = f(w . x + b) = f((0*2) + (1*3) + 0) = f(3) = 0.9526; o1 = f(w . [h1,h2] + b) = f((0*h1) + (1*h2)+0) = f(0.9526) = 0.7216. The output of the neural network is 0.7216.
- A nn can have any number of layers and any number of neurons in those layers. Idea stays the same: feed inputs to get outputs at the end.
- Let's implement feedforward for our nn:


### Neural Networks Basics
- Neuron = weighted sum of inputs + activation.
- Forward pass = compute outputs through hidden → output.
- Loss function = Mean Squared Error (MSE).
- Backpropagation = compute gradients w.r.t. weights using chain rule.
- Gradient descent = update weights to minimize loss.

### This Project
- Input = 2 features (weight, height).
- Hidden layer = 2 neurons, sigmoid activation.
- Output = 1 neuron, sigmoid activation (closer to 0 → Male, closer to 1 → Female).
- Loss = MSE (mean squared error).
- Training = stochastic gradient descent (SGD), one sample at a time.

### Code Walkthrough
- `OurNeuralNetwork.feedforward(x)` → compute hidden activations and final output.
- `OurNeuralNetwork.train(data, labels)` → forward pass + manual backprop + update weights.
- Loss is logged every few epochs to show training progress.
- After training, we test with new inputs (Emily, Frank).

## My Steps
- Step 1: Learned the math of a single neuron → coded `Neuron` class.
- Step 2: Combined neurons into a simple 2→2→1 network.
- Step 3: Implemented forward pass and tested with toy inputs.
- Step 4: Derived gradients for sigmoid + MSE by hand.
- Step 5: Implemented SGD training loop and saw loss decrease.

## Learnings
- Forward pass is intuitive; backprop clicked after coding step by step.
- Sigmoid derivative has a neat closed form: `f’(x) = f(x) * (1 - f(x))`.
- Small toy dataset helps understand gradient descent mechanics.
- Training loss steadily decreases, showing the network “learns.”

## Part 2: Digits
- Dataset: sklearn Digits (8×8 grayscale images of digits 0–9).
Preprocessing:
- StandardScaler to normalize input features.
- One-hot encoding for labels.
Model:
- Input: 64 (8×8 pixels).
- Hidden: Configurable (default 64), ReLU activation.
- Output: 10 neurons, Softmax activation.
- Loss: Cross-entropy.
Training:
- Mini-batch stochastic gradient descent (SGD).
- Epochs configurable (default 30).
- Validation accuracy tracked.
- Results: Achieved ~97% validation accuracy with 1 hidden layer of size 64 (snippet below):
    Epoch 30 | val_loss 0.1209 | val_acc 97.22%
    Final validation accuracy: 97.22%   

