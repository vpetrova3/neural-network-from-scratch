# NN From Scratch (Digits)

Tiny 2-layer neural network (NumPy only) trained on sklearn Digits (8×8).

## Quickstart
```bash
pip install -r requirements.txt
python main.py --epochs 30 --hidden 64 --lr 0.1 --batch 64


## Notes & Theory (study section)
Taking notes with all main points from the project (explanation of the code, nn theory and steps to completion):
- Let's look at the basics: neurons (basic unit of a NN), they take input, do math with it and produce one output. It's like taking x1+x2 and then producing y. Each input is multiplied by a weight, then all weighted inputs are added together with a bias b, and the sum is passed through an activation function. Activation function is used to to turn an unbounded input into an output with a nice, predictable form. Commonly used one is the sigmoid function (it only outputs numbers in range (0,1)). It comperesses big numbers to those.


### Neural Networks Basics
- Neuron = weighted sum of inputs + activation.
- Forward pass = compute outputs.
- Loss function = measures how wrong predictions are.
- Backpropagation = compute gradients w.r.t. weights.
- Gradient descent = update weights to minimize loss.

### This Project
- Input = 64 (8x8 pixels).
- Hidden layer = ReLU, size configurable.
- Output = 10 (digits 0–9), Softmax + cross-entropy.
- Training = minibatch SGD, track validation accuracy.

### Code Walkthrough
- `MLP.forward()` → compute hidden + logits.
- `MLP.backward()` → backpropagation with gradients.
- `MLP.step()` → update weights with learning rate.
- `train()` → loop over epochs, save best weights.
- `predict()` → load weights, test accuracy.
## My Steps
- Step 1: Created repo, set up venv, added requirements.
- Step 2: Wrote skeleton MLP class.
- Step 3: Implemented forward pass → tested shapes.
- Step 4: Added backprop + trained → saw accuracy ~95%.
- Step 5: Took notes on how softmax + cross-entropy work together.

## Learnings
- Importance of data standardization before training.
- Backprop math clicked after coding it line by line.
- Saving weights lets me reload models for predictions.
