# NN From Scratch (Digits)

Tiny 2-layer neural network (NumPy only) trained on sklearn Digits (8×8).

## Quickstart
```bash
pip install -r requirements.txt
python main.py --epochs 30 --hidden 64 --lr 0.1 --batch 64


## Taking notes with all main points from the project (explanation of the code, nn theory and steps to completion):
- Let's look at the basics: neurons (basic unit of a NN), they take input, do math with it and produce one output. It's like taking x1+x2 and then producing y. Each input is multiplied by a weight, then all weighted inputs are added together with a bias b, and the sum is passed through an activation function. Activation function is used to to turn an unbounded input into an output with a nice, predictable form. Commonly used one is the sigmoid function (it only outputs numbers in range (0,1)). It comperesses big numbers to those.
- 
##