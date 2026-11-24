
# SNN Architecture and Gradient Explanation

## 1. Forward Pass

The model takes MNIST images and passes them through
eight hidden layers using SELU activation.

Architecture:
- Input: (B, 1, 28, 28)
- Flatten: (B, 784)
- Hidden layer 1: (B, 128)
- Hidden layers 2–8: (B, 128)
- Output logits: (B, 10)

For each hidden layer:

Z_l = H_(l-1) @ W_l.T + b_l

H_l = SELU(Z_l)

The output layer produces 10 logits corresponding
to the 10 MNIST classes.

## 2. SELU Activation

SELU uses two different expressions:

For x > 0:
SELU(x) = lambda * x

For x <= 0:
SELU(x) = lambda * alpha * (exp(x) - 1)

Where:
lambda = 1.0507
alpha = 1.6733

## 3. Backward Pass

The network uses cross-entropy loss.

The gradient of the loss with respect to the
output logits is:

dL/dlogits = (softmax(logits) - one_hot(labels)) / B

Gradients propagate backward through the output
linear layer, the SELU activation, and the
previous linear layers using the chain rule.

SELU derivative (away from zero):

For x > 0:
SELU'(x) = lambda

For x < 0:
SELU'(x) = lambda * alpha * exp(x)

For a hidden layer, the chain rule gives:

dL/dZ_l = (dL/dH_l) * SELU'(Z_l)

dL/dW_l = (dL/dZ_l).T @ H_(l-1)

This allows the optimizer to update the weights
during training.

## 4. Gradient Shape Verification

Batch size: 16

Input shape: (16, 1, 28, 28)
Output shape: (16, 10)

Output weight gradient: (10, 128)
Last hidden weight gradient: (128, 128)

All gradient shape checks passed.

## 5. Correctness Tests

- Layer shape assertions: PASSED
- SELU value checks: PASSED
- SELU gradient checks: PASSED
- Tiny-batch overfitting: PASSED

The model successfully overfits 16 MNIST samples.

These tests verify basic implementation correctness,
but do not establish generalization or prove
self-normalization.
