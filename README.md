# Neural Network Based PID Controller

This project replaces a traditional PID controller with a trained neural network that learns to replicate PID behavior using supervised learning — implemented from scratch in MATLAB and Simulink without using any toolboxes.

## 📌 Objective

- Replace a Ziegler–Nichols tuned PID controller with a neural network.
- Train the NN to mimic the PID's output based on past error history.
- Integrate the trained NN into a Simulink closed-loop control system.

## 🚀 Project Overview

### ✔️ What We Did:
- Built and tuned a PID controller using classical methods (Ziegler–Nichols).
- Collected training data: `[e(t), e(t-1), e(t-2)] → u(t)`
- Trained a feedforward neural network (2 hidden layers) using custom MATLAB code (no toolbox).
- Replaced the PID block with a `MATLAB Function block` in Simulink.
- Verified the response matched the original PID controller.


## 📷 Sample Output

![Screenshot 2025-06-13 120313](https://github.com/user-attachments/assets/2f315ef5-4bc3-45fe-aa1c-f45aaed73cad)


## 📚 Key Concepts Used
- Supervised learning
- Feedforward neural networks
- Backpropagation & gradient descent
- PID control theory
- Simulink block replacement
