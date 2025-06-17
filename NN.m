% === Load Your Data (from Simulink workspace) ===
X = inputs';     % Now 3 x N (each column is [e(t); e(t-1); e(t-2)])
Y = targets';    % Now 1 x N

% === Network Architecture ===
input_size = 3;
hidden1 = 10;
hidden2 = 5;
output_size = 1;
N = size(X, 2);  % Number of training samples

% === Initialize Weights and Biases ===
W1 = randn(hidden1, input_size) * 0.1;
b1 = zeros(hidden1, 1);

W2 = randn(hidden2, hidden1) * 0.1;
b2 = zeros(hidden2, 1);

W3 = randn(output_size, hidden2) * 0.1;
b3 = zeros(output_size, 1);

% === Activation Functions ===
relu = @(x) max(0, x);
drelu = @(x) x > 0;

% === Training Parameters ===
lr = 0.01;
epochs = 1000;

% === Training Loop ===
for epoch = 1:epochs
    total_loss = 0;
    
    for i = 1:N
        % Forward pass
        x = X(:, i);
        y = Y(:, i);
        
        z1 = W1*x + b1;
        a1 = relu(z1);
        
        z2 = W2*a1 + b2;
        a2 = relu(z2);
        
        z3 = W3*a2 + b3;
        y_hat = z3;

        % Compute loss (MSE)
        err = y_hat - y;
        total_loss = total_loss + err^2;

        % Backpropagation
        dL_dy = 2 * err;

        dW3 = dL_dy * a2';
        db3 = dL_dy;

        da2 = W3' * dL_dy;
        dz2 = da2 .* drelu(z2);
        dW2 = dz2 * a1';
        db2 = dz2;

        da1 = W2' * dz2;
        dz1 = da1 .* drelu(z1);
        dW1 = dz1 * x';
        db1 = dz1;

        % Update weights
        W3 = W3 - lr * dW3;
        b3 = b3 - lr * db3;

        W2 = W2 - lr * dW2;
        b2 = b2 - lr * db2;

        W1 = W1 - lr * dW1;
        b1 = b1 - lr * db1;
    end

    % Print average loss
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, MSE Loss = %.6f\n', epoch, total_loss / N);
    end
end

% === Save weights (optional) ===
save('trained_nn_weights.mat', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3');

% === Inference Function ===
% Use this to predict in MATLAB or Simulink later
function u = nn_pid_predict(x, W1, b1, W2, b2, W3, b3)
    relu = @(x) max(0, x);
    a1 = relu(W1 * x + b1);
    a2 = relu(W2 * a1 + b2);
    u  = W3 * a2 + b3;
end

% === Example usage: Predict output ===
test_sample = X(:, 1);  % First input example
predicted_output = nn_pid_predict(test_sample, W1, b1, W2, b2, W3, b3);
fprintf('Predicted output for first input: %.4f\n', predicted_output);
