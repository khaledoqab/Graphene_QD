% Load the saved sparse matrix data
load('sparse_matrix.mat');
% Convert to MATLAB sparse format
H = sparse(H);

% Solve for the 2000 smallest eigenvalues by magnitude (Shift-and-Invert method)
sigma = 0;  % Shift close to zero
num_eigenvalues = 2000;
disp('Computing 2000 smallest eigenvalues (by magnitude) in MATLAB...');
tic;
eigenvalues = eigs(H, num_eigenvalues, 1e-20);
toc;

% Display first 10 eigenvalues
disp('First 20 eigenvalues:');
disp(eigenvalues(1:20));

eigenvalues_sorted = sort(eigenvalues);

figure;
plot(eigenvalues_sorted, 'bo-', 'LineWidth', 1.5);
xlabel('Index');
ylabel('Eigenvalue');
title('Sorted Smallest Eigenvalues (by Magnitude)');
grid on;