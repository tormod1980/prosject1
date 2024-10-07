import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Define the Franke function
def franke_function(x, y):
    term1 = 0.75 * np.exp(-(9*x - 2)**2 / 4.0 - (9*y - 2)**2 / 4.0)
    term2 = 0.75 * np.exp(-(9*x + 1)**2 / 49.0 - (9*y + 1) / 10.0)
    term3 = 0.5 * np.exp(-(9*x - 7)**2 / 4.0 - (9*y - 3)**2 / 4.0)
    term4 = -0.2 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return term1 + term2 + term3 + term4

# Generate data
np.random.seed(42)
n = 1000
x = np.random.rand(n)
y = np.random.rand(n)
z = franke_function(x, y) + 0.1 * np.random.randn(n)  # Adding stochastic noise

# Prepare the design matrix for polynomial fitting
def design_matrix(x, y, degree):
    m = len(x)
    X = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            X = np.hstack((X, (x**(i-j) * y**j).reshape(-1, 1)))
    return X

# Split the data into training and test sets
X = design_matrix(x, y, 5)
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perform Ridge regression using matrix inversion
def ridge_regression(X, y, lambda_):
    I = np.eye(X.shape[1])
    return np.linalg.pinv(X.T @ X + lambda_ * I) @ X.T @ y

# Evaluate the model
lambdas = [0.1, 1, 10, 100]
degrees = range(1, 6)
mse_train = {l: [] for l in lambdas}
mse_test = {l: [] for l in lambdas}
r2_train = {l: [] for l in lambdas}
r2_test = {l: [] for l in lambdas}

for lambda_ in lambdas:
    for degree in degrees:
        X_train = design_matrix(x[:800], y[:800], degree)
        X_test = design_matrix(x[800:], y[800:], degree)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        beta = ridge_regression(X_train_scaled, z[:800], lambda_)
        z_train_pred = X_train_scaled @ beta
        z_test_pred = X_test_scaled @ beta
        
        mse_train[lambda_].append(mean_squared_error(z[:800], z_train_pred))
        mse_test[lambda_].append(mean_squared_error(z[800:], z_test_pred))
        r2_train[lambda_].append(r2_score(z[:800], z_train_pred))
        r2_test[lambda_].append(r2_score(z[800:], z_test_pred))

# Plot the results
plt.figure(figsize=(14, 6))

for lambda_ in lambdas:
    plt.subplot(1, 2, 1)
    plt.plot(degrees, mse_train[lambda_], label=f'Train MSE (λ={lambda_})')
    plt.plot(degrees, mse_test[lambda_], label=f'Test MSE (λ={lambda_})')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(degrees, r2_train[lambda_], label=f'Train $R^2$ (λ={lambda_})')
    plt.plot(degrees, r2_test[lambda_], label=f'Test $R^2$ (λ={lambda_})')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('$R^2$ Score')
    plt.legend()

plt.show()
