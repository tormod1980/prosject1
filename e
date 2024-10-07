import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

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

# Perform OLS regression
model = LinearRegression()
model.fit(X_train_scaled, z_train)
z_train_pred = model.predict(X_train_scaled)
z_test_pred = model.predict(X_test_scaled)

# Calculate MSE
mse_train = mean_squared_error(z_train, z_train_pred)
mse_test = mean_squared_error(z_test, z_test_pred)

# Bootstrap resampling
n_bootstraps = 100
mse_bootstrap = np.zeros(n_bootstraps)
for i in range(n_bootstraps):
    X_resampled, z_resampled = resample(X_train_scaled, z_train)
    model.fit(X_resampled, z_resampled)
    z_test_pred = model.predict(X_test_scaled)
    mse_bootstrap[i] = mean_squared_error(z_test, z_test_pred)

# Calculate bias and variance
bias = np.mean(z_test_pred) - np.mean(z_test)
variance = np.var(z_test_pred)
noise = np.var(z_test - z_test_pred)

# Plot the results
plt.figure(figsize=(10, 6))
plt.hist(mse_bootstrap, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(mse_test, color='r', linestyle='--', label='Test MSE')
plt.xlabel('MSE')
plt.ylabel('Frequency')
plt.title('Bootstrap Resampling of Test MSE')
plt.legend()
plt.show()

print(f"Bias: {bias}")
print(f"Variance: {variance}")
print(f"Noise: {noise}")
