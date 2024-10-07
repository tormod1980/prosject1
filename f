import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'OLS': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

# Perform k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

mse_results = {name: [] for name in models}

for name, model in models.items():
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        z_train, z_test = z[train_index], z[test_index]
        
        model.fit(X_train, z_train)
        z_pred = model.predict(X_test)
        
        mse = mean_squared_error(z_test, z_pred)
        mse_results[name].append(mse)

# Calculate average MSE for each model
avg_mse = {name: np.mean(mse_results[name]) for name in mse_results}

# Plot the results
plt.figure(figsize=(10, 6))
for name in mse_results:
    plt.plot(range(1, k+1), mse_results[name], label=f'{name} MSE')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('Cross-Validation MSE for Different Models')
plt.legend()
plt.show()

# Print average MSE
for name in avg_mse:
    print(f'Average MSE for {name}: {avg_mse[name]}')

# Compare with bootstrap results
# Assuming you have the bootstrap MSE results stored in a variable `bootstrap_mse`
# bootstrap_mse = {'OLS': [...], 'Ridge': [...], 'Lasso': [...]}
# avg_bootstrap_mse = {name: np.mean(bootstrap_mse[name]) for name in bootstrap_mse}

# Print comparison
# for name in avg_mse:
#     print(f'Average MSE for {name} (Cross-Validation): {avg_mse[name]}')
#     print(f'Average MSE for {name} (Bootstrap): {avg_bootstrap_mse[name]}')
