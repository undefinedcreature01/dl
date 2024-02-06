
```
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x_values = np.linspace(-7, 7, 100)

# Calculate sigmoid values
y_values = sigmoid(x_values)

# Plot the sigmoid function
plt.plot(x_values, y_values, label='Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True)
plt.show()
```
