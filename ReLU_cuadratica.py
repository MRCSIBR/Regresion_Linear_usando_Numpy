import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Generate quadratic progression: y = x^2 for x in [1, 21]
x = np.arange(1, 22, dtype=np.float32).reshape(-1, 1)  # Shape: (21, 1)
y = (x ** 2).flatten()  # Shape: (21,)

# 2. Build Neural Network with ReLU activation
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(1,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)  # Linear output for regression
])

# 3. Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train the model
print("Training the neural network...")
history = model.fit(x, y, epochs=3000, verbose=0)

# 5. Predict using the trained model
y_pred = model.predict(x).flatten()

# 6. Plot using Plotly
fig = go.Figure()

# Add original data
fig.add_trace(go.Scatter(
    x=x.flatten(),
    y=y,
    mode='markers+lines',
    name='Original Quadratic (y = x²)',
    marker=dict(size=8, color='blue'),
    line=dict(dash='dot')
))

# Add predicted data
fig.add_trace(go.Scatter(
    x=x.flatten(),
    y=y_pred,
    mode='markers+lines',
    name='Neural Network Prediction',
    marker=dict(size=6, color='red', symbol='x'),
    line=dict(color='orange')
))

# Customize layout
fig.update_layout(
    title="Neural Network (ReLU) Fitting Quadratic Progression",
    xaxis_title="x",
    yaxis_title="y",
    legend_title="Legend",
    template="plotly_white"
)

# Show plot
fig.show()

# Optional: Print training loss
final_loss = history.history['loss'][-1]
print(f"Final Training Loss (MSE): {final_loss:.4f}")
