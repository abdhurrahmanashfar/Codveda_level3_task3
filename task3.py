

# Note:
# TensorFlow could not be used due to an environment issue in VSCode 
# that caused import errors and conflicts with NumPy. 
# As a result, I used scikit-learn to implement and train a neural network using the MLPClassifier, 
# which provides a simple feedforward architecture suitable for classification tasks like MNIST.


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. Load and preprocess the MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # Normalize pixel values to [0, 1]
y = y.astype("int")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data (important for MLPClassifier)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Design neural network architecture
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Two hidden layers
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    batch_size=64,
    max_iter=20,
    verbose=True,
    random_state=42
)

# 3. Train the model
print("Training model...")
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 5. Plot the loss curve
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
