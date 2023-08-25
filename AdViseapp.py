import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample sales data
data = {
    'ad_spend': [230, 250, 180, 300, 210, 270, 330, 290, 310, 280],
    'sales': [12, 15, 11, 16, 14, 17, 18, 15, 19, 16]
}

df = pd.DataFrame(data)

# Splitting data
X = df[['ad_spend']]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Neural network model
modelo = Sequential()
modelo.add(Dense(10, activation='relu', input_shape=(1,)))
modelo.add(Dense(10, activation='relu'))
modelo.add(Dense(1))
modelo.compile(optimizer='adam', loss='mse')
modelo.fit(X_train_scaled, y_train, epochs=100, batch_size=1, verbose=0)

# Prediction function
def predict_sales(ad_spend):
    scaled_input = scaler.transform([[ad_spend]])
    predicted_sales = modelo.predict(scaled_input)
    return predicted_sales[0][0]


# Sample prediction
ad_spend_value = 320
predicted_sales = predict_sales(ad_spend_value)
print(f"For an ad spend of {ad_spend_value}, predicted sales are: {predicted_sales:.2f}.")
