import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


df = pd.read_csv("dataset.csv")
x = df.drop(columns=["price_range"])
y = df["price_range"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=1)

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

price_ranges = {
    0: "₹5,000 - ₹10,000",
    1: "₹10,000 - ₹15,000",
    2: "₹15,000 - ₹20,000",
    3: "₹20,000 - ₹30,000"
}

st.title("Mobile Price Prediction")
st.write(f"Model Accuracy: {accuracy:.2f}")

st.header("Enter Mobile Features")
new_phone = []
for feature in x.columns:
    min_val = df[feature].min()
    max_val = df[feature].max()
    if min_val == 0 and max_val == 1:
        value = st.slider(f"{feature} (Choose 0 or 1)", 0, 1, 0)
    else:
        value = st.slider(f"{feature} (Min: {min_val}, Max: {max_val})", int(min_val), int(max_val), int(max_val) if min_val == max_val else int(min_val))
    new_phone.append(value)

if st.button("Predict Price"):
    new_phone_scaled = scaler.transform([new_phone])
    predicted_category = model.predict(new_phone_scaled)[0]
    predicted_price_range = price_ranges.get(predicted_category, "Unknown")
    st.success(f"Predicted Price Range: {predicted_price_range}")
