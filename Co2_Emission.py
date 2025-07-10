import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# -----------------------
# Load CSV Data
# -----------------------
@st.cache_data
def load_data():
    data = pd.read_csv("Dataset.csv")
    data['route_name'] = data['origin'] + " ➜ " + data['destination']
    return data

st.title("🚚 GreenRoute - Eco-Friendly Delivery Route Optimizer")

df = load_data()

# -----------------------
# Preprocessing
# -----------------------
categorical_cols = ['vehicle_type', 'traffic_level', 'weather']
numeric_cols = ['fuel_consumption', 'distance_km', 'cargo_weight', 'trip_duration']
target_col = 'carbon_emission'

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

scaler = StandardScaler()
scaled_nums = scaler.fit_transform(df[numeric_cols])
scaled_num_df = pd.DataFrame(scaled_nums, columns=numeric_cols)

X = pd.concat([encoded_cat_df, scaled_num_df], axis=1)
y = df[target_col]

# -----------------------
# Model
# -----------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer=Adam(0.01), loss='mse', metrics=['mae'])
model.fit(X, y, epochs=50, batch_size=16, verbose=0)

# -----------------------
# User Input for 3 Routes with route name dropdown
# -----------------------
st.header("📥 Enter Route Details")

route_options = df[['route_name', 'route_id']].drop_duplicates().set_index('route_name').to_dict()['route_id']

def input_route(i):
    st.subheader(f"Route {i+1}")
    selected_route = st.selectbox(f"Select Route {i+1}", list(route_options.keys()), key=f"route{i}")
    vehicle = st.selectbox(f"Vehicle Type {i+1}", ['diesel', 'petrol', 'electric'], key=f"v{i}")
    traffic = st.selectbox(f"Traffic Level {i+1}", ['low', 'medium', 'high'], key=f"t{i}")
    weather = st.selectbox(f"Weather {i+1}", ['clear', 'rainy', 'foggy', 'snowy'], key=f"w{i}")
    fuel = st.slider(f"Fuel Consumption (L/km) {i+1}", 0.1, 0.6, 0.3, 0.01, key=f"f{i}")
    distance = st.slider(f"Distance (km) {i+1}", 10, 200, 50, 1, key=f"d{i}")
    cargo = st.slider(f"Cargo Weight (kg) {i+1}", 500, 2000, 1000, 10, key=f"c{i}")
    duration = st.slider(f"Trip Duration (min) {i+1}", 30, 240, 60, 5, key=f"dur{i}")
    return {
        'route_name': selected_route,
        'vehicle_type': vehicle,
        'traffic_level': traffic,
        'weather': weather,
        'fuel_consumption': fuel,
        'distance_km': distance,
        'cargo_weight': cargo,
        'trip_duration': duration
    }

# Collect inputs for 3 routes
routes = [input_route(i) for i in range(3)]

# Prediction
if st.button("🔍 Find Optimal Route"):
    routes_df = pd.DataFrame(routes)
    encoded_input = pd.DataFrame(encoder.transform(routes_df[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
    scaled_input = pd.DataFrame(scaler.transform(routes_df[numeric_cols]), columns=numeric_cols)
    final_input = pd.concat([encoded_input, scaled_input], axis=1)

    predictions = model.predict(final_input).flatten()
    routes_df['Predicted Emission (kg CO₂)'] = predictions

    best_idx = predictions.argmin()

    st.subheader("📊 Route Comparison")
    st.dataframe(routes_df[['route_name'] + categorical_cols + numeric_cols + ['Predicted Emission (kg CO₂)']])

    st.success(f"✅ Optimal Route: {routes_df.iloc[best_idx]['route_name']} with predicted emission: {predictions[best_idx]:.2f} kg CO₂")
