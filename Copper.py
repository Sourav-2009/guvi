# Import necessary libraries (Shell 1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import pickle

# Load the dataset (Shell 2)
df = pd.read_excel('Copper_Set.xlsx')

# Data Understanding and Exploration (Shell 3)
st.write("Data preview:")
st.dataframe(df.head())

# Identify the types of variables
st.write("Data types:")
st.write(df.dtypes)

# Check for missing values
st.write("Missing values:")
st.write(df.isna().sum())

# Dropdown filters for graphs
if 'status' in df.columns:
    status_filter = st.selectbox('Filter by Status', options=['All'] + df['status'].unique().tolist())
else:
    status_filter = 'All'

if 'item type' in df.columns:
    item_type_filter = st.selectbox('Filter by Item Type', options=['All'] + df['item type'].unique().tolist())
else:
    item_type_filter = 'All'

# Search bar filter
search_term = st.text_input('Search by Customer Name')

# Date range filter
if 'item_date' in df.columns:
    date_range = st.slider('Select Date Range', min_value=pd.to_datetime(df['item_date']).min(), max_value=pd.to_datetime(df['item_date']).max(), value=(pd.to_datetime(df['item_date']).min(), pd.to_datetime(df['item_date']).max()))
else:
    date_range = None

# Apply filters
df_filtered = df.copy()
if status_filter != 'All':
    df_filtered = df_filtered[df_filtered['status'] == status_filter]
if item_type_filter != 'All':
    df_filtered = df_filtered[df_filtered['item type'] == item_type_filter]
if search_term:
    df_filtered = df_filtered[df_filtered['customer'].str.contains(search_term, case=False, na=False)]
if date_range:
    df_filtered = df_filtered[(pd.to_datetime(df_filtered['item_date']) >= date_range[0]) & (pd.to_datetime(df_filtered['item_date']) <= date_range[1])]

# Check for skewness and outliers using Plotly charts


# Data Preprocessing (Shell 4)
# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df['material_ref'] = imputer.fit_transform(df[['material_ref']])

# Treat reference columns as categorical variables
df['material_ref'] = df['material_ref'].astype('category')

# Handle skewness using log transformation for 'selling_price'
df['selling_price'] = np.log1p(df['selling_price'])

# Encoding categorical variables
label_encoder = LabelEncoder()
df['status'] = label_encoder.fit_transform(df['status'])

# Split the dataset into train and test sets (Shell 5)
X = df.drop(['selling_price', 'status'], axis=1)
y_regression = df['selling_price']
y_classification = df['status']

X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regression Model (Shell 6)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train_scaled, y_train_reg)
y_pred_reg = regressor.predict(X_test_scaled)

mse = mean_squared_error(y_test_reg, y_pred_reg)
st.write(f'Mean Squared Error (Regression Model): {mse}')

# Classification Model (Shell 7)
classifier = ExtraTreesClassifier(random_state=42)
classifier.fit(X_train_class, y_train_class)
y_pred_class = classifier.predict(X_test_class)

accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)
st.write(f'Accuracy: {accuracy}')
st.write(f'Precision: {precision}')
st.write(f'Recall: {recall}')
st.write(f'F1 Score: {f1}')

# Save models and encoders using Pickle (Shell 8)
with open('regression_model.pkl', 'wb') as f:
    pickle.dump(regressor, f)
with open('classification_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Streamlit User Interface (Shell 9)
def predict(input_data, task):
    if task == 'Regression':
        model = pickle.load(open('regression_model.pkl', 'rb'))
        prediction = model.predict(scaler.transform(input_data))
        prediction = np.expm1(prediction)  # Reverse transformation of log to get original scale
        return prediction
    elif task == 'Classification':
        model = pickle.load(open('classification_model.pkl', 'rb'))
        prediction = model.predict(input_data)
        return label_encoder.inverse_transform(prediction)

# Interactive Streamlit App
st.title("Industrial Copper Modeling Prediction")
st.markdown("""
<div style='background-color: #f0f0f0; padding: 10px;'>
  <h2 style='text-align: center;'>Predict Selling Price or Lead Status</h2>
  <img src='/mnt/data/image.png' alt='Copper Industry' style='display: block; margin-left: auto; margin-right: auto; width: 50%;'>
</div>
""", unsafe_allow_html=True)

# User input fields for model prediction
task = st.selectbox('Choose a task', ['Regression', 'Classification'])
input_data = []
for column in X.columns:
    value = st.text_input(f'Enter value for {column}:')
    try:
        value = float(value)
    except ValueError:
        st.warning(f'Invalid input for {column}, please enter a numeric value.')
    input_data.append(value)

input_data = np.array(input_data).reshape(1, -1)
if st.button('Predict'):
    result = predict(input_data, task)
    st.write(f'Prediction: {result}')

# Visualizations (Shell 10)
# Pie chart for Classification of Status

# Scatter plot for Thickness vs Selling Price

# Additional Visualizations
# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_filtered.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)
