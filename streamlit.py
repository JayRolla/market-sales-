
import pickle
import streamlit as st




# Load the pickled XGBoost model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessing objects
with open('preprocessing_objects.pkl', 'rb') as f:
    preprocessing_objects = pickle.load(f)

# Function to preprocess input features
def preprocess_features(df):
    # Extract preprocessing objects
    label_encoder = preprocessing_objects['label_encoder']
    visibility_avg = preprocessing_objects['visibility_avg']
    visibility_bins = preprocessing_objects['visibility_bins']
    mrp_bins = preprocessing_objects['mrp_bins']

    # Apply label encoding to categorical columns
    df['Outlet_Type_Encoded'] = label_encoder.transform(df['Outlet_Type'])

    # Apply mean visibility per item
    df['Adjusted_Item_Visibility'] = df.apply(lambda row: visibility_avg[row['Item_Identifier']] if row['Item_Visibility'] == 0 else row['Item_Visibility'], axis=1)

    # Binning for Item_Visibility
    df['Item_Visibility_Bin'] = pd.cut(df['Item_Visibility'], bins=visibility_bins, labels=['Low', 'Medium', 'High', 'Very High'], right=False)

    # Binning for Item_MRP
    df['Item_MRP_Bin'] = pd.cut(df['Item_MRP'], bins=mrp_bins, labels=['Low', 'Medium', 'High', 'Very High'])

    # Drop unnecessary columns
    df = df.drop(columns=['Item_Identifier', 'Item_Type', 'Outlet_Identifier'])

    return df

# Function to make predictions
def predict_sales(df):
    # Preprocess input features
    df = preprocess_features(df)

    # Make prediction using relevant preprocessed features
    X = df[['Item_Weight', 'Outlet_Type_Encoded', 'Adjusted_Item_Visibility', 'Item_Visibility_Bin', 'Item_MRP_Bin']]

    # Make prediction
    prediction = model.predict(X)

    return prediction

# Streamlit app
def main():
    st.title('Future Sales Prediction')
    st.write('Fill in the details to predict future sales:')

    # Input form for features
    item_identifier = st.text_input('Item Identifier')
    item_weight = st.number_input('Item Weight')
    item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
    item_visibility = st.number_input('Item Visibility')
    item_type = st.selectbox('Item Type', ['Snack Foods', 'Dairy', 'Frozen Foods', 'Fruits and Vegetables', 'Others'])
    item_mrp = st.number_input('Item MRP')
    outlet_identifier = st.text_input('Outlet Identifier')
    outlet_establishment_year = st.number_input('Outlet Establishment Year')
    outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
    outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.selectbox('Outlet Type', ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    # Create a DataFrame with the user input features
    user_input = pd.DataFrame({
        'Item_Identifier': [item_identifier],
        'Item_Weight': [item_weight],
        'Item_Fat_Content': [item_fat_content],
        'Item_Visibility': [item_visibility],
        'Item_Type': [item_type],
        'Item_MRP': [item_mrp],
        'Outlet_Identifier': [outlet_identifier],
        'Outlet_Establishment_Year': [outlet_establishment_year],
        'Outlet_Size': [outlet_size],
        'Outlet_Location_Type': [outlet_location_type],
        'Outlet_Type': [outlet_type]
    })

    # Make prediction when button is clicked
    if st.button('Predict'):
        prediction = predict_sales(user_input)
        st.success(f'Predicted Sales: {prediction[0]}')

if __name__ == '__main__':
    main()