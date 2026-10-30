## Customer Churn Prediction
(in development)

### About the Project

Customer Churn Prediction is a machine learning project that aims to predict whether a customer is likely to leave a telecommunications company based on their service usage, account information, subscription details, and billing history.

The project uses the IBM Telco Customer Churn dataset to explore customer behaviour, identify patterns associated with churn, and build a classification model that can help businesses understand and identify customers at risk of leaving.

### How It Works

1. **Data Collection:** Uses the IBM Telco Customer Churn dataset containing customer information and churn records.
2. **Data Cleaning:** Handles missing values, converts numerical data into appropriate formats, and removes unnecessary columns.
3. **Data Preprocessing:** Separates numerical and categorical features, scales numerical values using StandardScaler, and encodes categorical values using OneHotEncoder.
4. **Data Splitting:** Divides the dataset into training and testing sets to evaluate the model on unseen data.
5. **Model Training:** Uses Logistic Regression as the initial classification algorithm to predict customer churn.
6. **Model Evaluation:** Evaluates predictions using accuracy and a confusion matrix.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost (planned for experimentation)

### Future Plans

- Experiment with additional machine learning algorithms.
- Improve churn detection using precision, recall, and F1-score.
- Save the trained model for reuse.
- Build a backend using FastAPI.
- Develop an interactive frontend using React and Tailwind CSS.
- Deploy the application so users can enter customer details and receive churn predictions.
