##  About the Project: Customer Churn Prediction System

The **Customer Churn Prediction System** is a complete end-to-end machine learning project that predicts whether a customer is likely to leave (churn) or stay with a company based on their usage patterns, subscription type, and account information. It is designed to help businesses **proactively identify customers at risk** and take corrective actions to improve retention.

### 🔍 Problem Statement

In many service-based industries like telecom, churn (customer leaving the service) can significantly impact revenue. Retaining existing customers is often cheaper than acquiring new ones. Thus, being able to **predict churn in advance** allows businesses to target at-risk customers with offers or better service.

---

### 📊 Dataset Description

The dataset used is `customer_churn_system.csv`, which includes information such as:

- **Demographics:** Gender, Senior Citizen, Partner, Dependents  
- **Service Usage:** Internet services, Streaming services, Phone services  
- **Account Info:** Contract type, Payment method, Monthly & Total charges  
- **Target Variable:** `Churn` — Whether the customer left or not

---

### 🧪 Key Steps Performed

1. **Data Cleaning & Preprocessing**
   - Converted `TotalCharges` to numeric
   - Handled missing values
   - Label encoding for binary categorical variables
   - One-hot encoding for multi-class categorical variables
   - Feature scaling using `StandardScaler`

2. **Data Balancing**
   - Used `RandomOverSampler` to handle class imbalance (churn is rare in most datasets)

3. **Model Building**
   - Trained a `RandomForestClassifier` with 100 estimators
   - Applied `class_weight="balanced"` to further handle imbalance

4. **Evaluation**
   - Evaluated model using `accuracy_score` and `confusion matrix`
   - Achieved a solid accuracy score (dynamically shown in UI)
   - Visualized prediction distribution using pie chart

5. **User Interface (UI)**
   - Built a responsive Gradio web app interface
   - Input features collected using radio buttons, dropdowns, sliders
   - Returns:
     - Predicted result (Churn or Stay)
     - Accuracy of model
     - Confidence level of the prediction
     - Confidence warning message (if confidence is low)
     - Confusion matrix plot
     - Pie chart of predicted churn distribution

---

### 💡 Why This Project Stands Out

- ✅ **Real-World Use Case**: Churn prediction is a key application in telecom, SaaS, banking, etc.
- ✅ **Production-Ready**: GUI is interactive, intuitive, and clean—can be used directly by non-tech users.
- ✅ **Confidence Insights**: Includes confidence % and auto-warnings to improve decision-making reliability.
- ✅ **Smart Engineering**: `TotalCharges` is **automatically calculated**, reducing user input error.
- ✅ **Visualized Outputs**: Confusion matrix and pie chart add trust and transparency.
- ✅ **Deployable Anywhere**: Easily deployable to Hugging Face Spaces, Streamlit, Render, etc.

---

### 🎯 What This Project Demonstrates (Skill-wise)

- ✅ Machine Learning Model Building
- ✅ Data Preprocessing (Handling NaNs, Label Encoding, One-Hot Encoding)
- ✅ Model Evaluation & Metrics
- ✅ Class Imbalance Handling (with `imblearn`)
- ✅ Interactive App Development using **Gradio**
- ✅ Feature Engineering (auto-calculating TotalCharges)
- ✅ Confidence Scoring and Probability Interpretation
- ✅ End-to-End ML Pipeline & Deployment Readiness

  
### LinkedIn Profile

🔗 [LinkedIn Profile](https://www.linkedin.com/in/mohd-altamash-0997592a6?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

---

### License

This project is licensed under the [MIT License](LICENSE).  
You're free to use and modify it, but *you must give credit* to the original author: **Mohd Altamash**.
