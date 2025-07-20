import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and preprocess dataset
dataset = pd.read_csv("customer_churn_prediction.csv")


dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")
dataset.dropna(inplace=True)


# Encoding Binary columns into 0 or 1
binary_column = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

label = LabelEncoder()
for col in binary_column:
    dataset[col] = label.fit_transform(dataset[col])


# Ordinal Encoding  Multiclass columns in different in numeric values
multi_class_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]
dataset = pd.get_dummies(dataset, columns=multi_class_cols)

for col in dataset.columns:
    if dataset[col].dtype == "bool":
        dataset[col] = dataset[col].astype(int)


# features and target values separating
x = dataset.drop("Churn", axis=1)
y = dataset["Churn"]


# For balancing the Churn data
balance = RandomOverSampler()
balance_x, balance_y = balance.fit_resample(x, y)

# training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(
    balance_x, balance_y, test_size=0.1, random_state=42
)

# scaling the dataset
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# back to change into normal
x_train_scaler = pd.DataFrame(x_train_scaler, columns=x.columns)
x_test_scaler = pd.DataFrame(x_test_scaler, columns=x.columns)


# Machine learning Model
model = RandomForestClassifier(
    n_estimators=100, random_state=90, class_weight="balanced"
)
model.fit(x_train_scaler, y_train)

# accuracy
y_pred = model.predict(x_test_scaler)
accuracy = accuracy_score(y_test, y_pred)


# Plot: Confusion matrix
def graph():
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title("Actual vs Predicted Churn")
    return fig


# Plot: Pie chart
def pie_chart():
    fig, ax = plt.subplots(figsize=(4, 4))
    labels = ["Stay", "Churn"]
    sizes = [sum(y_pred == 0), sum(y_pred == 1)]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Predicted Churn")
    return fig


# Main prediction function
def predict_churn(
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
):
    input_data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": int(SeniorCitizen),
        "Partner": 1 if Partner == "Yes" else 0,
        "Dependents": 1 if Dependents == "Yes" else 0,
        "tenure": int(tenure),
        "PhoneService": 1 if PhoneService == "Yes" else 0,
        "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(MonthlyCharges) * int(tenure),
    }

    categories = {
        "MultipleLines": ["No", "No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "No internet service", "Yes"],
        "OnlineBackup": ["No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No internet service", "Yes"],
        "TechSupport": ["No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    }

    for col, options in categories.items():
        for opt in options:
            input_data[f"{col}_{opt}"] = 1 if locals()[col] == opt else 0

    for col in x.columns:
        if col not in input_data:
            input_data[col] = 0

    # change scaling data into normal
    df = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)

    # making prediction
    prediction = model.predict(df_scaled)[0]

    # Proba
    proba = model.predict_proba(df_scaled)[0]

    # confidence
    confidence = round(max(proba) * 100, 2)

    if confidence < 50:
        note = "⚠️ Low confidence prediction. Please verify input."
    else:
        note = "✅ Confident prediction."

    result = "Customer will Churn" if prediction == 1 else "Customer will Stay"
    return (
        result,
        f"{accuracy*100:.2f} %",
        f"{confidence} %",
        note,
        graph(),
        pie_chart(),
    )


# Gradio UI
gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio(["0", "1"], label="Senior Citizen"),
        gr.Radio(["Yes", "No"], label="Partner"),
        gr.Radio(["Yes", "No"], label="Dependents"),
        gr.Slider(0, 72, step=1, label="Tenure"),
        gr.Radio(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["No", "No phone service", "Yes"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["No", "No internet service", "Yes"], label="Online Security"),
        gr.Dropdown(["No", "No internet service", "Yes"], label="Online Backup"),
        gr.Dropdown(["No", "No internet service", "Yes"], label="Device Protection"),
        gr.Dropdown(["No", "No internet service", "Yes"], label="Tech Support"),
        gr.Dropdown(["No", "No internet service", "Yes"], label="Streaming TV"),
        gr.Dropdown(["No", "No internet service", "Yes"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Radio(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
            label="Payment Method",
        ),
        gr.Number(label="Monthly Charges"),
    ],
    outputs=[
        gr.Label(label="Result"),
        gr.Label(label="Accuracy"),
        gr.Label(label="Confidence"),
        gr.Textbox(label="Confidence Note"),
        gr.Plot(label="Confusion Matrix"),
        gr.Plot(label="Churn Prediction by Pie Chart"),
    ],
    title="Customer Churn Prediction System",
    description="Enter customer details below to check if they will churn or not.",
    article="MADE BY MOHD ALTAMASH",
).launch(share=True)
