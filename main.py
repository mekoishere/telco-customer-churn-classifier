import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score
import joblib

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)
    
X = df.drop('Churn', axis=1)
y = df['Churn']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# dużo więcej klientów zostaje niż odchodzi więc trzeba znaleźć model który sobie z tym poradzi
# musi umieć przewidzieć 1 a nie tylko 0 w kolumnie churn, chcemy jak nawiększy recall dla 1

# model = RandomForestClassifier(random_state=42, class_weight='balanced')
model = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)

param_grid = { # dla HistGradientBoostingClassifier
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300], 
    'max_depth': [3, 5, 10, None], 
    'l2_regularization': [0.0, 0.1, 1.0]
}

# param_grid = { # dla RandomForestClassifier
#     'n_estimators': [100, 200, 300],       
#     'max_depth': [5, 10, 15, None],        
#     'min_samples_split': [2, 5, 10],       
#     'min_samples_leaf': [1, 2, 4],       
#     'max_features': ['sqrt', 'log2']
# }

scorer = make_scorer(recall_score, pos_label=1)

search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=3, n_jobs=-1)

# model.fit(X_train, y_train)
search.fit(X_train, y_train)

# print(f"Najlepsze parametry: {search.best_params_}")

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
# print("Wynik modelu:")
# print(classification_report(y_test, y_pred))

joblib.dump(best_model, 'churn_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')



# --------- test dla danego klienta ---------
test_customer_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 20,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 50.0,
    'TotalCharges': 600.0
}

df_test = pd.DataFrame([test_customer_data])

df_test = pd.get_dummies(df_test)

df_test = df_test.reindex(columns=X.columns, fill_value=0)

prediction = best_model.predict(df_test)
probability = best_model.predict_proba(df_test)

if prediction[0] == 1:
    print(f"Klient odejdzie.")
    print(f"Pewność modelu: {probability[0][1]*100:.2f}%")
else:
    print(f"Klient prawdopodobnie zostanie.")
    print(f"Pewność modelu: {probability[0][0]*100:.2f}%")