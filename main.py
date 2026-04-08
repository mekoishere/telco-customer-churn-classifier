import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)
    
X = df.drop('Churn', axis=1)
y = df['Churn']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# dużo więcej klientów zostaje niż odchodzi więc trzeba znaleźć model który sobie z tym poradzi
# musi umieć przewidzieć 1 a nie tylko 0 w kolumnie churn, chcemy jak nawiększy recall w classification report

# model = RandomForestClassifier(random_state=42) - recall 0.44
# model = RandomForestClassifier(random_state=42, class_weight='balanced') - recall 0.45
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # recall - 0.54
model = HistGradientBoostingClassifier(class_weight='balanced', random_state=42) # recall - 0.76 - ten w miarę umie znaleźć tych którzy odejdą, ale czasem zaznacza błędnie klientów którzy zostaliby, biznesowo jest to lepsza sytuacja niż stracenie klienta

model.fit(X_train, y_train)
    
y_pred = model.predict(X_test)
print("Wynik modelu:")
print(classification_report(y_test, y_pred))
