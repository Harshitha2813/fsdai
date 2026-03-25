import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df = pd.read_excel(r"C:\Users\karna\Downloads\Twitter_Datas.csv.xlsx")
df.dropna(subset=['clean_text', 'category'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)
X = df['clean_text']
y = df['category']
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Sentiment Analyzer Ready! Type 'exit' to stop.")
while True:
    user_input = input("Enter Tweet: ").lower()
    if user_input == "exit":
        print("program stopped")
        break
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]

    # ✅ Safer mapping (depends on dataset labels)
    if prediction == 1:
        print("Sentiment: Positive")
    elif prediction == 0:
        print("Sentiment: Neutral")
    else:
        print("Sentiment: Negative")