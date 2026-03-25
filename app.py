from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

df = pd.read_csv(r"C:\Users\karna\fsdai\Sentiment Analysis\Twitter_Data.csv")
df.dropna(subset=['clean_text', 'category'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)

X = df['clean_text']
y = df['category']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec,y)

@app.route("/")
def home():
    return render_template("indexx.html")

@app.route("/predict", methods=["POST"])
def analysis():
    data=request.get_json()
    user_input = data["text"]
    
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    if prediction == 1:
        result="Positive"
    elif prediction == 0:
        result="Neutral"
    else:
        result="Negative"

    return jsonify({"result":result})

if __name__ == "__main__":
    app.run(debug=True)


