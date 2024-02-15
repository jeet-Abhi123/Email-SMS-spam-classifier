#SMS spam classifier
import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

add_selectbox = st.sidebar.radio(
    "How would you rate our SMS Spam detection system?",
    ("Excellent", "Average", "Bad")
)

st.title("SMS Spam Classifier")



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    #1.preprocess
    transformed_sms = transform_text(input_sms)
    #2.vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.Display
    if result==1 :
        st.header("SPAM message!")
    else:
        st.header("Not a Spam message")

