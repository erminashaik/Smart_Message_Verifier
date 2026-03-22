import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#using pandas
msgs ={
    'txt': ["Congratulations! You've been selected as our lucky winner of a brand new iPhone 15. Claim your prize now: [Link]" ,
            "You won a $1000 amazon gift card! Click here to claim your reward today before it expires.",
           "Bank Alert: Unusual activity detected on your account." ,
           "Your card has been temporarily locked. Log in here to secure it: [Link]" ,
           "URGENT: Your account has been suspended due to a billing issue. Update your payment information immediately to avoid deactivation.",
           "IRS Notice: You are eligible for a tax refund of 50,000/- rupees. Click here to submit your bank details and receive your payout.",
           "congratulation you just won a lottery",
           "hey, dear can we meet today",
           " where are you",
           "i'm waiting for you..",
            "hi",
            "hello",
            "good morning have a great day ahead"],
    'scoring': [1,1,1,1,1,1,1,2,2,2,2,2,2]
}
df = pd.DataFrame(msgs)
#vectorizing the info

vectorizer =CountVectorizer()
x=vectorizer.fit_transform (df['txt'])
y=df["scoring"].values
model=MultinomialNB()
model.fit(x,y)
#interactive

st.title(" Smart Message Verifier")
st.write("Good Morning! Hope you have a Great day ahead :)")
user_input = st.text_input("Paste your message to verify:", placeholder="Type here...")
if user_input:
    user_vec= vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]
    st.info("Your Message Was Analyzed...")
    if prediction ==1:
         st.error("Result: ALERT! The Message Was Found To Be Suspicious. Double Check Before Procceding.")
    else:
        st.success("Result: The Message Was Found To Be Safe.")

st.divider()
st.markdown("<p style='text-align: center;'><i>\"Think before you click, Don't let the rush lead to a loss\"</i></p>",unsafe_allow_html=True)
st.caption("Developed and Designed By: SHAIK ERMINA | 2026 " )
