# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:20:11 2020

@author: admin
"""

from flask import Flask, render_template, request
#from sklearn.feature_extraction.text import CountVectorizer
import pickle

# email file
filename = 'spam_mail.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('emvec.pkl','rb'))

#sms file
smsfile = 'spam_sms.pkl'
sms_clf = pickle.load(open(smsfile,'rb'))
smscv = pickle.load(open('smsvec.pkl','rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/email")
def email():
    return render_template('email.html')

@app.route("/predictmail", methods=['POST'])
def predictmail():
    if request.method == 'POST':
        message = request.form["message"]
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)

@app.route("/sms")
def sms():
    return render_template('message.html')

@app.route("/predictsms", methods=['POST'])
def predictsms():
    if request.method == 'POST':
        message = request.form["message"]
        data = [message]
        test = smscv.transform(data).toarray()
        predi = sms_clf.predict(test)
    return render_template("result.html", prediction = predi)

@app.route("/contact")
def contact():
    return render_template('contact.html')    

if __name__ == "__main__":
    app.run(debug=True)
