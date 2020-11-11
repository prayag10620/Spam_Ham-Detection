# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:20:11 2020

@author: admin
"""
import numpy as np
from flask import Flask, render_template, request
#from flask_mail import Mail, Message
#from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import pickle
#from config import mail_username, mail_password

SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100

# email file
filename = 'spam_mail.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('emvec.pkl','rb'))

#sms file
smsfile = 'spam_sms.pkl'
sms_clf = pickle.load(open(smsfile,'rb'))
smscv = pickle.load(open('smsvec.pkl','rb'))

#ycomment file
yclf = tf.keras.models.load_model('rnn9422.h5')
ycv = pickle.load(open('nntok.pkl','rb'))

app = Flask(__name__)

app.config['SECRET_KEY'] = "MySecureSecretKey"
#app.config['MAIL_SERVER'] = "smtp-mail.outlook.com"
#app.config['MAIL_PORT'] = 587
#app.config['MAIL_USE_TLS'] = True
#app.config['MAIL_USE_SSL'] = False
#app.config['MAIL_USERNAME'] = mail_username
#app.config['MAIL_PASSWORD'] = mail_password
#mail = Mail(app)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

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
        Prob = clf.predict_proba(vect) * 100
        ham = round(Prob[0][0],2)
        spam = round(Prob[0][1],2) 
        print(ham,spam)
    return render_template('result.html', prediction = my_prediction, Ham = ham, Spam = spam)

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
        Prob = sms_clf.predict_proba(test) * 100
        ham = round(Prob[0][0],2)
        spam = round(Prob[0][1],2) 
        #print(ham,spam)
    return render_template("result.html", prediction = predi, Ham = ham, Spam = spam)

@app.route("/ycomment")
def ycomment():
    return render_template('youtube.html')

@app.route("/predictcomment", methods=['POST'])
def predictcomment():
    if request.method == 'POST':
        message = request.form["message"]
        sequence = ycv.texts_to_sequences([message])
        # pad the sequence
        sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
        # get the prediction
        prediction = yclf.predict(sequence)[0]
        pred = yclf.predict(sequence)[0] * 100
        ham = round(pred[0],2)
        spam = round(pred[1],2)
        # one-hot encoded vector, revert using np.argmax
        y_prediction = np.argmax(prediction)
    return render_template('result.html', prediction = y_prediction, Ham = ham, Spam = spam)
    

@app.route("/contact", methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        #name = request.form.get("name")
        #email = request.form.get("email")
        #phone = request.form.get("phone")
        #message = request.form.get("message")
        #msg = Message(subject=f"mail from {name}", body=f"Name: {name}\nE-mail: {email}\nPhone: {phone}\n\n{message}", sender=mail_username, recipients=["raghavagrawal1777@gmail.com"])
        #mail.send(msg)
       
        return render_template('contact.html', success=True)
        

    return render_template('contact.html') 
       
if __name__ == "__main__":
    app.run(debug=True)
