# Load libraries
from flask import Flask, jsonify
import joblib
from flask_bootstrap import Bootstrap
from flask import render_template, redirect, url_for, request, send_from_directory, flash
import pandas as pd
import os
import sklearn



#Set up Flask
TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('static')
# instantiate flask 
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key="Pet_Adoption"
app.config['SESSION_COOKIE_SECURE'] = False
Bootstrap(app)
clf = joblib.load(open("random_forest_clf.pkl", "rb")) # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("random_forest_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

# Home page, renders the intro.html template

@app.route('/')
def home():
    return render_template('index.html', title='homepage')

@app.route('/machinelearning')
def ML():
    return render_template('ML.html', title='MachineLearning')

@app.route('/spervised', methods=['GET', 'POST'])
def spervised():
    if request.method == 'POST':
        primary_breed= int(request.form['primary_breed'])
        color= int(request.form['color'])
        age= int(request.form['age'])
        gender= int(request.form['gender'])
        size= int(request.form['size'])
        coat= int(request.form['coat'])
        mix_breed= bool(request.form['mix_breed'])
        house_trained= bool(request.form['house_trained'])
        spayed_neutered= bool(request.form['spayed_neutered'])
        special_need= bool(request.form['special_need'])
        shot_current= bool(request.form['shot_current'])
        gw_childern= int(request.form['gw_childern'])
        gw_dog= int(request.form['gw_dog'])
        gw_cat= int(request.form['gw_cat'])
        tag= bool(request.form['tag'])
        photo= bool(request.form['photo'])

        feature_columns=[mix_breed,house_trained,spayed_neutered,special_need,shot_current,tag,photo,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        feature_columns[primary_breed]=1
        feature_columns[color]=1
        feature_columns[age]=1
        feature_columns[gender]=1
        feature_columns[size]=1
        feature_columns[coat]=1
        feature_columns[gw_childern]=1
        feature_columns[gw_dog]=1
        feature_columns[gw_cat]=1

        x_feature=[feature_columns]

        print(x_feature)

        query = pd.DataFrame(x_feature)


        print(query)
        # query = query.reindex(columns=model_columns, fill_value=0)

        prediction = list(clf.predict(query))

        print(prediction)
        image=f"static/assets/{prediction[0]}.png"
        result=prediction[0].replace("_"," ").upper()
        flash(result)
        
    return render_template('ML.html', title='spervised',image=image)





if __name__ == '__main__':

    app.run(debug=True)
