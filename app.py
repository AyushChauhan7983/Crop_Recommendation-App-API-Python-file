import pickle
import numpy as np

from flask import Flask,request,jsonify

model = pickle.load(open('rf.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World!"

@app.route('/predict',methods=['POST'])
def predict():
    nitrogen = request.form.get('nitrogen')
    phosphorous = request.form.get('phosphorous')
    potassium = request.form.get('potassium')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    pH = request.form.get('pH')
    rainfall = request.form.get('rainfall')

    feature_list = [nitrogen, phosphorous, potassium, temperature, humidity, pH, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    prediction = model.predict(single_pred)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Blackgram", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Watermelon", 16: "Lentil", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return jsonify({'crop':crop})

if __name__ == '__main__':
    app.run(debug=True)