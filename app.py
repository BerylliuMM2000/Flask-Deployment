import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    def outputConvert(output):
        if output < 1.5: return "Setosa"
        if output < 2.5: return "Versicolor"
        return "Virginica"
    output = outputConvert(prediction)

    return render_template('index.html', prediction_text='Iris type should be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)