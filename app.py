import json
import pickle
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, url_for, render_template

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('best_xgb_model.pkl', 'rb'))
scaler = pickle.load(open('standard_scaler.pkl', 'rb'))

columns_to_scale = ['n_tokens_content',
       'n_unique_tokens', 'n_non_stop_unique_tokens','num_hrefs',
       'average_token_length', 'num_keywords','self_reference_min_shares', 'self_reference_max_shares',
       'self_reference_avg_sharess','LDA_00','LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 'global_subjectivity',
       'global_sentiment_polarity', 'global_rate_positive_words',
       'global_rate_negative_words', 'avg_positive_polarity']

@app.route('/')
def home():
    column_names = ['n_tokens_content', 'n_unique_tokens', 'n_non_stop_unique_tokens',
                    'num_hrefs', 'average_token_length', 'kw_max_min', 'kw_avg_min',
                    'kw_avg_max', 'kw_max_avg', 'kw_avg_avg', 'self_reference_min_shares',
                    'self_reference_avg_sharess', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03',
                    'LDA_04', 'global_subjectivity', 'global_sentiment_polarity',
                    'global_rate_positive_words', 'global_rate_negative_words',
                    'avg_positive_polarity', 'avg_negative_polarity']
    return render_template('index.html', column_names=column_names)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # Extract only the columns to scale
    data_to_scale = {key: data[key] for key in columns_to_scale}
    new_data = scaler.transform(np.array(list(data_to_scale.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    predictions_original = np.expm1(output[0])
    return jsonify({'predictions_original': predictions_original.tolist()})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    # Extract only the columns to scale
    data_to_scale = {columns_to_scale[i]: data[i] for i in range(len(columns_to_scale))}
    final_input = scaler.transform(np.array(list(data_to_scale.values())).reshape(1, -1))
    output = regmodel.predict(final_input)
    predictions_original = np.expm1(output[0])
    return render_template("index.html", prediction_text="The predicted number of shares is {}".format(predictions_original))

if __name__ == "__main__":
    app.run(debug=True)
