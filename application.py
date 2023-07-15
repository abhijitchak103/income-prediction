from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
import mlflow
import mlflow.sklearn


application = Flask(__name__)


app = application


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    
    else:
        data = CustomData(
            age = float(request.form.get('age')), 
            children = int(request.form.get('children')),
            bmi = float(request.form.get('bmi')),            
            sex = request.form.get('sex'), 
            region = request.form.get('region'),
            smoker = request.form.get('smoker')
        )


        final_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict_data(final_df)

        result = round(pred[0])

        return render_template('results.html', final_result=f"{result}")
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)