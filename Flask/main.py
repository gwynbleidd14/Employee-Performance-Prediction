import numpy as np
import requests
import pickle
import io
import base64
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open("IBM Files\model_xgb.pkl", "rb"))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/submit")
def submit():
    return render_template('submit.html')

@app.route("/pred", methods = ['POST'])
def pred():
    # Collecting form data
    quarter = request.form['quarter']
    department = request.form['department']
    day = request.form['day']
    team = request.form['team']
    targeted_productivity = request.form['targeted_productivity']
    smv = request.form['smv']
    over_time = request.form['over_time']
    incentive = request.form['incentive']
    idle_time = request.form['idle_time']
    idle_men = request.form['idle_men']
    no_of_style_change = request.form['no_of_style_change']
    no_of_workers = request.form['no_of_workers']
    month = request.form['month']
    
    # Preparing input for model
    total = [[int(quarter), int(LabelEncoder().fit_transform([department])[0]), 
                  int(LabelEncoder().fit_transform([day])[0]), int(LabelEncoder().fit_transform([team])[0]),
                  float(targeted_productivity), float(smv), int(over_time), int(incentive),
                  float(idle_time), int(idle_men), int(no_of_style_change), float(no_of_workers), 
                  int(LabelEncoder().fit_transform([month])[0])]]
    
    # Prediction
    prediction = model.predict(total)[0]  # Assuming this returns a scalar value
    print(f"Prediction: {prediction}")
    
    if prediction <= 0.3:
        text = 'The employee is Averagely Productive.'
    elif 0.3 < prediction <= 0.8:
        text = 'The employee is Medium Productive.'
    else:
        text = 'The employee is Highly Productive.'

    graphs = []

    # Bar chart
    plt.figure()
    categories = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
    values = [targeted_productivity, smv, over_time, idle_time]
    plt.bar(categories, [float(v) for v in values], color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Employee Productivity Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Scatter plot
    plt.figure()
    plt.scatter([1, 2, 3, 4], [float(targeted_productivity), float(smv), float(over_time), float(idle_time)])
    plt.xlabel('Parameters')
    plt.ylabel('Values')
    plt.title('Scatter Plot of Employee Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Pie chart
    plt.figure()
    sizes = [float(targeted_productivity), float(smv), float(over_time), float(idle_time)]
    labels = ['Targeted Productivity', 'SMV', 'Over Time', 'Idle Time']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Pie Chart of Employee Parameters')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    # Histogram
    plt.figure()
    data = np.random.randn(100)
    plt.hist(data, bins=20, color='purple')
    plt.xlabel('Random Data')
    plt.ylabel('Frequency')
    plt.title('Histogram of Random Data')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    graphs.append(graph_image)
    plt.close()

    return render_template('submit.html', prediction_text = text, graphs = graphs)

if __name__ == "__main__":
    app.run(debug=True)