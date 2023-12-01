import base64

from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import plotly.express as px
import pandas as pd
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import seaborn as sns

app = Flask(__name__)

# Load the RandomForestRegressor model
with open('model_random_forest.pkl', 'rb') as file:
    model = pickle.load(file)

# Load car data (replace 'cleaned_cars_data.csv' with your actual CSV file)
cars_data = pd.read_csv('cleaned_cars_data.csv')
selected_features = ['year', 'hp', 'cylinders', 'doors', 'highway_mpg', 'city_mpg', 'popularity', 'price']
subset_data = cars_data[selected_features]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from the form
            hp = float(request.form['hp'])
            cylinders = int(request.form['cylinders'])
            year = int(request.form['year'])
            doors = float(request.form['doors'])

            # Create a NumPy array from the user input for prediction
            input_features = np.array([[hp, cylinders, year, doors]])

            # Make a prediction using the loaded model
            prediction = model.predict(input_features)

            # Render HTML template with prediction and input features
            return render_template('prediction_result.html', hp=hp, cylinders=cylinders,
                                   year=year, doors=doors, prediction=round(prediction[0], 2))

        except Exception as e:
            # Handle exceptions or errors here
            print(str(e))
            return jsonify({'error': 'An error occurred during prediction.'})


@app.route('/plot')
def plot():
    # Assuming cars_data and subset_data are defined somewhere

    def generate_plot_div():
        # Generate Plotly figures
        fig1 = px.scatter(cars_data, x='doors', y='price', color='doors')
        fig1.update_layout(width=400, height=400, title="Scatter Plot Price vs Doors")
        div_fig1 = fig1.to_html(full_html=False)

        fig2 = px.scatter(cars_data, x='cylinders', y='price', color='cylinders')
        fig2.update_layout(width=400, height=400,title="Scatter Plot Price vs cylinders")
        div_fig2 = fig2.to_html(full_html=False)

        fig3 = px.scatter(cars_data, x='hp', y='price')
        fig3.update_layout(width=400, height=400,title="Scatter Plot Price vs hp")
        div_fig3 = fig3.to_html(full_html=False)

        return div_fig1, div_fig2, div_fig3

    def generate_base64_image():
        correlation_matrix = subset_data.corr()

        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                               annot_kws={"size": 10})
        plt.title('Correlation Matrix', fontsize=14)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

        with open('correlation_matrix.png', 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return img_base64

    div_fig1, div_fig2, div_fig3 = generate_plot_div()
    img_base64 = generate_base64_image()

    return render_template('dashboard.html', div_fig1=div_fig1, div_fig2=div_fig2, div_fig3=div_fig3, img_base64=img_base64)


if __name__ == '__main__':
    app.run(debug=True)
