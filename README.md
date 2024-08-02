# Recommendation Engine

This project is a recommendation engine built using Flask and Surprise library for generating product recommendations based on synthetic user-product interactions. The dataset used is from an e-commerce site (Myntra).

## Features

- Load and preprocess product data
- Generate synthetic user-product interactions
- Train a recommendation model using SVD
- Provide top N product recommendations for a user

## Requirements

- Python 3.x
- Flask
- Flask-Caching
- pandas
- scikit-surprise

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/benjamin-davis1/recommendation_engine.git
    cd recommendation-engine
    ```

2. Install the required packages:
    ```sh
    pip install Flask Flask-Caching pandas scikit-surprise
    ```

3. Place the dataset file (`myntra_products_catalog.csv`) in the root directory of the project.

## Running the Application

1. Run the application:
    ```sh
    python data_collection.py
    ```

2. The server will start at `http://127.0.0.1:5000`.

3. To get recommendations, make a GET request to the `/recommend` endpoint with a `user_id` parameter:
    ```sh
    curl "http://127.0.0.1:5000/recommend?user_id=user1"
    ```

4. The top N recommendations Id and score will be generated for that user.

