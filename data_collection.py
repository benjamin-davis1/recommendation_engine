import pandas as pd
from flask import Flask, request, jsonify
from flask_caching import Cache
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

def load_data():
    data = pd.read_csv('myntra_products_catalog.csv')
    
    data = data[['ProductID', 'ProductName', 'ProductBrand', 'Gender', 'Price (INR)', 'Description', 'PrimaryColor']]
    data.columns = ['product_id', 'product_name', 'product_brand', 'gender', 'price', 'description', 'primary_color']
    
    return data

def prepare_data(data):
    # Generate synthetic user-product interactions
    ratings = pd.DataFrame({
        'user_id': ['user1', 'user2', 'user3', 'user1', 'user2', 'user3', 'user1', 'user2', 'user3'],
        'product_id': data['product_id'].sample(9).values,
        'rating': [5, 4, 3, 5, 4, 3, 4, 5, 2]
    })
    
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    
    return trainset, testset, ratings

def train_model(trainset):
    algo = SVD()
    algo.fit(trainset)
    return algo

# Get top N recommendations for a user
def get_top_n_recommendations(user_id, algo, data, ratings, n=10):
    unique_products = data['product_id'].unique()
    user_rated_products = ratings[ratings['user_id'] == user_id]['product_id'].unique()
    
    predictions = []
    for product_id in unique_products:
        if product_id not in user_rated_products:
            predictions.append((product_id, algo.predict(user_id, product_id).est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/recommend', methods=['GET'])
@cache.cached(timeout=60)
def recommend():
    user_id = request.args.get('user_id')
    if user_id:
        recommendations = get_top_n_recommendations(user_id, algo, data, ratings)
        recommendations = [(int(product_id), float(score)) for product_id, score in recommendations]
        return jsonify(recommendations)
    return jsonify({'error': 'User ID is required'}), 400

if __name__ == '__main__':
    data = load_data()
    trainset, testset, ratings = prepare_data(data)
    algo = train_model(trainset)
    app.run(debug=True)
