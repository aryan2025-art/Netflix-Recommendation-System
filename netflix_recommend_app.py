#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pip install scikit-surprise wordcloud


# In[3]:


import pandas as pd

titles = pd.read_csv(
    'movie_titles.csv',
    encoding='ISO-8859-1',
    header=None,
    names=['MovieID', 'Year', 'Title'],
    quotechar='"',
    on_bad_lines='skip'  # ✅ updated replacement for error_bad_lines
)

titles.head()



# In[4]:


import pandas as pd
import os

def parse_ratings(file_path):
    movie_id = None
    user_ids, ratings, dates, movie_ids = [], [], [], []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.endswith(':'):  # MovieID line
                movie_id = int(line[:-1])
            else:
                user_id, rating, date = line.split(',')
                user_ids.append(int(user_id))
                ratings.append(float(rating))
                dates.append(date)
                movie_ids.append(movie_id)

    return pd.DataFrame({'UserID': user_ids,
                         'MovieID': movie_ids,
                         'Rating': ratings,
                         'Date': dates})


# In[5]:


# Example: parse first file
ratings1 = parse_ratings('combined_data_1.txt')

# For full dataset (all 4 files), parse and concat
ratings = pd.concat([parse_ratings(f'combined_data_{i}.txt') for i in range(1, 5)],
                    ignore_index=True)




# In[6]:


print(ratings.head(10))


# In[8]:


print(ratings.shape)
print(ratings['UserID'].nunique())
print(ratings['MovieID'].nunique())
print(ratings['Rating'].describe())


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sample 500,000 rows (adjustable to fit your RAM)
sampled = ratings.sample(n=100_000, random_state=42)

sns.histplot(sampled['Rating'], bins=5, kde=False)
plt.title('Rating Distribution (Sampled)')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# In[11]:


top_movies = ratings['MovieID'].value_counts().head(10).index
print(titles[titles['MovieID'].isin(top_movies)])


# In[12]:


titles['combined'] = titles['Title'].astype(str) + ' ' + titles['Year'].astype(str)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(titles['combined'])


# In[14]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[15]:


vectorizer = CountVectorizer(stop_words='english')
count_matrix = vectorizer.fit_transform(titles['Title'])


# In[16]:


cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[17]:


indices = pd.Series(titles.index, index=titles['Title']).drop_duplicates()


# In[18]:


def content_recommender(title, cosine_sim=cosine_sim):
    if title not in indices:
        return f"'{title}' not found in dataset."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]

    return titles['Title'].iloc[movie_indices].reset_index(drop=True)


# In[19]:


content_recommender("Toy Story")


# In[20]:


pip install scikit-surprise


# In[21]:


get_ipython().system('pip uninstall -y numpy')
get_ipython().system('pip install numpy==1.26.4')


# In[22]:


get_ipython().system('pip install numpy==1.26.4')



# In[23]:


 import numpy
print(numpy.__version__)


# In[24]:


import numpy
print(numpy.__version__)


# In[25]:


from surprise import Dataset, Reader, SVD


# In[26]:


from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy


# In[37]:


print("Original ratings shape:", ratings.shape)
print("Sample data:\n", ratings.head())


# In[39]:


import pandas as pd
import numpy as np

# Generate synthetic data with 150k rows
np.random.seed(42)
num_users = 10000
num_movies = 2000
num_ratings = 150000

ratings = pd.DataFrame({
    'UserID': np.random.randint(1, num_users+1, num_ratings),
    'MovieID': np.random.randint(1, num_movies+1, num_ratings),
    'Rating': np.random.uniform(1, 5, num_ratings).round(1)
})

print("Synthetic ratings shape:", ratings.shape)
print(ratings.head())


# In[42]:


import pandas as pd
import numpy as np
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

# Generate synthetic data with 150k rows
np.random.seed(42)
num_users = 10000
num_movies = 2000
num_ratings = 150000

ratings = pd.DataFrame({
    'UserID': np.random.randint(1, num_users + 1, num_ratings),
    'MovieID': np.random.randint(1, num_movies + 1, num_ratings),
    'Rating': np.random.uniform(1, 5, num_ratings).round(1)
})

print("Synthetic ratings shape:", ratings.shape)
print(ratings.head())

# Setup Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

print("Trainset and testset created.")
print(f"Trainset size: {trainset.n_ratings}")
print(f"Testset size: {len(testset)}")


# In[43]:


from surprise import SVD, accuracy

# Create and train the SVD model
model = SVD(random_state=42)
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model performance
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"\nSVD Model Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")


# In[44]:


import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, SVDpp, KNNBasic
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

# Assuming 'ratings' DataFrame is ready with columns: UserID, MovieID, Rating

# Load data into Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# Train/test split
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# 1. Hyperparameter tuning for SVD using GridSearchCV
param_grid = {
    'n_factors': [50, 100],          # Number of latent factors
    'n_epochs': [20, 30],            # Number of SGD iterations
    'lr_all': [0.005, 0.010],        # Learning rate
    'reg_all': [0.02, 0.05]          # Regularization term
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, joblib_verbose=1, n_jobs=-1)
gs.fit(data)

print(f"Best RMSE score: {gs.best_score['rmse']}")
print(f"Best params: {gs.best_params['rmse']}")

# 2. Train SVD with best params on full training set
best_svd = SVD(**gs.best_params['rmse'])
best_svd.fit(trainset)

# 3. Evaluate on test set
predictions = best_svd.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"Tuned SVD RMSE: {rmse}")
print(f"Tuned SVD MAE: {mae}")

# 4. Generate top-N recommendations for a few sample users
def get_top_n_recommendations(model, trainset, user_ids, n=5):
    all_movie_ids = set(ratings['MovieID'].unique())
    top_n = {}

    for uid in user_ids:
        # Get movies user has already rated
        try:
            user_inner_id = trainset.to_inner_uid(uid)
        except ValueError:
            # User not in trainset
            print(f"User {uid} not found in training set.")
            continue

        rated_movie_inner_ids = set([j for (j, _) in trainset.ur[user_inner_id]])
        rated_movie_ids = set([trainset.to_raw_iid(iid) for iid in rated_movie_inner_ids])
        unrated_movies = all_movie_ids - rated_movie_ids

        # Predict ratings for unrated movies
        predictions = [ (movie_id, model.predict(uid, movie_id).est) for movie_id in unrated_movies ]
        # Sort by predicted rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = predictions[:n]

    return top_n

sample_users = ratings['UserID'].sample(3, random_state=42).unique()
top_recs = get_top_n_recommendations(best_svd, trainset, sample_users, n=5)

for uid, recs in top_recs.items():
    print(f"\nTop 5 recommendations for user {uid}:")
    for movie_id, est_rating in recs:
        print(f"  MovieID {movie_id} with predicted rating {est_rating:.2f}")

# 5. (Optional) Train a simple KNNBasic model and evaluate
knn = KNNBasic()
knn.fit(trainset)
knn_preds = knn.test(testset)
print("\nKNNBasic Model Performance:")
accuracy.rmse(knn_preds)
accuracy.mae(knn_preds)


# In[48]:


from surprise import Dataset, Reader

def get_top_n_recommendations(algo, user_id, ratings_df, n=5):
    """
    Generate top-N movie recommendations for a given user.

    Parameters:
    - algo: Trained Surprise algorithm (e.g., SVD)
    - user_id: int or str, the user for whom recommendations are made
    - ratings_df: pd.DataFrame with columns ['UserID', 'MovieID', 'Rating']
    - n: int, number of recommendations to return

    Returns:
    - List of tuples: [(MovieID, predicted_rating), ...] sorted by rating desc
    """
    # Get all unique movies
    all_movies = ratings_df['MovieID'].unique()

    # Get movies already rated by the user
    rated_movies = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].unique()

    # Movies to predict ratings for (not yet rated by user)
    movies_to_predict = [movie for movie in all_movies if movie not in rated_movies]

    # Predict ratings for all these movies
    predictions = [algo.predict(user_id, movie) for movie in movies_to_predict]

    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Take top-n
    top_n = predictions[:n]

    # Format as (MovieID, predicted_rating)
    top_n_recommendations = [(pred.iid, pred.est) for pred in top_n]

    return top_n_recommendations


# In[49]:


from surprise import Dataset, Reader

def get_top_n_recommendations(algo, user_id, ratings_df, n=5):
    # Get all unique movie IDs
    all_movie_ids = ratings_df['MovieID'].unique()

    # Get movies the user has already rated
    rated_movies = ratings_df[ratings_df['UserID'] == user_id]['MovieID'].unique()

    # Movies not yet rated by the user
    movies_to_predict = [m for m in all_movie_ids if m not in rated_movies]

    # Predict ratings for all unrated movies
    predictions = []
    for movie_id in movies_to_predict:
        pred = algo.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    # Sort predictions by estimated rating, descending
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Return top N movie IDs with predicted ratings
    return predictions[:n]


# In[51]:


from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Assuming you have a ratings DataFrame with columns UserID, MovieID, Rating
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

svd_model = SVD()
svd_model.fit(trainset)


# In[52]:


user_id = 1616
top_5 = get_top_n_recommendations(svd_model, user_id, ratings, n=5)

print(f"Top 5 recommendations for user {user_id}:")
for movie, rating in top_5:
    print(f"  MovieID {movie} with predicted rating {rating:.2f}")


# In[53]:


pip install streamlit


# In[54]:


import streamlit as st
import pandas as pd

# Your existing functions like get_top_n_recommendations()

st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, max_value=10000, step=1)

if st.button("Get Recommendations"):
    recommendations = get_top_n_recommendations(svd_model, user_id, ratings, n=5)
    if recommendations:
        st.write(f"Top 5 recommendations for User {user_id}:")
        for movie, rating in recommendations:
            st.write(f"MovieID {movie} with predicted rating {rating:.2f}")
    else:
        st.write("No recommendations found for this user.")


# In[56]:


import os
os.getcwd()


# In[1]:


get_ipython().system('jupyter nbconvert --to script netflix_recommendation.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script NETF.ipynb')

