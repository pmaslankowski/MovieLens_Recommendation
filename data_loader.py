import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATASET_DIRECTORY = 'ml-20m/'

def load_ratings():
    ''' returns pandas dataframe with all ratings '''
    return pd.read_csv(DATASET_DIRECTORY + 'ratings.csv')

def load_user_item_matrix():
    ''' returns sparse user/item matrix with following shape: (number_of_users, number_of_movies) '''

    def remove_gaps_from_movie_id(ratings):
        movieIdMap = { k : v for (v, k) in enumerate(sorted(list(ratings['movieId'].unique()))) }
        ratings['movieId'] = ratings['movieId'].apply(lambda x: movieIdMap[x])
    
    def get_user_item_matrix(ratings):
        rating_values = ratings['rating'].tolist()
        us = (ratings['userId']-1).tolist()
        mov = ratings['movieId'].tolist()
        return csr_matrix((rating_values, (us, mov)), shape=(max(us)+1, max(mov)+1))
    
    ratings = pd.read_csv(DATASET_DIRECTORY + 'ratings.csv')
    remove_gaps_from_movie_id(ratings)
    return get_user_item_matrix(ratings)