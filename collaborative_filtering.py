import numpy as np
from math import *
import sys

class collaborative_filtering(object):
    def __init__(self, test_file, out_file):
        self.training_data = np.array([])
        self.train_udf = np.array([])
        self.test_file = test_file
        self.out_file = out_file
        self.testing_users = {}
        self.testing_movies = {}

    def fetch_data(self):
        self.training_data = np.loadtxt("train.txt")
        t = np.loadtxt(self.test_file)
        #Organize test data to easier to work with
        for item in t:
            self.testing_users[item[0]] = [0]*1000
        for item in t:
            self.testing_users[item[0]][int(item[1])-1] = int(item[2]) #Movie is -1 index
        #Get a dictionary of desired movie predictions for each user
        for item in t:
            if item[2] == 0:
                if item[0] in self.testing_movies:
                    self.testing_movies[item[0]].append(item[1])
                else:
                    self.testing_movies[item[0]] = [item[1]]

        #UDF matrix formation
        udf = np.array([])
        counts = np.zeros(shape=(1000))
        for item in self.training_data:
            for i, j in enumerate(item):
                if j != 0:
                    counts[i] = counts[i] + 1
        for i,j in enumerate(counts):
            if j != 0:
                counts[i] = log10(200/j)
            else:
                counts[i] = 0
        self.train_udf = self.training_data * counts


    def cosine_similarity(self, person_x, person_y):
        assert person_x.size == person_y.size
        numerator =  sum(x*y for x,y in zip(person_x, person_y))
        denominator = (sqrt(sum(x*x for x in person_x))) * (sqrt(sum(y*y for y in person_y)))
        return numerator/denominator

    def pearson_similarity(self, person_x, person_y, movie):
        assert person_x.size == person_y.size
        pxm = sum(float(x) for x in person_x if x != 0)/np.count_nonzero(person_x)
        pym = sum(float(y) for y in person_y if y != 0)/np.count_nonzero(person_y)
        numerator = sum(((float(x)-pxm)*(float(y)-pym) for x,y in zip(person_x, person_y)))
        denominator = (sqrt(sum(((float(x)-pxm))**2 for x in person_x))) * (sqrt(sum((float(y)-pym)**2 for y in person_y)))
        return float(numerator)/float(denominator)

    def adjusted_cosine_similarity(self, wanted_movie, movie, averages):
        assert wanted_movie.size == movie.size
        wanted_movie_mm = np.zeros(shape=(200))
        movie_mm = np.zeros(shape=(200))
        #Subract average user rating from the movies
        i = 0
        for wm, m in zip(wanted_movie, movie):
            if m != 0:
                movie_mm[i] = (m - averages[i])
            if wm != 0:
                wanted_movie_mm[i] = (wm - averages[i])
            i+=1

        numerator = sum(((float(x)*float(y)) for x, y in zip(movie_mm, wanted_movie_mm)))
        denominator = (sqrt(sum((float(x))**2 for x in movie_mm))) * (sqrt(sum(float(y)**2 for y in wanted_movie_mm)))
        try:
            return float(numerator)/ float(denominator)
        except:
            return 0.0

    def reccomend_user_based_cosine(self, person, movie, data):
        similarities = {}
        ratings = {}

        #Get similarity scores for users and their ratings
        for i, row in enumerate(data):
            if row[int(movie)-1] != 0:
                s = self.cosine_similarity(person, row)
                if (s!=0):
                    similarities[i] = s #Array of similarity vals
                    ratings[i] = row[int(movie)-1]   #array of these peoples ratings
        #sorted_users = sorted(similarities.iterkeys(), key=lambda k:similarities[k], reverse=True)
        #divide by total similaries, each person score minus persons mean
        s_ratings = 0
        n_ratings = 0
        for item in person:
            if (item != 0):
                s_ratings+=item
                n_ratings+=1

        average = float(s_ratings)/float(n_ratings)
        if len(similarities) < 2:
            return average

        #Try removing similarities of 1
        try:
            rating = self.rate(similarities, ratings)
            if rating <=0.5:
                rating = 1
            if isnan(rating):
                rating = average
            if rating > 5:
                rating = 5
            return rating
        #Exceptions when no similar users are found
        except:
            return average
    def reccomend_user_based_pearson(self, person, movie, data):
        similarities = {}
        temp_similarities = {}
        ratings = {}
        averages = {}
        data_tmp = data
        person = np.delete(person, movie-1)
        #only add averages of rated items
        s_ratings = 0
        n_ratings = 0
        for item in person:
            if (item != 0):
                s_ratings+=item
                n_ratings+=1

        average = float(s_ratings)/float(n_ratings)
        for i, row in enumerate(data_tmp):
            if row[movie-1] != 0:
                ratings[i] = row[movie-1]
                averages[i] = sum(float(x) for x in row if x!=0)/np.count_nonzero(row)
                row = np.delete(row, movie-1)
                s = self.pearson_similarity(person, row, movie)
                similarities[i] = s #Array of similarity vals

        """
        #Remove some neighbors
        similarities = {key: temp_similarities[key] for key in temp_similarities}
        for key in temp_similarities:
            if temp_similarities[key] < 0.05:
                similarities.pop(key, None)
                ratings.pop(key, None)
                averages.pop(key, None)
        """
        if len(similarities) == 1:
            return average

        try:
            rating = self.rate_minus_mean(similarities, ratings, averages)
            if isnan(rating):
                return average
            if rating+average > 5:
                return 5.0
            if (rating+average) < 0.5:
                return 1.0
            return rating+average
            #Exceptions when no similar users are found
        except:
            return average


    def reccomend_user_based_pearson_udf(self, person, movie, data):
        similarities = {}
        ratings = {}
        averages = {}
        data_tmp = data
        udf_tmp = self.train_udf
        person = np.delete(person, movie-1)
        #only add averages of rated items
        s_ratings = 0
        n_ratings = 0
        for item in person:
            if (item != 0):
                s_ratings+=item
                n_ratings+=1

        average = float(s_ratings)/float(n_ratings)
        i = 0
        for row, urow in zip(data_tmp, udf_tmp):
            if row[movie-1] != 0:
                ratings[i] = row[movie-1]
                averages[i] = sum(float(x) for x in row if x!=0)/np.count_nonzero(row)
                urow = np.delete(urow, movie-1)
                s = self.pearson_similarity(person, urow, movie)
                similarities[i] = s #Array of similarity vals
            i+=1

        try:
            rating = self.rate_minus_mean(similarities, ratings, averages)
            if isnan(rating):
                return average
            if rating+average > 5:
                return 5.0
            if rating+average == 0:
                return average
            if ((rating+average) < 0.5): #IS this the best method?
                return 1.0
            return rating+average
        except:
            return average

            #Exceptions when no similar users are found
    def reccomend_user_based_pearson_amp(self, person, movie, data):
        amp = 3.5
        similarities = {}
        ratings = {}
        averages = {}
        data_tmp = data
        person = np.delete(person, movie-1)
        #only add averages of rated items
        s_ratings = 0
        n_ratings = 0
        for item in person:
            if (item != 0):
                s_ratings+=item
                n_ratings+=1

        average = float(s_ratings)/float(n_ratings)
        for i, row in enumerate(data_tmp):
            if row[movie-1] != 0:
                ratings[i] = row[movie-1]
                averages[i] = sum(float(x) for x in row if x!=0)/np.count_nonzero(row)
                row = np.delete(row, movie-1)
                s = self.pearson_similarity(person, row, movie)
                similarities[i] = s #Array of similarity vals

        #Amplitfication step
        for key in similarities:
            similarities[key] = similarities[key]**2.5

        try:
            rating = self.rate_minus_mean(similarities, ratings, averages)
            if isnan(rating):
                return average
            if rating+average > 5:
                return 5.0
            if (rating+average) < 0.5:
                return 1.0
            return rating+average
            #Exceptions when no similar users are found
        except:
            return average

    def reccomend_item_based(self, person, movie, data):
        similarities = {}
        ratings = {}
        averages = []
        data_tmp = data
        movies = []
        averages_user = {}
        item_matrix = np.matrix.transpose(data)
        #only add averages of rated items
        s_ratings = 0
        n_ratings = 0
        for item in person:
            if (item != 0):
                s_ratings+=item
                n_ratings+=1
        average = float(s_ratings)/float(n_ratings)
        #Movie we want to rated
        wanted_movie = item_matrix[movie-1]
        #Get movies person has rated
        for i, movie in enumerate(person):
            if movie != 0:
                movies.append(i+2) #i starts at 0, and person matrix is already movie-1 index
        movies_ind = sorted(movies)
        #Get user averages for adjusted cosine
        for i, row in enumerate(data_tmp):
            averages_user[i] = sum(float(x) for x in row if x!=0)/np.count_nonzero(row)
        j = 0 #movie index
        for i, row in enumerate(item_matrix):
            if (j == len(movies_ind)):
                break;
            if i == (movies_ind[j]-1):
                ratings[i] = person[i-1]
                s = self.adjusted_cosine_similarity(wanted_movie, row, averages_user)
                similarities[i] = s #Array of similarity vals
                j+=1
        averages = [average] * len(similarities)

        try:
            rating = self.rate_minus_mean_adj(similarities, ratings, averages)
            if isnan(rating):
                return average
            if rating+average > 5:
                return 5.0
            if (rating+average) < 0.5:
                return 1.0
            return rating+average
            #Exceptions when no similar users are found
        except:
            return average


    def rate_minus_mean(self, similarities, ratings, averages):
        assert len(similarities) == len(ratings)
        denominator = sum(abs(x) for x in similarities.itervalues())
        numerator = (sum(x*(y-z) for x,y,z in zip(similarities.values(), ratings.values(), averages.values())))
        return numerator/denominator

    def rate_minus_mean_adj(self, similarities, ratings, averages):
        #Values is a array for adjusted cosine
        assert len(similarities) == len(ratings)
        denominator = sum(abs(x) for x in similarities.itervalues())
        numerator = (sum(x*(y-z) for x,y,z in zip(similarities.values(), ratings.values(), averages)))
        return numerator/denominator

    def rate(self, similarities, ratings):
        assert len(similarities) == len(ratings)
        return (1/(sum(similarities.itervalues())))*sum(x*y for x,y in zip(similarities.values(), ratings.values()))


    def predict(self):
        n = 0
        outdata = np.zeros(shape=((sum(len(x) for x in self.testing_movies.values())),3))
        for user, values in self.testing_users.iteritems():
            for movie in self.testing_movies[user]:
                #My implementation
                p_rating = self.reccomend_user_based_pearson(np.array(values), movie, self.training_data)
                c_rating =self.reccomend_user_based_cosine(np.array(values), movie, self.training_data)
                p_rating = (p_rating + c_rating)/2
                outdata[n] = np.array([int(user), int(movie), int(round(p_rating))])
                n+=1
        np.savetxt(self.out_file, outdata, delimiter=' ', fmt='%d')

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Please pass method, test file, and output file as arguments. Methods include cosine and pearson"
    scp, test_file, out_file = sys.argv
    c = collaborative_filtering(test_file, out_file)
    c.fetch_data()
    c.predict()
