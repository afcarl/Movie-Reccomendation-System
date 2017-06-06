import numpy as np
from math import *
import sys

class collaborative_filtering(object):
    def __init__(self, method, test_file, out_file):
        self.training_data = np.array([])
        self.train_udf = np.array([])
        self.test_file = test_file
        self.out_file = out_file
        self.testing_users = {}
        self.testing_movies = {}

        #Run the correct method
        if method == "pearson":
            self.mthd = self.reccomend_user_based_pearson
        elif method == "cosine":
            self.mthd = self.reccomend_user_based_cosine
        elif method == "iuf":
            self.mthd = self.reccomend_user_based_pearson_iuf
        elif method == "amplification":
            self.mthd = self.reccomend_user_based_pearson_amp
        elif method == "item":
            self.mthd = self.reccomend_item_based
        elif method == "personal":
            pass
        else:
            print "Incorrect method, Methods include:\npearson\ncosine\niuf\namplification\nitem\npersonal"
            sys.exit()

    def fetch_data(self):
        """
        Grab all the data and organize it properly
        """
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
        """
        Takes two arrays, returns their cosine similarity
        """
        assert person_x.size == person_y.size
        numerator =  sum(x*y for x,y in zip(person_x, person_y))
        denominator = (sqrt(sum(x*x for x in person_x))) * (sqrt(sum(y*y for y in person_y)))
        return numerator/denominator

    def pearson_similarity(self, person_x, person_y, movie):
        """
        Takes two arrays, returns their Pearson correlation
        """
        assert person_x.size == person_y.size
        #Grab averages
        pxm = sum(float(x) for x in person_x if x != 0)/np.count_nonzero(person_x)
        pym = sum(float(y) for y in person_y if y != 0)/np.count_nonzero(person_y)
        numerator = sum(((float(x)-pxm)*(float(y)-pym) for x,y in zip(person_x, person_y)))
        denominator = (sqrt(sum(((float(x)-pxm))**2 for x in person_x))) * (sqrt(sum((float(y)-pym)**2 for y in person_y)))
        return float(numerator)/float(denominator)

    def adjusted_cosine_similarity(self, wanted_movie, movie, averages):
        """
        Takes two array, returns their Pearson correlation. Used for Item based
        Collaborative Filtering
        """
        assert wanted_movie.size == movie.size
        wanted_movie_mm = np.zeros(shape=(200))
        movie_mm = np.zeros(shape=(200))
        #Subract average user rating from the movies for adjusted part
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

    def rate_minus_mean(self, similarities, ratings, averages):
        """
        Used to return final rating deviation value for user. Inputs are
        a dictionary of Pearson similarities, ratings, and average values for user's as keys
        """
        assert len(similarities) == len(ratings)
        denominator = sum(abs(x) for x in similarities.itervalues())
        numerator = (sum(x*(y-z) for x,y,z in zip(similarities.values(), ratings.values(), averages.values())))
        return numerator/denominator

    def rate_minus_mean_adj(self, similarities, ratings, averages):
        """
        Used to return final rating deviation for adjusted cosine similarity. Inputs are
        a dictionary of similarities and ratings, and a list of average ratings
        """
        assert len(similarities) == len(ratings)
        denominator = sum(abs(x) for x in similarities.itervalues())
        #averages is array instead of dict for adjusted cosine
        numerator = (sum(x*(y-z) for x,y,z in zip(similarities.values(), ratings.values(), averages)))
        return numerator/denominator

    def rate(self, similarities, ratings):
        """
        Returns weighted average predicted rating when using cosine similarity
        Takes dictionary of similarities and ratings as Inputs
        """
        assert len(similarities) == len(ratings)
        return (1/(sum(similarities.itervalues())))*sum(x*y for x,y in zip(similarities.values(), ratings.values()))

    def reccomend_user_based_cosine(self, person, movie, data):
        """
        Takes the test person, movie to predict, and training data as arguments
        Returns persons predicted rating based on cosine similarity
        """
        similarities = {}
        ratings = {}
        #Get similarity scores for users and their ratings
        for i, row in enumerate(data):
            if row[int(movie)-1] != 0:
                s = self.cosine_similarity(person, row)
                if (s!=0):
                    similarities[i] = s #Array of similarity vals
                    ratings[i] = row[int(movie)-1]   #array of these peoples ratings
        #divide by total similaries, each person score minus persons mean
        s_ratings = 0
        n_ratings = 0
        for item in person:
            if (item != 0):
                s_ratings+=item
                n_ratings+=1

        average = float(s_ratings)/float(n_ratings)
        #K furthest neighbors
        if len(similarities) < 8: #Keeps getting better with ensemble, try 6, last was 5 @0.74335 10 @ 0.741955 15 @0.74466 ->@ 8 is max
            return average

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
        """
        Takes the test person, movie to predict, and training data as arguments
        Returns persons predicted rating based on pearson correlation
        """
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

        #Increases predictive power slightly
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

    def reccomend_user_based_pearson_iuf(self, person, movie, data):
        """
        Takes the test person, movie to predict, and training data as arguments
        Returns persons predicted rating based on pearson correlation from a IUF matrix
        """
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
            if ((rating+average) < 0.5):
                return 1.0
            return rating+average
        except:
            return average

    def reccomend_user_based_pearson_amp(self, person, movie, data):
        """
        Takes the test person, movie to predict, and training data as arguments
        Returns persons predicted rating using pearson correlation and case amplification
        """
        amp = 2.5
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
        """
        Takes the test person, movie to predict, and training data as arguments
        Returns persons predicted rating using item based collaborative filtering
        """
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
                break; #Only rated so many movies
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

    def predict(self):
        """
        Predicts movie ratings using the method passed as command line argument
        Outputs a file containing all the ratings
        """
        n = 0
        outdata = np.zeros(shape=((sum(len(x) for x in self.testing_movies.values())),3))
        #Get test ratings
        for user, values in self.testing_users.iteritems():
            for movie in self.testing_movies[user]:
                #Compare test data with all the training data
                p_rating = self.mthd(np.array(values), movie, self.training_data)
                outdata[n] = np.array([int(user), int(movie), int(round(p_rating))])
                n+=1
        #Output in correct format
        np.savetxt(self.out_file, outdata, delimiter=' ', fmt='%d')

    def predict_ensemble(self):
        """
        Predicts movie ratings using a weighted ensemble method.
        Combines cosine similarity, pearson correlation, iuf, and item based collaborative Filtering
        Outputs a text file
        """
        n = 0
        outdata = np.zeros(shape=((sum(len(x) for x in self.testing_movies.values())),3))
        for user, values in self.testing_users.iteritems():
            for movie in self.testing_movies[user]:
                #My implementation, just use all of them
                p_rating = self.reccomend_user_based_pearson(np.array(values), movie, self.training_data)
                c_rating =self.reccomend_user_based_cosine(np.array(values), movie, self.training_data)
                u_rating = self.reccomend_user_based_pearson_iuf(np.array(values), movie, self.training_data)
                i_rating = self.reccomend_item_based(np.array(values), movie, self.training_data) #
                #Ensemble method with weighted averages
                p_rating = ((1-0.760958791659826)*p_rating+(1-0.781686094237399)*c_rating+(1-0.761328189131505)*u_rating+(1-0.854662616975866)*i_rating)/((1-0.781686094237399)+(1-0.760958791659826)+(1-0.761328189131505)+(1-0.774380245008199)+(1-0.854662616975866))
                outdata[n] = np.array([int(user), int(movie), int(round(p_rating))])
                n+=1
        np.savetxt(self.out_file, outdata, delimiter=' ', fmt='%d')

if __name__ == "__main__":
    assert len(sys.argv) == 4, "Please pass method, test file, and output file as arguments."
    scp, method, test_file, out_file = sys.argv

    c = collaborative_filtering(method, test_file, out_file)
    c.fetch_data()
    if method != "personal":
        c.predict()
    else:
        c.predict_ensemble()
