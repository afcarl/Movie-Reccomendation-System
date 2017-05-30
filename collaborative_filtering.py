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
            counts[i] = log10(200/j)

        self.train_udf = self.training_data * counts
        print self.train_udf

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
        """
        for person, similarity in zip(similarities.iterkeys(), similarities.itervalues()):
            if similarity < 0.3:
                print similarity
                similarities.pop(person)
                ratings.pop(person)
        print similarities
        print ratings
        """
        try:
            rating = self.rate(similarities, ratings)
            if rating <=0.5:
                rating = 1
            if isnan(rating):
                rating = 3
            if rating > 5:
                rating = 5
            return rating
        #Exceptions when no similar users are found
        except:
            return 3
    def reccomend_user_based_pearson(self, person, movie, data):
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
        try:
            rating = self.rate_minus_mean(similarities, ratings, averages)
            if isnan(rating):
                return average
            if rating+average > 5:
                return 5.0
            if rating+average < 0.5:
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
        try:
            rating = self.rate_minus_mean(similarities, ratings, averages)
            if isnan(rating):
                return average
            if rating+average > 5:
                return 5.0
            if rating+average < 0.5:
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

    def rate(self, similarities, ratings):
        assert len(similarities) == len(ratings)
        return (1/(sum(similarities.itervalues())))*sum(x*y for x,y in zip(similarities.values(), ratings.values()))


    def predict(self):
        n = 0
        outdata = np.zeros(shape=((sum(len(x) for x in self.testing_movies.values())),3))
        for user, values in self.testing_users.iteritems():
            for movie in self.testing_movies[user]:
                #Add easier functionality to automate below this line
                p_rating = self.reccomend_user_based_pearson(np.array(values), movie, self.training_data)
                outdata[n] = np.array([int(user), int(movie), int(round(p_rating))])
                n+=1
        np.savetxt(self.out_file, outdata, delimiter=' ', fmt='%d')

if __name__ == "__main__":
    assert len(sys.argv) == 3, "Please pass method, test file, and output file as arguments. Methods include cosine and pearson"
    scp, test_file, out_file = sys.argv
    c = collaborative_filtering(test_file, out_file)
    c.fetch_data()
    c.predict()
