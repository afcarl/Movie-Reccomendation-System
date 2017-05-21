import numpy as np
import pandas as pd
from math import *
import sys

class collaborative_filtering(object):
    def __init__(self, mthd, test_file, out_file):
        self.training_data = np.array([])
        self.test_file = test_file
        self.out_file = out_file
        self.testing_users = {}
        self.testing_movies = {}

        if mthd == "cosine":
            self.mthd = self.cosine_similarity
        else if mthd == "pearson":
            self.mthd = self.pearson_similarity
        else:
            print "method options include cosine or pearson"

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

    def cosine_similarity(self, person_x, person_y):
        assert person_x.size == person_y.size

        numerator =  sum(x*y for x,y in zip(person_x, person_y))
        denominator = (sqrt(sum(x*x for x in person_x))) * (sqrt(sum(y*y for y in person_y)))
        return numerator/denominator

    def pearson_similarity(self, person_x, person_y):
        assert person_x.size == person_y.size

        pxm = np.mean(person_x)
        pym = np.mean(person_y)

        numerator = sum(((float(x)-pxm)*(float(y)-pym) for x,y in zip(person_x, person_y)))
        denominator = (sqrt(sum((float(x)-pxm)**2 for x in person_x))) * (sqrt(sum((float(y)-pym)**2 for y in person_y)))
        return float(numerator/denominator)


    def reccomend(self,f, person, movie, data):
        similarities = {}
        ratings = {}
        #Get user averages
        averages = np.mean(data, axis=0)
        #Get similarity scores for users and their ratings
        for i, row in enumerate(data):
            if row[int(movie)-1] != 0:
                similarities[i] = f(person, row) #Array of similarity vals
                ratings[i] = row[int(movie)-1]   #array of these peoples ratings

        #sorted_users = sorted(similarities.iterkeys(), key=lambda k:similarities[k], reverse=True)
        try:
            rating = (1/(sum(similarities.itervalues())))*sum(x*(y-z) for x,y,z in zip(similarities.values(), ratings.values(), averages))
            return rating
        except:
            return 2.5

    def predict(self):
        n = 0
        outdata = np.zeros(shape=((sum(len(x) for x in self.testing_movies.values())),3))
        for user, values in self.testing_users.iteritems():
            for movie in self.testing_movies[user]:
                p_rating = self.reccomend(self.mthd, np.array(values), movie, self.training_data)
                outdata[n] = np.array([user, movie, p_rating])
                n+=1
        np.savetxt(self.out_file, outdata, delimiter=' ', fmt='%.3f')

    def my_own_implementation(self):
        pass
        #normalize data


if __name__ == "__main__":
    assert len(sys.argv) == 4, "Please pass method, test file, and output file as arguments. Methods include cosine and pearson""
    scp, mthd, test_file, out_file = sys.argv
    c = collaborative_filtering(mthd, test_file, out_file)
    c.fetch_data()
    c.predict()
