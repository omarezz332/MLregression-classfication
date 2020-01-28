import pandas as pd 
import numpy as np 
import json
import operator
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data_credits = pd.read_csv('tmdb_5000_credits_train.csv')
data_movies  = pd.read_csv('tmdb_5000_movies_train.csv')

data_movies.isna().sum()

data_movies = data_movies.dropna(how='all')

combined_data=pd.merge(data_movies,data_credits[['movie_id','cast','crew']],left_on='id',right_on='movie_id')


combined_data.to_csv('CombinedData.csv')

combined_data = pd.read_csv('CombinedData.csv')

combined_data=combined_data.drop(['homepage','id','original_language','original_title','overview','status','tagline','title','movie_id'],axis=1)

combined_data.corr()['vote_average']

def add_new_col(name , val):
    df = pd.DataFrame(combined_data)
    df.insert(6,name,val,True)
    df.to_csv('CombinedData.csv')

def transform_dict_to_n_columns(combined_data, col, focus):
    d = dict()
    c = dict()
    i = 0
    List = []
    for item in combined_data[col]:
        temp = json.loads(item)
        for it in temp:
            List.append(it[focus])
            lastv = it[focus]

    repeted = Counter(List)
    repeted = repeted.most_common(200)
    comul = 0
    v_n_col = []
    for item in combined_data[col]:
        temp = json.loads(item)
        comul=0
        for it in temp:
            for i in repeted:
                if it[focus] == i[0]:
                    comul += 1
        v_n_col.append(comul)
    colname = "new " + str(col)
    add_new_col(colname, v_n_col)

    return repeted

def transform_dict(combined_data,col,focus):
  d = dict()
  c = dict()
  i=0
  for item in combined_data[col]:
      temp = json.loads(item)
      for it in temp :
        if it[focus] not in d :
          d[it[focus]]=combined_data.iloc[i,combined_data.shape[1]-4]
          c[it[focus]]=1

        else:
          d[it[focus]]+=(combined_data.iloc[i,combined_data.shape[1]-4])
          c[it[focus]]+=1
      i+=1


  final=dict()
  for item in d :
      final[item]=d[item]/c[item]


  output=[]
  for item in combined_data[col]:
    tempo = json.loads(item)
    adder=0
    for it in tempo :
       adder+=final[it[focus]]
    output.append(adder)
  output = np.reshape(output,(combined_data.shape[0],))

  return output


def transform_re_date(combined_data, col):
    output = []
    for item in combined_data[col]:
        x = item.split("/")
        weight = float(2019) - float(x[2])
        output.append(weight)
    add_new_col("new release_date" , output)



def transform_crew(combined_data, col, focus):
    d = dict()
    c = dict()
    i = 0
    for item in combined_data[col]:
        temp = json.loads(item)
        for it in temp:
            if it[focus] not in d and (it['department'] == 'Production' or it['department'] == 'Directing'):

                d[it[focus]] = combined_data.iloc[i, combined_data.shape[1] - 4]
                c[it[focus]] = 1
            elif it[focus] in d and (it['department'] == 'Production' or it['department'] == 'Directing'):
                d[it[focus]] += (combined_data.iloc[i, combined_data.shape[1] - 4])
                c[it[focus]] += 1
        i += 1
    final = dict()
    for item in d:
        final[item] = d[item] / c[item]
    output = []
    for item in combined_data[col]:
        tempo = json.loads(item)
        adder = 0
        for it in tempo:
            if (it['department'] == 'Production' or it['department'] == 'Directing'):
                adder += final[it[focus]]
        output.append(adder)
    output = np.reshape(output, (combined_data.shape[0],))
    return output

#start
def preprocessing_call(combined_data):

    keywords_comul = transform_dict_to_n_columns(combined_data,'keywords','name')

    genres_comul = transform_dict(combined_data,'genres','name')
    colname = "new geners"
    add_new_col(colname, genres_comul)

    crew_comul = transform_crew(combined_data,'crew','name')
    colname = "new crew"
    add_new_col(colname, crew_comul)

    production_countries_comul = transform_dict_to_n_columns(combined_data,'production_countries','name')

    production_companies_comul = transform_dict_to_n_columns(combined_data,'production_companies','name')

    cast_comul = transform_dict_to_n_columns(combined_data,'cast','name')

    spoken_languages_comul = transform_dict_to_n_columns(combined_data,'spoken_languages','name')

    transform_re_date(combined_data,'release_date')

    combined_data=combined_data.drop(['production_countries','release_date','production_companies','genres','spoken_languages','keywords','cast','crew'],axis=1)

    combined_data=combined_data.dropna()

    combined_data.to_csv('CombinedData.csv')

    print(combined_data.corr()['vote_average'])

    combined_data.to_csv('CombinedData.csv')

    preprocessed_data = combined_data.loc[:, combined_data.columns != 'vote_average']
    labels = combined_data['vote_average']

    return preprocessed_data,labels

preprocessed_data,labels = preprocessing_call(combined_data)
X_train,X_test,y_train,y_test=train_test_split(preprocessed_data,labels,test_size=0.2,random_state=40)

clf = LinearRegression().fit(X_train,y_train)
y_hat = clf.predict(X_test)
print("mse linear is : ",mean_squared_error(y_test,y_hat))


cls = linear_model.LinearRegression()
X=np.expand_dims(preprocessed_data['vote_count'], axis=1)
Y=np.expand_dims(labels, axis=1)
cls.fit(X,Y)
prediction= cls.predict(X)
plt.scatter(X, Y)
plt.xlabel('duration', fontsize = 20)
plt.ylabel('interval', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()


#####################################


poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print('Mean Square Error for poly reg d = 2 is : ', metrics.mean_squared_error(y_test, prediction))





#############################

#end


'''
genres =transform_dict(combined_data,'genres','name')
production_companies = transform_dict(combined_data,'production_companies','name')
cast = transform_dict(combined_data,'cast','name')
crew = transform_crew(combined_data,'crew','name')


g = dict()
g['budget']=combined_data['budget']
g['genres']=genres
g['popularity']=combined_data['popularity']
g['production_companies']=production_companies
g['revenue']=combined_data['revenue']
g['runtime']=combined_data['runtime']
g['cast']=cast
g['crew']=crew
g['vote_count']=combined_data['vote_count']
g['vote weight']=combined_data['vote_count']*combined_data['vote_average']
g['vote_average']=combined_data['vote_average']
final_data = pd.DataFrame(g)
print(final_data)

final_data=final_data.dropna()

final_data.to_csv('FinalData.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = final_data[['budget','genres','popularity','production_companies','revenue','runtime','cast','crew','vote_count','vote weight']]
labels = final_data['vote_average']
data = np.array(data)
labels=np.array(labels)
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=42)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error



poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.fit_transform(X_test)
clf_poly = poly_features.fit(X_train_poly,y_train)
prd = clf_poly.predict(X_test_poly)
print("mse poly 2 is : ",mean_squared_error(y_test,prd))



clf = LinearRegression().fit(X_train,y_train)
y_hat = clf.predict(X_test)
print("mse linear is : ",mean_squared_error(y_test,y_hat))

'''
