import pandas as pd 
import numpy as np 
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data_credits = pd.read_csv('tmdb_5000_credits.csv')
data_movies  = pd.read_csv('tmdb_5000_movies_classification.csv')

data_movies.isna().sum()

data_movies = data_movies.dropna(how='all')

combined_data=pd.merge(data_movies,data_credits[['movie_id','cast','crew']],left_on='id',right_on='movie_id')


combined_data.to_csv('CombinedData_classification.csv')

combined_data = pd.read_csv('CombinedData_classification.csv')

combined_data=combined_data.drop(['homepage','id','original_language','original_title','overview','status','tagline','title','movie_id'],axis=1)


def rate_encoding():
    # 1 for High
    # 2 for Intermediate
    # 0 for else
    enc=[]
    for item in combined_data['rate']:
        if item == 'High':
            enc.append(2)
        elif item == 'Intermediate':
            enc.append(1)
        else:
            enc.append(0)
    add_new_col('encoded_rate',enc)

def add_new_col(name , val):
    df = pd.DataFrame(combined_data)
    df.insert(6,name,val,True)
    df.to_csv('CombinedData_classification.csv')

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
        if math.isnan(float(item)):
            output.append(0)
        else:
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

    genres_comul = transform_dict_to_n_columns(combined_data,'genres','name')



    crew_comul = transform_crew(combined_data,'crew','name')
    colname = "new crew"
    add_new_col(colname, crew_comul)

    production_countries_comul = transform_dict_to_n_columns(combined_data,'production_countries','name')

    production_companies_comul = transform_dict_to_n_columns(combined_data,'production_companies','name')

    cast_comul = transform_dict_to_n_columns(combined_data,'cast','name')

    spoken_languages_comul = transform_dict_to_n_columns(combined_data,'spoken_languages','name')

    #transform_re_date(combined_data,'release_date')

    rate_encoding()

    combined_data=combined_data.drop(['production_countries','release_date','rate','production_companies','genres','spoken_languages','keywords','cast','crew'],axis=1)

    combined_data=combined_data.dropna()

    combined_data.to_csv('CombinedData_classification.csv')

    print(combined_data.corr()['encoded_rate'])

    combined_data.to_csv('CombinedData_classification.csv')

    preprocessed_data = combined_data.loc[:, combined_data.columns != 'encoded_rate']
    labels = combined_data['encoded_rate']

    return preprocessed_data , labels

preprocessed_data , labels = preprocessing_call(combined_data)

X_train,X_test,y_train,y_test=train_test_split(preprocessed_data,labels,test_size=0.2,random_state=40)

print(np.shape(X_train))
#end
