import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

def get_feature_name(file):
    features = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            features.append(line.strip().split(','))
    return features

def get_normalized(data):
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_features_scaled = pd.DataFrame(x_scaled, columns=data.columns)
    return df_features_scaled
    
def get_imputed(data):
    fill_NaN = SimpleImputer(missing_values=np.nan, strategy='median')
    imputed_DF = pd.DataFrame(fill_NaN.fit_transform(data))
    return imputed_DF

def get_clustered(data):
    kmeans = KMeans(n_clusters=5, 
                    init='k-means++', 
                    max_iter=10000, 
                    n_init=10, 
                    random_state=42)
    cluster_label = kmeans.fit_predict(data)
    
    return cluster_label

# reading input feature values and assesment score file
data = pd.read_csv('input\\MSCOPE-DataAnalytics1_DATA_Updated_post deployment.csv')
assesment= pd.read_excel('input\\Assesment.xlsx', 
                         index_col=None, 
                        sheet_name='All event')  

# assigning a uniqe ID with each row
data['combined_id']=data["record_id"].astype(str)+"_"+data["redcap_event_name"]
assesment['combined_id']=assesment["record_id"].astype(str)+"_"+assesment["redcap_event_name"]

# re-arrange column order to make the COMBINED_ID first column 
data = data[ [data.columns.tolist()[-1]] + data.columns.tolist()[:-1] ]
assesment = assesment[ [assesment.columns.tolist()[-1]] + assesment.columns.tolist()[:-1] ]

data = data.set_index('combined_id')
assesment = assesment.set_index('combined_id')

assesment_features=['overall_prepar_recent',
                    'physperf_prepar_recent',
                    'mentalperf_prepar_recent',
                    'tactical_prepar_recent',
                    'physhealth_prepar_recent',
                    'menthealth_prepar_recent']

# Loding interested feature names by category
features = get_feature_name('input\\features.txt')
feature_category= [f[0] for f in features]
print('Feature category :: ',feature_category)   

for feature in features:
    categor=feature[0]
    data_c= pd.merge(data[feature[1:]].copy(), 
                     assesment[assesment_features].copy(), 
                     on = 'combined_id', how='right')
    
    normalize_data = get_normalized(data_c[ feature[1:] ]) 
    imputed_data = get_imputed(normalize_data)
    cluster_label = get_clustered(imputed_data)
    #clustered_data = data_with_assesment[feature[1:]+assesment_features]
    data_c['cluster'] = cluster_label
    data_c.to_csv('output\\'+str(categor)+'.csv')
    
