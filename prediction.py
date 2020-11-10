import pandas as pd
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import yellowbrick as yb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.features.rankd import Rank1D, Rank2D 
from yellowbrick.features.radviz import RadViz 
from yellowbrick.features.pcoords import ParallelCoordinates 
from yellowbrick.features.jointplot import JointPlotVisualizer
from yellowbrick.features.pca import PCADecomposition
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import Ridge, Lasso 
from sklearn.model_selection import train_test_split
from yellowbrick.regressor import PredictionError, ResidualsPlot
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
from sklearn.cluster import KMeans

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

data = pd.read_csv("Datasets/KDDTrain+.txt", header = None, names = col_names)

X = data[col_names].to_numpy()
y = data.label.to_numpy()
df=data
df.protocol_type = preprocessing.LabelEncoder().fit_transform(df["protocol_type"])
df.service = preprocessing.LabelEncoder().fit_transform(df["service"])
df.flag = preprocessing.LabelEncoder().fit_transform(df["flag"])
df.label = preprocessing.LabelEncoder().fit_transform(df["label"])
X = df[col_names].to_numpy()
y = df.label.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
labels = data['label'].copy()
labels[labels!='normal'] = 'anomaly'
test_data = pd.read_csv("Datasets/KDDTrain+.txt", header=None, names = col_names)
k = 60
km = KMeans(n_clusters = k)

t0 = time()
km.fit(features)
tt = time()-t0
print("Clustered in {} seconds".format(round(tt,3)))


label_names = list(map(
    lambda x: pd.Series([labels[i] for i in range(len(km.labels_)) if km.labels_[i]==x]), 
    range(k)))



clusters = []
for i in range(len(label_names)):
    clusters.append(label_names[i].value_counts().index.tolist()[0])

def predict(s):
	df= pd.DataFrame.from_dict(s)
	t0 = time()
	pred = km.predict(corrected)
	tt = time() - t0

	new_labels = [] 
	for i in pred:                   
    	new_labels.append(clusters[i])
    return pred[0]