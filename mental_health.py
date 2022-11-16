import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("mental_health.csv")
male_str = ["male", "m", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in data.iterrows():

    if str.lower(col.Gender) in male_str:
        data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

#stk_list = ['A little about you', 'p']
#data = data[~train_df['Gender'].isin(stk_list)]
data['Gender']=data['Gender'].map({'male':0,'female':1, 'trans':2})
data['family_history']=data['family_history'].map({'No':0,'Yes':1})
data['treatment']=data['treatment'].map({'No':0,'Yes':1})
data['remote_work']=data['remote_work'].map({'No':0,'Yes':1})
data['seek_help']=data['seek_help'].map({'No':0,'Yes':1, 'Don\'t know':2})
data['anonymity']=data['anonymity'].map({'No':0,'Yes':1, 'Don\'t know':2})
data['phys_health_consequence']=data['phys_health_consequence'].map({'No':0,'Yes':1, 'Maybe':2})
data['obs_consequence']=data['obs_consequence'].map({'No':0,'Yes':1})

data = np.array(data)

#print(data)
X = data[1:,1:-1]
y = data[1:, 4]
y = y.astype('int')
X = X.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
stack = StackingClassifier(classifiers=[clf1,clf2, clf3], meta_classifier=lr)
stack.fit(X_train, y_train)

pickle.dump(stack,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))