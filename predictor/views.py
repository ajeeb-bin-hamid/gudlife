from django.shortcuts import render
from urllib import request
from django.http import HttpResponse, HttpResponseRedirect
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

# Create your views here.
 
def index(request):
    return render(request, "index.html")

def predict(request):
    if request.method == 'POST':
        age = request.POST['Age']
        gender = request.POST['Gender']
        fam = request.POST['Family_history']
        rem = request.POST['Remote_eork']
        seak = request.POST['Seak_help']
        anon = request.POST['anonymity']
        phy = request.POST['phys_health_consequence']
        obs = request.POST['obs_consequence']
        
        int_features=[]
        #int_features.append(int(age))
        int_features.append(int(gender))
        int_features.append(int(fam))
        int_features.append(int(rem))
        int_features.append(int(seak))
        int_features.append(int(anon))
        int_features.append(int(phy))
        int_features.append(int(obs))
        #int_features=[int(x) for x in request.POST.values()]
        final=[np.array(int_features)]
        print(int_features)
        print(final)
        prediction=model.predict_proba(final)
        output='{0:.{1}f}'.format(prediction[0][1], 2)

        if output>str(0.5):
            return render(request, 'index.html',{"pred":'You need a treatment.\nProbability of mental illness is {}'.format(output)})
        else:
            return render(request,'index.html',{"pred":'You do not need treatment.\n Probability of mental illness is {}'.format(output)})
    