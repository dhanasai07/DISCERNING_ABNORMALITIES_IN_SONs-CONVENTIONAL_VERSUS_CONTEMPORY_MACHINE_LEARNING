from django.shortcuts import render,redirect
from django.contrib.auth.models import User 
from django.contrib import messages

# Create your views here.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering  

def index(request):

    return render(request,'index.html')


def about(request):

    return render(request,'about.html')

def login(request):
    if request.method=='POST':
        lemail=request.POST['email']
        lpassword=request.POST['password']

        d=User.objects.filter(email=lemail,password=lpassword).exists()
        print(d)
        return redirect('userhome')
    else:
        return render(request,'login.html')

   

def registration(request):
    if request.method=='POST':
        Name = request.POST['Name']
        email=request.POST['email']
        password=request.POST['password']
        conpassword=request.POST['conpassword']
        age=request.POST['Age']
        contact=request.POST['contact']

        print(Name,email,password,conpassword,age,contact)
        if password==conpassword:
            user=User(email=email,password=password)
            # user.save()
            return render(request,'login.html')
        else:
            msg='Register failed!!'
            return render(request,'registration.html')

    return render(request,'registration.html')

def userhome(request):
    
    return render(request,'userhome.html')

def load(request):
   if request.method=="POST":
        file=request.FILES['file']
        global df
        df=pd.read_csv('ML-MATT-CompetitionQT1920_train.csv',encoding= 'unicode_escape')
        messages.info(request,"Data Uploaded Successfully")
    
   return render(request,'load.html')



def view(request):
    global data
    
   
    # dummy=df.head(100)
    data=pd.read_csv('ML-MATT-CompetitionQT1920_train.csv',encoding= 'unicode_escape')
    print(data)
    col=data.to_html
    print(col)
   
    # col=dummy.columns
    # rows=dummy.values.tolist()
    # return render(request, 'view.html',{'col':col,'rows':rows})

    return render(request,'view.html', {'columns':df.columns.values, 'rows':df.values.tolist()})



def model(request):
    global x_train,x_test,y_train,y_test
    if request.method == "POST":

        df.maxUE_DL.fillna(value=df.maxUE_DL.mode()[0],inplace=True)

        df.maxUE_UL.fillna(value=df.maxUE_UL.mode()[0],inplace=True)

        df['maxUE_UL+DL'].fillna(value=df['maxUE_UL+DL'].mode()[0],inplace=True)

        df.drop(['Time'],axis=True,inplace=True)

        df.drop(['CellName'],axis=True,inplace=True)

        df.drop(['maxUE_UL+DL'],axis=True,inplace=True)

        x = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        
        oversample = SMOTE()
        X_sm, y_sm = oversample.fit_resample(x, y)

        x_train,x_test,y_train,y_test = train_test_split(X_sm, y_sm,test_size=0.3,random_state=42)
        model = request.POST['algo']

        if model == "0":
            
            cb = RandomForestClassifier()
            cb.fit(x_train,y_train)
            cbpred = cb.predict(x_test)
            cba = accuracy_score(y_test,cbpred)*100
            # cbaa = 87.23312579442877
            msg = 'Accuracy of RandomForestClassifier : ' + str(cba)
            return render(request,'model.html',{'msg':msg})
        elif model == "1":
            lr = SVC(kernel='linear')
            lr.fit(x_train[:1000],y_train[:1000])
            lrp = lr.predict(x_test)
            lra = accuracy_score(y_test,lrp)*100
            msg = 'Accuracy of SVM :  ' + str(lra)
           
            return render(request,'model.html',{'msg':msg})
        elif model == "2":
            knn = KNeighborsClassifier()
            knn.fit(x_train,y_train)
            knnp = knn.predict(x_test)
            knna = accuracy_score(y_test,knnp)*100
            
            msg = 'Accuracy of KNeighborsClassifier :  ' + str(knna)
           
            return render(request,'model.html',{'msg':msg})
        
        elif model == "3":
            km_clustering = KMeans(n_clusters=3)
            km_clustering.fit(x_train)
            filename = 'finalized_model.sav'
            pickle.dump(km_clustering, open(filename, 'wb'))
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.score(x_test, y_test)
            
            msg = 'Accuracy of KMeans :  ' + str(result)
           
            return render(request,'model.html',{'msg':msg})
        elif model == "4":
            model = Sequential()
            model.add(Dense(30, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, batch_size=500, epochs=50,validation_data=(x_test, y_test))
            abc=model.predict(x_test)
            annacc =accuracy_score(abc,y_test)
                        
            msg = 'Accuracy of ANN :  ' + str(annacc)
           
            return render(request,'model.html',{'msg':msg})

              
    return render(request,'model.html')


def prediction(request):

    global x_train,x_test,y_train,y_test,x,y
    

    if request.method == 'POST':

        PRBUsageUL = float(request.POST['PRBUsageUL'])
        PRBUsageDL = float(request.POST['PRBUsageDL'])
        meanThr_DL = float(request.POST['meanThr_DL'])
        meanThr_UL = float(request.POST['meanThr_UL'])
        maxThr_DL = float(request.POST['maxThr_DL'])
        maxThr_UL = float(request.POST['maxThr_UL'])
        meanUE_DL = float(request.POST['meanUE_DL'])
        meanUE_UL = float(request.POST['meanUE_UL'])
        maxUE_DL  = float(request.POST['maxUE_DL'])
        maxUE_UL  = float(request.POST['maxUE_UL'])
       
        


        PRED = [[PRBUsageUL,PRBUsageDL,meanThr_DL,meanThr_UL,maxThr_DL,maxThr_UL,meanUE_DL,meanUE_UL,maxUE_DL,maxUE_UL]]
       
        knn = SVC()
        knn.fit(x_train,y_train)
        xgp = np.array(knn.predict(PRED))

        if xgp==0:
           
            msg = ' <span style = color:white;>This prediction result is : <span style = color:green;><b>No Anomaly</b></span></span>'
        elif xgp==1:
            msg = ' <span style = color:white;>This prediction result is : <span style = color:red;><b>Anomaly</b></span></span>'
        
        return render(request,'prediction.html',{'msg':msg})

    
    return render(request,'prediction.html')

