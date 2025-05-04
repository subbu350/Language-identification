from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from googletrans import Translator


global filename
global X,Y
global accuracy, precision, recall, fscore
global X_train, X_test, y_train, y_test, scaler, dataset, tfidf_vectorizer, rf
global labels

main = tkinter.Tk()
main.title("Language Identification for Multilingual Machine Translation") #designing main screen
main.geometry("1300x1200")
translator = Translator()
 
#fucntion to upload dataset
def uploadDataset():
    global filename, dataset, labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset") #upload dataset file
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename) #read dataset from uploaded file
    dataset = dataset.dropna()
    labels, count = np.unique(dataset['language'], return_counts=True)
    text.insert(END,"Dataset Values\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    
        
def preprocessing():
    text.delete('1.0', END)
    global dataset, scaler
    global X_train, X_test, y_train, y_test, X, Y, tfidf_vectorizer
    #replace missing values with 0
    le = LabelEncoder()
    dataset['language'] = pd.Series(le.fit_transform(dataset['language'].astype(str))) #encoding non-numeric labels into numeric
    dataset = dataset.dropna()
    Y = dataset['language'].ravel()
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), norm='l2', smooth_idf=True, analyzer='char')
    X = tfidf_vectorizer.fit_transform(dataset['text'].ravel()).toarray()
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Text To Numeric Vector\n\n")
    text.insert(END,str(X)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split data into train & test
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train GAN algorithm : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to train GAN algorithm : "+str(X_test.shape[0])+"\n")
    X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1) #split data into train & test

def calculateMetrics(algorithm, predict, y_test):
    global labels
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    

def trainKNN():
    text.delete('1.0', END)
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    accuracy = []
    precision = []
    recall = []
    fscore = []
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("KNN", predict, y_test)

def trainSVM():
    svm_cls = svm.SVC()
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("SVM", predict, y_test)

def trainRF():
    global rf
    if os.path.exists("model/rf.pckl"):
        f = open('model/rf.pckl', 'rb')
        rf = pickle.load(f)
        f.close()
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        f = open('model/rf.pckl', 'wb')
        pickle.dump(rf, f)
        f.close()  
    predict = rf.predict(X_test)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("Random Forest", predict, y_test)

def detectLanguage():
    text.delete('1.0', END)
    global labels, scaler, rf, translator
    input_text = tf1.get()
    tf1.delete(0, END)
    temp = input_text
    temp = tfidf_vectorizer.transform([temp]).toarray()
    temp = scaler.transform(temp)
    predict = rf.predict(temp)[0]
    predict = int(predict)
    detected_lang = labels[predict]
    translation = translator.translate(input_text).text
    text.insert(END,"Input Text = "+input_text+"\n\n")
    text.insert(END,"Detected Language = "+detected_lang+"\n\n")
    text.insert(END,"Translated Text = "+translation)


def graph():
    df = pd.DataFrame([['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','F1 Score',fscore[0]],['KNN','Accuracy',accuracy[0]],
                       ['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','F1 Score',fscore[1]],['SVM','Accuracy',accuracy[1]],
                       ['Random Forest','Precision',precision[2]],['Random Forest','Recall',recall[2]],['Random Forest','F1 Score',fscore[2]],['Random Forest','Accuracy',accuracy[2]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()



font = ('times', 16, 'bold')
title = Label(main, text='Language Identification for Multilingual Machine Translation')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Language Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessing)
processButton.place(x=330,y=550)
processButton.config(font=font1) 

knnButton = Button(main, text="Train KNN Algorithm", command=trainKNN)
knnButton.place(x=570,y=550)
knnButton.config(font=font1)

svmButton = Button(main, text="Train SVM Algorithm", command=trainSVM)
svmButton.place(x=850,y=550)
svmButton.config(font=font1)

rfButton = Button(main, text="Train Random Forest Algorithm", command=trainRF)
rfButton.place(x=50,y=600)
rfButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=330,y=600)
graphButton.config(font=font1)

l1 = Label(main, text='Input Text:')
l1.config(font=font)
l1.place(x=50,y=650)

tf1 = Entry(main,width=70)
tf1.config(font=font)
tf1.place(x=160,y=650)

detectButton = Button(main, text="Language Detection & Translation", command=detectLanguage)
detectButton.place(x=970,y=650)
detectButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
