#MLP on Wine Quality with different architectures [13]
from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_squared_error,r2_score

X,y=load_wine(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=0)
sc=StandardScaler().fit(Xtr)
Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)

for h in [(5,),(50,),(50,30),(100,50,25)]:
    m=MLPClassifier(hidden_layer_sizes=h,max_iter=1000,random_state=0).fit(Xtr,ytr)
    print(f"Layers {str(h):<16} | Accuracy: {accuracy_score(yte,m.predict(Xte)):.4f}")
