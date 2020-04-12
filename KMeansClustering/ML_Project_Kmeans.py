import numpy as np     # linear algebra
import pandas as pd    # data processing, CSV file I/O
import matplotlib.pyplot as plt
    
def graphic(x,y): #for elbow graphic
    plt.plot(x, y)
    plt.ylabel('within cluster sum of squared errors')
    plt.xlabel('number of clusters (k)')
    plt.title('WSS vs. K Graph\nThe Elbow Method')
    plt.legend()
    plt.show()

def scatt(k, data, centroids, clusters): #for scattering all samples
    color=['red','blue','green','cyan','magenta']
    labels=['cluster1','cluster2','cluster3','cluster4','cluster5']
    for i in range(k):
        clus = data[clusters == i]
        plt.scatter(clus.iloc[:,0],clus.iloc[:,1],c=color[i],label=labels[i])
    plt.scatter(centroids[:,0],centroids[:,1],s=200,c='yellow',label='Centroids')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.legend()
    plt.show()

def pre_scatt(train_all,centers_new, clusters,k): #columns are determined
    for sc in range (len(train_all.iloc[0])): #sc represents column number
        if sc == 0: 
            scatt(k,train_all.iloc[:,[sc,sc+1]], centers_new[:,[sc,sc+1]], clusters)
            scatt(k,train_all.iloc[:,[sc,sc+2]], centers_new[:,[sc,sc+2]], clusters)
            #scatt(k,train_all.iloc[:,[sc,sc+3]], centers_new[:,[sc,sc+3]], clusters) #use when have more than 3 features
        elif sc == 1:
            scatt(k,train_all.iloc[:,[sc,sc+1]], centers_new[:,[sc,sc+1]], clusters)
            #scatt(k,train_all.iloc[:,[sc,sc+2]], centers_new[:,[sc,sc+2]], clusters) #use when have more than 3 features
        #elif sc == 2:                                                                #use when have more than 3 features
            #scatt(k,train_all.iloc[:,[sc,sc+1]], centers_new[:,[sc,sc+1]], clusters) #use when have more than 3 features

def wss(k,c,n): 
    centers = np.zeros((k,c))

    for knn in range(k):
        centers[knn] = train_all.loc[knn,:].values #inital centers
        #centers = np.random.randn(k,c) + mean.values.reshape(1,3) #random initial centers (includes nan, error occured)

    centers_old = np.zeros(centers.shape) 
    centers_new = centers.copy() 
    clusters = np.zeros(n)
    distances = np.zeros((n,k))
    
    stopper = np.linalg.norm(centers_new - centers_old)

    while stopper > 0.001:
        for i in range(k):
            #distance of each example to centers
            distances[:,i] = np.linalg.norm(train_all - centers_new[i], axis=1) #L1 distances 
            
        clusters = np.argmin(distances, axis = 1) #determined clusters
        centers_old = centers_new.copy() #old centers stored
        
        for t in range(k):
            #each cluster's mean calculated
            centers_new[t] = np.mean(train_all[clusters == t], axis=0)

        stopper = np.linalg.norm(centers_new - centers_old)
        #stop if centers not changed 

    err = 0
    for p in range (len(distances)):
        err += distances[p,clusters[p]]
    y_wss.append(err)
    return y_wss,distances,clusters,centers_new

#data    
train_all = pd.read_csv("../proje/Mall_Customers_norm.csv")
train_all = train_all.reset_index(drop=True)
train_all = train_all.drop(labels = ["CustomerID"], axis = 1) 
train_all['Gender'].replace(['Female','Male'], [0,1],inplace=True)
train_all = train_all.drop(labels = ["Gender"], axis = 1) 

#variables
n = train_all.shape[0]
c = train_all.shape[1]
clusters = np.zeros(len(train_all))
mean = np.mean(train_all, axis = 0)
std = np.std(train_all, axis = 0)
x_k = []
y_wss = []

for k in range(2,11):
    
    x_k.append(k)
    y_wss,distances,clusters,centers_new = wss(k,c,n)
graphic(x_k,y_wss)

k = int(input("Please enter the k value according to elbow:\n"))
y_wss = []
y_wss,distances,clusters,centers_new = wss(k,c,n)
print("New Centers are\n",centers_new,"\nWSS is:", y_wss)

pre_scatt(train_all,centers_new, clusters,k) #observing segments via each feature pair