import numpy as np
import matplotlib.pyplot as plt

def GaussianMatrix(X,X_test,sigma):
    row,col=X.shape[0],X_test.shape[0]
    GassMatrix=np.zeros(shape=(row,col))
    X=np.asarray(X)
    i=0
    for v_i in X:
        j=0
        for v_j in X_test:
            GassMatrix[i,j]=Gaussian(v_i.T,v_j.T,sigma)
            j+=1
        i+=1
    return GassMatrix
def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))
sigma = 15
points = 150
x = np.linspace(0,2 * np.pi,points,endpoint=True)
# x = np.arange(0,40,0.5)
y = np.sin(x)
y_noise = y + 0.05 * np.random.randn(points) #加噪声
plt.scatter(x,y_noise,color = 'r',label='train points')
K = GaussianMatrix(x,x,sigma)
w = np.linalg.pinv(np.dot(K.T, K)).dot(K.T).dot(y_noise)#参数
x_test = np.linspace(0,2 * np.pi,20)
k = GaussianMatrix(x,x_test,sigma)
y_test = w.dot(k)

plt.plot(x_test,y_test,label='estimate y')
plt.plot(x,y,color='b',label='real y')
plt.legend()
plt.show()