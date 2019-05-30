import math
import numpy as np
from scipy.spatial.distance import squareform as squareform
from scipy.interpolate import interp1d
from scipy import sparse
import random
from scipy.spatial.distance import pdist

def linmap(x,a,b): 
    return (((x - a[0]) * (b[1] - b[0])) / (a[1] - a[0])) + b[0]

def norm(S):
    return np.linalg.norm(S)

def adj(S):  
    return S.conj().T

def isvec(z): 
    return z.shape == (z.size,)

def sum_squareform(n):

    ncols=int(n*(n-1)/2)
    I=[0.0]*ncols
    J=[0.0]*ncols

    val = [1]*(2*ncols)
    k=0
    
    for i in range(1,n):
        I[k: k + (n-i)] = list(range(i, n))
        k = k + (n-i)
    k = 0
    for i in range(1,n):
        J[k: k + (n-i)] = [i-1]*(n-i)
        k = k + (n-i)
    # print(J+I,I+J)
    # S= sparse.coo_matrix([range(ncols), range(ncols)], [I +J ,J+ I], 1, ncols,n)
    S= sparse.coo_matrix((val, ( list(range(ncols))+list(range(ncols)) ,I+J)),  shape=(ncols, n) )

    return S , S.T

def kop(w,S) :
    return S * w


def feval(w,z): 
    return 2*w.T*z


def fprox(w,c, z): 
    a =w-2*c*z
    return a.clip(min=0) 


def geval(z,a): 
    return -a*sum(np.log(z))
    

def gstarprox(z,a,c): 
    return z- np.sqrt(z**2 + 4*a*c)/2
   

def hparms (b,c,w,w0): 
    if w0==0: 
        return b*norm(w)**2,  2*b*w, 2*b
    else:
        return b*norm(w)**2 + c*norm(w-w0)**2, 2*((b+c)*w -c*w0), 2*(b+c)
 
def gauss(p1, p2, sigma):
    return math.exp(- norm(p1-p2)**2 / sigma**2 )


def algo1(Z,a,b,c,w0, step_size, e, imax ):
    if isvec(Z):
        z=Z
    else:
        z=squareform(Z)
    l=len(z)
    
    # print(z)

    n=int((1+math.sqrt(1+8*l))//2)
    # print(n)
    if w0!=0:
        if isvec(w0):
            w0=w0
        else:
            w0=squareform(w0)
    else:
        w0=0
    # print(w0)
    w=np.zeros(z.shape)
    St, S = sum_squareform(n)
    print(S)
    normk = math.sqrt(2*(n-1))
    # print(normk)

    heval, hgrad, hbeta = hparms(b,c,w,w0)
    # print(hgrad)

    mu= hbeta + normk
    # print(mu)

    epsilon =linmap(e, [0, 1/(1+mu)], [0,1])
    # print(epsilon)


    # print (w.shape, S.shape)

    vn= kop(w, S)
    # print(vn)
    gn = linmap(step_size, [epsilon, (1-epsilon)/mu], [0,1])
    for i in range(imax):
        Yn=w-gn*(2*b*w + kop(vn, St))
        yn= vn + gn*(kop(w, S))
        Pn= fprox(Yn, gn, z)
        pn= gstarprox(yn, a, gn)
        Qn= Pn - gn*( 2*b*Pn + kop(pn, St))
        qn = pn + gn*kop(Pn, S)
        w = w - Yn+Qn
        vn = vn - yn +qn
        l1 =norm(-Yn +Qn)/norm(w)
        l2 = norm(-yn +qn)/norm(vn) 
        if  l1 < epsilon and l2 < epsilon :
            break
        print(l1, l2)
    return w, St


n=100

s = 1/2/math.sqrt(2)


edges =  [np.array([random.uniform(0,1), random.uniform(0,1)]) for i in range(n)]

inp= np.zeros((n,n))

for i in range(n):
    for j in range(n):
        inp[i][j]= gauss(edges[i], edges[j], .2)

for i in range(n):
    inp[i][i]=0.0


inp =np.asarray(inp)

a=.3
b=.5
heval, hgrad, hbeta = hparms(b,0,w,0)

w,S = algo1(inp,a, b, 0 , 0 ,1e-2, 1e-5, 100)
print(feval(w, squareform(inp)) + geval(kop(w,S), a) + heval)

w[w<1e-5] = 0
