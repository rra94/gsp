import math
import numpy as np
from scipy.spatial.distance import squareform as squareform
from scipy.interpolate import interp1d
from scipy import sparse

def linmap(x,a,b): return (((x - a[0]) * (b[1] - b[0])) / (a[1] - a[0])) + b[0]

def norm(S):return np.linalg.norm(S)

def adj(S):  return S.conj().T

def isvec(z): return z.shape == (z.size,)

def sum_squareform(n):

    ncols=int(n*(n-1)/2)
    I=[0.0]*ncols
    J=[0.0]*ncols

    val = [1]*ncols
    k=0
    
    for i in range(1,n):
        I[k: k + (n-i)] = list(range(i, n))
        k = k + (n-i+1)
    k = 0
    for i in range(1,n):
        J[k: k + (n-i)] = i-1
        k = k + (n-i+1)
    
    S= sparse.coo_matrix((val, (I,J)),  shape=(ncols, n) )

    return S , S.T

def kop(w,S) : return S * w

def feval(w,z): return 2*w.T*z

def fprox(w,c, z): return np.clip(w-2*c*z, 0) 

def geval(z,a): return -a*sum(np.log(z))
    
def gstarprox(z,a,c): return z- np.sqrt(z**2 + 4*a*c)/2
   
def hparms (b,c,w,w0): 
    if w0==0: 
        return b*norm(w)**2,  2*b*w, 2*b
    else:
        return b*norm(w)**2 + c*norm(w-w0)**2, 2*((b+c)*w -c*w0), 2*(b+c)
 

def algo1(Z,a,b,w0,d0, step_size, e, imax ):
    w1=np.zeroes(n)
    y1=np.zeros(n)

    if isvec(Z):
        z=Z
    else:
        z=squareform(Z)

    l=len(z)
    n=(1+math.sqrt(1+8*l))//2
    
    if w0!=0:
        if isvec(w0):
            w0=w0
        else:
            w0=squareform(w0)
    else:
        w0=0

    w=np.zeros((z.size, z.size))

    S, St = sum_squareform(n)

    normk = math.sqrt(2*(n-1))
    
    heval, hgrad, hbeta = hparms(b,c,w,w0)

    mu= hbeta + normk

    epsilon =lin_map(e, [0, 1/(1+mu)], [0,1])

    vn= kop(w)
    
    gn = lin_map(step_size, [epsilon, (1-epsilon)/mu], [0,1])

    for i in range(imax):
        Yn=w-gn*(2*b*w + kop(vn, St))
        yn= vn + gn*(kop(w, S))
        Pn= fprox(Yn, gn, z)
        pn= gstarprox(yn, a, gn)
        Qn= Pn - gn*( 2*b*Pn + kop(pn, St))
        qn = pn + gn*kop(Pn, S))
        w = w - Yn+Qn
        vn = vn - yn +qn
        if norm(-Yn +Qn)/norm(w) < epsilon and norm(-yn +qn)/norm(vn) <epsilon :
            break

    return squareform(w)
