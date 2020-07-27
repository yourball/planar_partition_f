from __future__ import division  
import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
#from numpy import logical_xor as XOR
#from numpy import logical_not as NOT
import itertools
import matplotlib.pyplot as plt

#=====================================================================================================================================

def allOR(A):
    res = A[0]
    for i in range(1,len(A)):
        res = OR(res,A[i])
    return res

def add(a,B):
    if len(B)==0:
        B = a
    else:
        B = np.vstack((B,a))  
    return B

def angle(n):
    n = n-1e-12
    n = n/np.sqrt(np.sum(n**2))
    phi = 2*np.pi*(n[1]<0)+np.arccos(n[0])*np.sign(n[1])
    return phi%(2*np.pi)

def trnglr_lattice(L,T,p):
    #-------------------------------------------------------------------------
    basisx = np.hstack((np.tile(np.hstack((np.arange(L),np.arange(-p,L-1+p))),int(T/2)),np.arange(L)))
    basisy = np.hstack((np.repeat(np.arange(T),np.tile([L,L-1+2*p],int(T/2))),np.repeat(T,L)))
    #--------------------------------------------------------------------------
    X,Y = np.meshgrid(basisx,basisy)
    set_mid = AND(AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0),AND(Y!=T,Y!=0))
    set_frm = AND(AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0),AND(Y==0,X*X.T!=int(L/2)*(int(L/2)-1)))
    set_afm = AND(AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0),AND(Y==0,X*X.T==int(L/2)*(int(L/2)-1)))
    set_ang = AND(OR(AND(X-X.T==0,np.abs(Y-Y.T)==1),(X-X.T)*(Y-Y.T)==1-2*(Y%2)),OR(Y%2==p,AND(X!=-p,X!=L-1)))#AND(Y%2==p,X!=0))
    set_bcd = AND(OR(AND(X-X.T==0,np.abs(Y-Y.T)==1),(X-X.T)*(Y-Y.T)==1-2*(Y%2)),AND(Y%2!=p,OR(X==-p,X==L-1)))
    set_ang = np.triu(set_ang)
    set_ang += set_ang.T
    set_bcd = np.triu(set_bcd)
    set_bcd += set_bcd.T
    sets = [set_mid,set_frm,set_afm,set_ang,set_bcd]#,set_bcd]
    return basisx,basisy,sets

def list_edges(adjmatrix,basisx,basisy):
    S = len(adjmatrix)
    edges = []
    for i in range(S):
        for j in range(i+1,S):
            if adjmatrix[i][j]!=0:
                edges = add([i,j],edges)
    E = len(edges)
    database = np.zeros([E,7],int)
    for i in range(E):
        database[i][0] = i
        database[i][1:3] = edges[i]
        #---------------------------------------
        vertex = edges[i][0]
        nn = np.arange(S)[adjmatrix[vertex]]
        nnidx, nnang = np.zeros(len(nn),int),np.zeros(len(nn))
        for s in range(len(nn)):
            nnidx[s] = np.arange(E)[OR(AND(edges.T[0]==vertex,edges.T[1]==nn[s]),AND(edges.T[1]==vertex,edges.T[0]==nn[s]))][0]
            n = np.array([basisx[nn[s]]-basisx[vertex],basisy[nn[s]]-basisy[vertex]])
            nnang[s] = angle(n)
        nnidx = nnidx[np.argsort(nnang)]
        indx0 = np.arange(len(nnidx))[nnidx==i][0]
        database[i][3],database[i][4] = nnidx[(indx0+1)%len(nnidx)],nnidx[(indx0-1)%len(nnidx)]
        #---------------------------------------
        vertex = edges[i][1]
        nn = np.arange(S)[adjmatrix[vertex]]
        nnidx, nnang = np.zeros(len(nn),int),np.zeros(len(nn))
        for s in range(len(nn)):
            nnidx[s] = np.arange(E)[OR(AND(edges.T[0]==vertex,edges.T[1]==nn[s]),AND(edges.T[1]==vertex,edges.T[0]==nn[s]))][0]
            n = np.array([basisx[nn[s]]-basisx[vertex],basisy[nn[s]]-basisy[vertex]])
            nnang[s] = angle(n)
        nnidx = nnidx[np.argsort(nnang)]
        indx0 = np.arange(len(nnidx))[nnidx==i][0]
        database[i][5],database[i][6] = nnidx[(indx0+1)%len(nnidx)],nnidx[(indx0-1)%len(nnidx)]
    return database

def visualize(database,basisx,basisy,adjmatrix,jmatrix):
    for i in range(len(adjmatrix)):
        for j in range(i+1,len(adjmatrix)):
            if adjmatrix[i][j]:#==-0.5-G1:
                plt.plot([basisx[i]+0.5*(basisy[i]%2),basisx[j]+0.5*(basisy[j]%2)],[basisy[i],basisy[j]],c='k')
    plt.axis('off')
    for i in range(len(adjmatrix)):
        plt.text(basisx[i]+0.5*(basisy[i]%2)-0.05,basisy[i]+0.1,str(i))
    for k in range(len(database)):
        i,j = database[k][1:3]
        #plt.text(0.5*(basisx[i]+basisx[j])+0.25*(basisy[j]%2)+0.25*(basisy[i]%2)-0.025,0.5*(basisy[i]+basisy[j])-0.025,str(k),bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
        plt.text(0.5*(basisx[i]+basisx[j])+0.25*(basisy[j]%2)+0.25*(basisy[i]%2)-0.025,0.5*(basisy[i]+basisy[j])-0.025,jmatrix[i,j],bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
    return 0

#def plot_conf(state,basisx,basisy,jmatrix):
#    plt.scatter(basisx+0.5*(basisy%2),basisy,s=1/max(L,T)*4000,edgecolors='k',c=state,cmap='gray')
#    plt.axis('off')
#    return 0

#=====================================================================================================================================
# generate random couplings from adjacency matrix
    
def random_couplings(adjmatrix):
    jmatrix = np.zeros([len(adjmatrix),len(adjmatrix)])
    for i in range(len(adjmatrix)):
        for j in range(i+1,len(adjmatrix)):
            if adjmatrix[i][j]!=0:
                jmatrix[i,j] = 0.5*np.random.normal(0,1)
                jmatrix[j,i] = jmatrix[i,j]
    return jmatrix

def model_couplings(sets,G):
    G1,G2,G3=G,G,G
    set_mid,set_frm,set_afm,set_ang,set_bcd = sets
    jmatrix = -0.5*(set_ang+set_bcd)+G1*(set_mid+set_frm+set_afm)-G1*set_ang-G2*set_frm+G2*set_afm-G3*set_bcd
    Delta = 0.5*(0.5*np.count_nonzero(set_ang+set_bcd)+G2*np.count_nonzero(set_frm+set_afm)\
                    +G1*np.count_nonzero(set_mid+set_frm+set_afm)+G3*np.count_nonzero(set_bcd))
    return jmatrix,Delta

#=================================================================================

def spin_basis(num_sites):
    full_basis = np.array(list(itertools.product([-1,1], repeat=num_sites))) 
    return full_basis
    
def partition_function_method(jmatrix,adjmatrix,E0,beta):
    E = np.array([])
    basis = spin_basis(len(adjmatrix))
    for i in range(len(basis)):
        state = basis[i]
        E = np.append(E,E0+0.5*np.dot(state,np.dot(jmatrix,state)))
    S = -np.log(0.5*np.sum(np.exp(-beta*E)))#+np.log(np.sum(np.exp(np.zeros(len(E[E<100])))))
    return S 

def cellular_automaton_method(L,tmax,q=2):
    s = (int(L/2)+1)%2
    P = np.ones(L+1,float)
    for t in range(tmax):
        P1,P2 = np.roll(P,+1),np.roll(P,-1)
        P[(t+s)%2::2] = q/(q**2+1)*(P1[(t+s)%2::2]+P2[(t+s)%2::2])
        P[0],P[-1] = 1,1
    S = -np.log(P[int(L/2)])
    return S

#=================================================================================
#
#Lph,T = 4,2
#
#q=2
#beta = np.log((q**2+1)/q)
#L = int(Lph/2)+(int(Lph/2)%2)
#p = 1-(int(Lph/2)%2)
#
##=================================================================================
#
#basisx,basisy,sets = trnglr_lattice(L,T,p)
#adjmatrix = allOR(sets)
#jmatrix,E0 = model_couplings(sets,G=100) 
#database = list_edges(adjmatrix,basisx,basisy)
#visualize(database,basisx,basisy,adjmatrix,jmatrix)  
#       
##=================================================================================
#
#S1 = partition_function_method(jmatrix,adjmatrix,E0,beta)
#S2 = cellular_automaton_method(L=Lph,tmax=T)
#         
#print(S1)
#print(S2)
#=================================================================================