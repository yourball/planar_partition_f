from __future__ import division  
import numpy as np
from numpy import logical_and as AND
from numpy import logical_or as OR
from numpy import logical_xor as XOR
#from numpy import logical_xor as XOR
from numpy import logical_not as NOT
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

def symmetrize(matrix):
    matrix = np.triu(matrix)
    return matrix+matrix.T

def is_symmetric(adjmatrix):
    return np.sum(np.abs(np.int_(adjmatrix)-np.int_(adjmatrix).T))==0

def trnglr_lattice(L,Lph,T,p,prob,plot=False):
    #-------------------------------------------------------------------------
    basisx = np.hstack((np.tile(np.hstack((np.arange(L),np.arange(-p,L-1+p))),int(T/2)),np.arange(L)))
    basisy = np.hstack((np.repeat(np.arange(T),np.tile([L,L-1+2*p],int(T/2))),np.repeat(T,L)))
    S = len(basisx)
    #--------------------------------------------------------------------------
    X,Y = np.meshgrid(basisx,basisy)
    Rset = OR(AND(AND(X-X.T==1,np.abs(Y-Y.T)==1),Y%2==1),AND(AND(X-X.T==0,np.abs(Y-Y.T)==1),Y%2==0))
    Lset = OR(AND(AND(X-X.T==-1,np.abs(Y-Y.T)==1),Y%2==0),AND(AND(X-X.T==0,np.abs(Y-Y.T)==1),Y%2==1))
    #---
    Rset0 = np.triu(Rset)
    Rset0[Rset0] = np.random.choice([True,False],np.count_nonzero(Rset0),p=(1-prob,prob))
    Rset0 += Rset0.T
    #---
    Lset0 = np.triu(Lset)
    Lset0[Lset0] = np.random.choice([True,False],np.count_nonzero(Lset0),p=(1-prob,prob))
    Lset0 += Lset0.T
    #---
    Hset0 = symmetrize(np.dot(np.triu(Rset0),np.tril(Lset0)))
    #---
    Rset1 = np.triu(Rset)
    Rset1 += Rset1.T
    #---
    Lset1 = np.triu(Lset)
    Lset1 += Lset1.T
    #---
    Hset1 = symmetrize(np.dot(np.triu(Rset1),np.tril(Lset1)))
    #---
    not_Rset = XOR(Rset0,Rset1)
    not_Lset = XOR(Lset0,Lset1)
    not_Hset  = XOR(Hset0,Hset1)
    #---
    
    diag_links_renorm = symmetrize(OR(AND(np.dot(np.tril(not_Hset),np.triu(not_Rset)),Lset0),AND(np.dot(np.triu(not_Hset),np.triu(not_Lset)),Rset0)))
    horizontal_links = AND(XOR(allOR([Rset1,Lset1,Hset1]),OR(diag_links_renorm,allOR([not_Rset,not_Lset,not_Hset]))),Hset0)
    empty_links = allOR([not_Rset,not_Lset,not_Hset])
    fm_links = AND(AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0),AND(Y==0,X*X.T!=int(L/2)*(int(L/2)-1)))
    antifm_links = AND(AND(np.abs(X-X.T)==1,np.abs(Y-Y.T)==0),AND(Y==0,X*X.T==int(L/2)*(int(L/2)-1)))
    bc_links = symmetrize(AND(AND(OR(AND(X-X.T==0,np.abs(Y-Y.T)==1),(X-X.T)*(Y-Y.T)==1-2*(Y%2)),AND(Y%2!=p,OR(X==-p,X==L-1))),NOT(empty_links)))
    diagonal_links = AND(AND(XOR(allOR([Rset1,Lset1,Hset1]),OR(diag_links_renorm,allOR([not_Rset,not_Lset,not_Hset]))),allOR([Rset0,Lset0])),NOT(bc_links))
    
    sole_measurements = int(np.count_nonzero(diag_links_renorm)/2)
    double_measurements = int(np.count_nonzero(not_Hset)/2)-sole_measurements
    #-----------------------------------------------------------------------------------------------------------------------------------------
    
    empty_links_diag = AND(empty_links,allOR([Rset1,Lset1]))
    measured = np.zeros([T,Lph],bool)
    for i in range(S):
        v = np.zeros(S,bool)
        v[i+1:] = empty_links_diag[i][i+1:]
        if p==0 and np.count_nonzero(v)>0:
            measured[basisy[i]][2*basisx[i]+basisx[v]-basisx[i]-(basisy[i]+1)%2+1-basisy[i]%2] = True
        if p==1 and np.count_nonzero(v)>0:
            measured[basisy[i]][2*basisx[i]+basisx[v]-basisx[i]+1] = True
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    
    sets = [diagonal_links,horizontal_links,diag_links_renorm,empty_links,fm_links,antifm_links,bc_links]
    
    if plot:
        visualize(basisx,basisy,horizontal_links,c='b')
        visualize(basisx,basisy,diagonal_links,c='k')
        visualize(basisx,basisy,empty_links,c='r',ls="--") 
        visualize(basisx,basisy,diag_links_renorm,c='g',ls=":")
        visualize(basisx,basisy,fm_links,c='orange',ls=":")
        visualize(basisx,basisy,antifm_links,c='cyan',ls=":")
        visualize(basisx,basisy,bc_links,c='violet',ls=":")
    
    return basisx,basisy,sets,measured,sole_measurements,double_measurements

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

def visualize(basisx,basisy,adjmatrix,c='k',ls="-"):
    for i in range(len(adjmatrix)):
        for j in range(i+1,len(adjmatrix)):
            if adjmatrix[i][j]:#==-0.5-G1:
                plt.plot([basisx[i]+0.5*(basisy[i]%2),basisx[j]+0.5*(basisy[j]%2)],[basisy[i],basisy[j]],c=c,ls=ls)
    plt.axis('off')
    for i in range(len(adjmatrix)):
        plt.text(basisx[i]+0.5*(basisy[i]%2)-0.05,basisy[i]+0.1,str(i))
#    for i in range(len(jmatrix)):
#        for j in range(i+1,len(jmatrix)):
#            if adjmatrix[i][j]:
#                plt.text(0.5*(basisx[i]+basisx[j])+0.25*(basisy[j]%2)+0.25*(basisy[i]%2)-0.025,0.5*(basisy[i]+basisy[j])-0.025,jmatrix[i,j],bbox={'facecolor': 'white', 'alpha': 1, 'pad': 1})
    return 0

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

def model_couplings(sets,G,double_measurements,tp = 'num',q=2):
    beta = np.log((q**2+1)/q)
    if tp == 'num':
        sign=1
    if tp == 'denom':
        sign=-1
    G1,G2,G3 = G,G,G
    K = 4*np.cosh(beta/2+2*beta*G1)*np.cosh(beta/2)
    D = -1/(2*beta)*np.log(K)
    c = 1/(2*beta)*np.log(np.cosh(beta/2+2*beta*G1)/np.cosh(beta/2))
    Y = np.exp(beta*(1+G1))+2*np.exp(+beta*G1)+np.exp(-beta*(1+3*G1))
    jmatrix,E0 = np.zeros([len(sets[0]),len(sets[0])]),0
    diagonal_links,horizontal_links,diag_links_renorm,empty_links,fm_links,antifm_links,bc_links = sets
    jmatrix = -(0.5+G1)*diagonal_links+G1*horizontal_links-(0.5+G1-c)*diag_links_renorm-G2*fm_links+sign*G2*antifm_links-(G3+0.5)*bc_links
    E0 += 0.5*(0.5*np.count_nonzero(diagonal_links)+G1*np.count_nonzero(horizontal_links)+(0.5+G1+D)*np.count_nonzero(diag_links_renorm)+\
               +G2*np.count_nonzero(fm_links)+G2*np.count_nonzero(antifm_links)+(G3+0.5)*np.count_nonzero(bc_links))-double_measurements/beta*np.log(Y)
    return jmatrix,E0

#=================================================================================

def spin_basis(num_sites):
    full_basis = np.array(list(itertools.product([-1,1], repeat=num_sites))) 
    return full_basis
    
def partition_function(jmatrix,E0,beta):
    E = np.array([])
    basis = spin_basis(len(jmatrix))
    for i in range(len(basis)):
        state = basis[i]
        E = np.append(E,E0+0.5*np.dot(state,np.dot(jmatrix,state)))
    S = -np.log(0.5*np.sum(np.exp(-beta*E)))
    return S 

def partition_function_method_direct(sets,G,double_measurements,q=2):
    beta = np.log((q**2+1)/q)
    jmatrix,E0 = model_couplings(sets,G,double_measurements,tp = 'num') 
    S_num = partition_function(jmatrix,E0,beta)
    jmatrix,E0 = model_couplings(sets,G,double_measurements,tp = 'denom') 
    S_den = partition_function(jmatrix,E0,beta)
    return S_num-S_den

def unitary_update(P,Omega,t,s,smp,L):
    walls = XOR(Omega,np.roll(Omega,+1))
    walls[(t+s)%2::2] = False
    walls[::L] = False
    supp = np.zeros(smp*L,bool)
    supp[walls] = np.random.randint(2,size=np.count_nonzero(walls))>0
    supp  = XOR(supp,np.roll(supp,-1))
    walls = XOR(walls,supp)
    Omega = XOR(Omega,walls)
    P += np.count_nonzero(walls.reshape(smp,L),axis=1)
    return P,Omega

def cellular_automaton_method(L,tm,measured,smp_std,q=2):
    s = (int(L/2)+1)%2
    beta2 = np.log((q**2+1)/(2*q))
    exp_sumA,exp_sum0 = np.zeros(tm+1,float),np.zeros(tm+1,float)
    exp_sumA[0],exp_sum0[0] = smp_std,smp_std
    P0,PA = np.zeros(smp_std,int),np.zeros(smp_std,int)
    Omega0 = np.zeros(smp_std*L,bool)
    OmegaA = np.repeat(True,2*smp_std)
    OmegaA[::2] = False
    OmegaA = np.repeat(OmegaA,int(L/2))
    for t in range(tm):
        C = np.tile(measured[t],smp_std)
        K = np.random.randint(2,size=np.count_nonzero(C))>0
        C[C] = K
        Omega0[C],OmegaA[C] = NOT(Omega0[C]),NOT(OmegaA[C])
        P0,Omega0 = unitary_update(P0,Omega0,t,s,smp_std,L)
        PA,OmegaA = unitary_update(PA,OmegaA,t,s,smp_std,L)
        exp_sumA[t+1] = np.sum(np.exp(-beta2*PA))
        exp_sum0[t+1] = np.sum(np.exp(-beta2*P0))
    RE = -np.log(exp_sumA)+np.log(exp_sum0)
    return RE[-1]


def trnglr_lattice_old(L,T,p):
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

def model_couplings_old(sets,G):
    G1,G2,G3=G,G,G
    set_mid,set_frm,set_afm,set_ang,set_bcd = sets
    jmatrix = -0.5*(set_ang+set_bcd)+G1*(set_mid+set_frm+set_afm)-G1*set_ang-G2*set_frm+G2*set_afm-G3*set_bcd
    Delta = 0.5*(0.5*np.count_nonzero(set_ang+set_bcd)+G2*np.count_nonzero(set_frm+set_afm)\
                    +G1*np.count_nonzero(set_mid+set_frm+set_afm)+G3*np.count_nonzero(set_bcd))
    return jmatrix,Delta


def cellular_automaton_method_old(L,tmax,q=2):
    s = (int(L/2)+1)%2
    P = np.ones(L+1,float)
    for t in range(tmax):
        P1,P2 = np.roll(P,+1),np.roll(P,-1)
        P[(t+s)%2::2] = q/(q**2+1)*(P1[(t+s)%2::2]+P2[(t+s)%2::2])
        P[0],P[-1] = 1,1
    S = -np.log(P[int(L/2)])
    return S
#=================================================================================
    
#Lph,T = 6,2
#L = int(Lph/2)+(int(Lph/2)%2)
#p = 1-(int(Lph/2)%2)
#
##=================================================================================
#
#basisx,basisy,sets,measured,sole_measurements,double_measurements = trnglr_lattice(L,Lph,T,p,prob=0.1,plot=False)
#adjmatrix = allOR(sets)
#database = list_edges(adjmatrix,basisx,basisy)  
#
###=================================================================================
#
#S1 = partition_function_method(sets,G=100,double_measurements=double_measurements)
#S2 = cellular_automaton_method(Lph,T,measured,smp_std=100000)
#print(S1)
#print(S2)

#=================================================================================