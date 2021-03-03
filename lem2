import numpy as np


def np_difference(arr1, arr2):
    S = np.array([], dtype= np.int8)
    k = -1
    for i in arr1[:, 0:2]:
        k += 1
        P = np.array([])
        for j in arr2:
            if np.all(i == j):
                P = np.append(P, 0)
            else:
                P = np.append(P, 1)
        if np.all(P):
            i = i.reshape(1, i.size)
            S = np.append(S, [k])
    return S


def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1
    return i


def matchedRecords(U,T):
    N = T.shape[0]
    a = T[0,0]
    indexs = np.ravel(np.argwhere(U[:,a] == T[0,1]))
    if N>1:
        for i in range(1,N):
            ind = np.ravel(np.argwhere(U[indexs,T[i,0]] == T[i,1]))
            indexs = indexs[ind]
    return indexs


# def lem2algorithm(U,B):
#     print(1)


U = np.array([[1,4,6,9,1],[2,4,7,8,1],[3,5,7,9,2],[3,4,6,8,1],[2,5,6,9,1],[2,5,7,9,2],[3,5,6,9,2]])
# print(U)
# print(enumerate(U)) # <enumerate object at 0x01B89F68>
B = np.array([i for (i, val) in enumerate(U) if val[-1] == 1])
# val是'numpy.ndarray' # np.array是将'list'转换成'numpy.ndarray'
# print(B) # [0 1 3 4]
G = B
G_data = U[G, 0:-1]
# print(G_data)
# print(G_data.shape[1]) # 4
n_attributes = G_data.shape[1]
Q = []

while G.shape[0] > 0:

    T = np.empty([0, 2], dtype = np.int8)
    B_notcontain_T = 1
    TG = np.array([[0,0,0],[0,0,0]],dtype=np.int8)
    # print(type(TG)) # <class 'list'>
    # print(range(n_attributes)) # range(0, 4)
    for i in range(n_attributes):
        # print(i)
        avalue = np.unique(G_data[:, i])
        avalue = avalue[:,np.newaxis]
        # print(avalue)
        tmp = avalue.size
        # print(tmp)
        resdata = np.array([],dtype=np.int8)
        # print(type(resdata)) # <class 'numpy.ndarray'>
        for j in avalue:
            resdata = np.append(resdata,np_count(G_data[:,i],j))
        resdata = resdata[np.newaxis,:].T
        a = i * np.ones((tmp, 1),dtype = np.int8)
        # print(a)
        b = np.concatenate((a, avalue, resdata), axis=1)
        # print(b)
        TG = np.concatenate((TG,b),axis = 0)
        # TG = np.append(TG,b) # [0 1 1 0 2 2 0 3 1 1 4 3 1 5 1 2 6 3 2 7 1 3 8 2 3 9 2]
        # TG = np.vstack((TG,b))
    TG = TG[2:-1,:]
    # print(TG)
    while T.size == 0 or B_notcontain_T:
        max_value = np.max(TG[:,2])
        tmp = np.ravel(np.argwhere(TG[:,2] == max_value))
        # print(tmp)
        if tmp.size > 1:
            av_pairs = TG[tmp,0:2]
            min_value = np.inf
            min_index = -1

            for i in np.arange(tmp.size):
                temp = (np.argwhere(U[:,av_pairs[i][0]] == av_pairs[i,1])).size
                if temp < min_value:
                    min_value = temp
                    min_index = i
            tmp = tmp[min_index]
        # print(tmp.size)
        rule = TG[tmp,0:2].reshape(tmp.size,2)
        T = np.concatenate((T, rule), axis=0)
        m = matchedRecords(U,rule)
        # print(m)
        G = np.intersect1d(m, G);
        # print(G)

        TG = np.empty([0, 3],dtype = np.int8)
        G_data = U[G,0:-1]
        for i in range(n_attributes):
            avalue = np.unique(G_data[:, i])
            avalue = avalue[:,np.newaxis]
            tmp = avalue.size
            resdata = np.array([],dtype=np.int8)
            for j in avalue:
                resdata = np.append(resdata,np_count(G_data[:,i],j))
            resdata = resdata[np.newaxis,:].T
            a = i * np.ones((tmp, 1),dtype = np.int8)
            b = np.concatenate((a, avalue, resdata), axis=1)
            TG = np.concatenate((TG,b),axis = 0)
        S = np_difference(TG[:,0:2], T)
        TG = TG[S,:]
        m = matchedRecords(U, T)
        if np.all(np.in1d(m, B)):
            B_notcontain_T = 0
        else:
            B_notcontain_T = 1

    ss = np.shape(T)[0]
    if ss>1:
        for i in np.arange(ss):
            S = np_difference(T[:,0:2], T[ss-i-1,0:2].reshape(1,2))
            tmp_rule = T[S,:]
            if tmp_rule.size == 0:
                break
            m = matchedRecords(U, tmp_rule)
            if np.all(np.in1d(m,B)):
                T = tmp_rule
    Q = Q + [T]
    new_G = np.array([])
    for i in range(len(Q)):
        rule_T = Q[i]
        new_G = np.append(new_G, matchedRecords(U, rule_T))
    G = np.setdiff1d(B, new_G)
    G_data = U[G, 0:-1]
# print(Q)
ss = len(Q)
w = []
for i in range(1,1+ss):
    ind = ss - i
    w = Q[0:ind] + Q[ind+1:]
    tmp_B = np.array([],dtype=np.int8)
    for j in range(len(w)):
        np.append(tmp_B, matchedRecords(U, w[j]))
        # tmp_B = np.unique(tmp_B)
    if np.all(np.in1d(tmp_B, B))and tmp_B.size == B.size:
        Q = w

print(Q)
