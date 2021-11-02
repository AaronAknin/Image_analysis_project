def PerformanceEvaluation(predict,class_test):
    L1 = [p[0] for p in predict]
    L2 = [p[1] for p in predict]
    Cosine = [p[2] for p in predict]
    sl1 = 0
    sl2 = 0
    scosine = 0
    for i in range(len(L1)):
        if L1[i] == class_test[i]:
            sl1 +=1
        if L2[i] == class_test[i]:
            sl2 +=1
        if Cosine[i] == class_test[i]:
            scosine +=1
    crr_l1 = sl1/len(L1)
    crr_l2 = sl2/len(L2)
    crr_cosine = scosine/len(Cosine)
    return crr_l1,crr_l2,crr_cosine