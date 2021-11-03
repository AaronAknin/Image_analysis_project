import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import roc_curve, auc

#calculating the CRR for the identification mode (CRR for all three measures, i.e., 
#L1, L2, and Cosine similarity, should be >=75% , the higher the better)
def PerformanceEvaluation(predict,class_test):
    L1 = [p[0] for p in predict]
    L2 = [p[1] for p in predict]
    Cosine = [p[2] for p in predict]
    sl1 = 0
    sl2 = 0
    scosine = 0
    #evaluate the matching result by checking if the prediction by distance matches the test labels
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
    return [crr_l1,crr_l2,crr_cosine]

    
def draw_Table3(predict_orig, predict_reduced, class_test):
    #store the crr of original feature set and reduced feature set
    predict_orig = PerformanceEvaluation(predict_orig, class_test)
    predict_reduced = PerformanceEvaluation(predict_reduced, class_test)
    
    columns = ('Original feature set', 'Reduced feature set')
    rows = ('L1 distance measure', 'L2 distance measure', 'Cosine similarity measure')
    cell_text = [["{:X}".format(i) for i in predict_orig], ["{:X}".format(i) for i in predict_reduced]] 
       
    fig, ax = plt.subplots() 
    ax.set_axis_off() 
    table = ax.table( 
        cellText = cell_text,
        rowLabels = rows,  
        colLabels = columns, 
        cellLoc ='center',  
        loc ='upper left')         
       
    ax.set_title('TABLE 3: Recognition Results Using Different Similarity Measure', 
                 fontweight ="bold") 
       
    plt.show() 
    
def plot_Fig10(predict_reduced, class_test):
    #Varaition of the recognition rate with changes of dimensionality of the reduced feature rate
    #dim<=107
    dim = range(1, 108)
    crr_reduced_cosine = []
    
    #calculate the crr of reduced feature vector using cosine similarity with different dimension(components in IrisMatching)
    for i in range(dim):
        crr = PerformanceEvaluation(predict_reduced, class_test)
        crr_reduced_cosine.append(crr[2])
    
    plt.plot(crr_reduced_cosine, dim)
    plt.ylabel("Correct recognition rate")
    plt.xlabel("Dimensionality of the feature vector")
    plt.show()
    
def plot_ROCCurve(X_test, y_test):
    model = NearestCentroid()
    y_score = model.fit(X_test, y_test).score(X_test, y_test)
    
    #calculate ROC curve
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(108):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    #plotting
    plt.figure()
    lw = 2
    plt.plot(
        fpr[2],
        tpr[2],
        color="blue",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    plt.show()
    
