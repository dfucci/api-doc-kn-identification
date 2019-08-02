    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:43:32 2017

@author: saeed
"""

import dataset_utils as du
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
import scikitplot as skplt
import scipy.stats as stats

def save_to_latex():
    n=12
    results=[]
    results_path = '../../results/multi-cado/labels_r/'

    classifier_names = ['LSTM1', 'LSTM2' ,'MLkNN', 'MF1', 'MF2','MK3', 'RAND']
    
    
    for l in range(n):
        tmp = []
        for clf in classifier_names:
            data = du.load_data(results_path+clf)
            l_r = data[l]
            tmp.append([clf]+l_r[0:3])
            
        results.append(tmp)
        
    #fmt = "%d, %d, %d, %s"
    #all_results = np.round(all_results, decimals=3)
    # all_results.astype('str')
    tex_out = "" 
    for k in range(n):
        #np.savetxt(, np.array(all_results[k]))
        du.save_data(results[k], '../../results_out/prf'+str(k)+'.csv', header=['classifier', 'precision', 'recall', 'f1-score'])
        if k%2 == 0:
            tex_out += r'''
\begin{table}[!htb]
            '''
        tex_out += r'''
    \begin{minipage}{.5\textwidth}
        \centering
        \caption{Caption '''+str(k)+'''}
        \label{tab:prf_'''+str(k)+'''}
        \pgfplotstabletypeset[col sep=comma,
     	header=true,  
     	precision=4,
        columns/classifier/.style={string type, column type=r, column name=\ },
        columns={classifier, precision, recall, f1-score},
        highlight col max ={prf'''+str(k)+r'''.csv}{precision}, 
        highlight col max ={prf'''+str(k)+r'''.csv}{recall}, 
        highlight col max ={prf'''+str(k)+r'''.csv}{f1-score},  
        every head row/.style={before row=\\\toprule, after row=\bottomrule}, 
        every even row/.style={before row={\rowcolor[gray]{0.92}}},
        every last row/.style={after row=\bottomrule}  
        ]{prf'''+str(k)+'''.csv}
 	\end{minipage}'''
        if k%2 != 0 and k > 0 or k==n-1:
            tex_out += r'''
\end{table}       
            '''

    
    text_file = open("../../results_out/tables.tex", "w")
    text_file.write(tex_out)
    text_file.close()


def roc_auc_plot(t,p, n_classes):
    y_test = t
    y_score = p
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["macro"], tpr["macro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[2], tpr[2], label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
        


def plot_pr(Y_test, y_score, n_classes):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    #plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))

def plot_pr2(t, ps, n_classes,l=0):
    from itertools import cycle
    # setup plot details
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
        
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for clf in range(n_classes):
        Y_test = t
        y_score = ps[clf]
        precision[clf], recall[clf], _ = precision_recall_curve(Y_test[:, l],
                                                            y_score[:, l])
        average_precision[clf] = average_precision_score(Y_test[:, l], y_score[:, l])
    
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
            y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                             average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))
    
    colors = cycle(['navy', 'turquoise', 'orange', 'cornflowerblue', 'teal', 'salmon'])
    colors = cycle(['#0072B2', '#d20606', '#009E73', '#ffbf00', '#CC79A7', '#999999'])
    #colors = cycle(['navy', 'turquoise'])
    
    
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')

    #l, = plt.plot(recall["micro"], precision["micro"], color='red', lw=2)
    #lines.append(l)
    #labels.append('micro-average Precision-recall (area = {0:0.2f})'
    #              ''.format(average_precision["micro"]))
    classifier_names = ['LSTM1 ', 'LSTM2 ', 'MLkNN', 'MF1     ', 'MF2     ', 'Rand   ']
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('{0} ({1:0.2f})'
                      ''.format(classifier_names[i], average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('')
    plt.legend(lines, labels, loc=(0.01, 0.01), prop=dict(size=10))
    
    
    plt.show()

if __name__ == '__main__':
    
    
    results_path = '../../results/multi-bench/bibtex/'
    results_path = '../../results/multi-cado/'
    
    classifier_names = ['lstm', 'most_frequent', 'uniform']
    classifier_names = ['lstm2']
    k=10
    
    labels = 1
    precision_results = {}
    recall_results = {}
    fscore_results = {}
    amount_results = {}
    auc_results = []
    fr = []
    cfms = {}
    
    java_aucs = []
    net_aucs = []

    for clf_name in classifier_names:
        precision_results[clf_name] = [] 
        recall_results[clf_name] = [] 
        fscore_results[clf_name] = [] 
        amount_results[clf_name] = [] 
        
        cfms[clf_name] = []
    
    d = du.load_data(results_path+'y_test')   
    pr_ids = du.load_data(results_path+'test_pr_ids.csv')   
    t = np.array(d)
    t = t.astype('float')
    
    pr_ids = np.array([int(i[0]) for i in pr_ids])
    
    java = np.array(pr_ids)
    net = np.array(pr_ids)
    java[java==8]=1
    java[java==7]=0
    java=java==1
    
    net[net==8]=0
    net[net==7]=1
    net = net==1    
    true_net = t[net] 
    true_java= t[java]
    
    prs=[]
    thr = .5
    for clf_name in classifier_names:
        auc_results = []
        fr = []
        pr = []
        re = []       
        
        for k_i in range(k):
            #d = du.load_data(results_path+clf_name+'/pr'+str(k_i))        
            ps = []
            
            d = du.load_data(results_path+'lstm1/pr7')     
            p = np.array(d)
            p = p.astype('float')
            ps.append(p)
            d = du.load_data(results_path+'lstm2/pr6')     
            p = np.array(d)
            p = p.astype('float')
            ps.append(p)
            d = du.load_data(results_path+'mlknn/pr0')     
            p = np.array(d)
            p = p.astype('float')
            ps.append(p)
            d = du.load_data(results_path+'dummy/MF1_pr')     
            p = np.array(d)
            p = p.astype('float')
            ps.append(p)
            d = du.load_data(results_path+'dummy/MF2_pr')     
            p = np.array(d)
            p = p.astype('float')
            ps.append(p)
            d = du.load_data(results_path+'dummy/RAND_pr')     
            p = np.array(d)
            p = p.astype('float')
            ps.append(p)
            
            for mm in range(12):
                plot_pr2(t,ps,6,mm)
            xx
            p[p>thr]=1
            p[p<=thr]=0
            pnet=p[net]
            jnet=p[java]
            
            prs.append(p)
            #continue
            
            print("\n")
            print ("Classification report: \n", (classification_report(t, p)))
            print ("Precision macro averaging:",(precision_score(t, p, average='macro')))
            print ("Recall macro averaging:",(recall_score(t, p, average='macro')))
            print ("F1 macro averaging:",(f1_score(t, p, average='macro')))
            #print ("AUC macro: ",(roc_auc_score(t, p, average='macro')))
            print("\n")            
            #print ("AUC micro: ",(roc_auc_score(t, p, average="micro")))
            #print ("Precision micro averaging:",(precision_score(t, p, average='micro')))
            #print ("Recall micro averaging:",(recall_score(t, p, average='micro')))
            #print ("F1 micro averaging:",(f1_score(t, p, average='micro')))
            #print("\n")

            #prf = precision_recall_fscore_support(t,p)
            #auc_results.append(roc_auc_score(t, p, average='macro'))
            fr.append(f1_score(t, p, average='macro'))
            pr.append(precision_score(t,p,average='macro'))
            re.append(recall_score(t,p,average='macro'))
            auc_results.append(roc_auc_score(t, p, average='macro'))
            #precision_results[clf_name].append(prf[0])
            #recall_results[clf_name].append(prf[1])
            #fscore_results[clf_name].append(prf[2])
            #amount_results[clf_name].append(prf[3])
            
            net_aucs.append(roc_auc_score(true_net,pnet,average='micro'))
            java_aucs.append(roc_auc_score(true_java,jnet,average='micro'))
            
            #cfm = []
            #for i in range(t.shape[1]):
            #    cfm.append(confusion_matrix(t[:,i], p[:,i]))

            #cfms[clf_name].append(cfm)
        print("MEAN pr: ", np.max(pr), ", MAX index: ", np.argmax(pr))
        print("MEAN re: ", np.max(re), ", MAX index: ", np.argmax(re))
        print("MEAN F1: ", np.max(fr), ", MAX index: ", np.argmax(fr))
        print("MEAN AUC: ", np.max(auc_results), ", MAX index: ", np.argmax(auc_results))
        plot_pr(t,p,12)
    """
    l = 1
    all_results = []
    print(fscore_results)
    for l in range(labels):
        if l == 4:
            continue
        tmp = []
        for clf_name in classifier_names:    
            tmp.append([clf_name, np.mean(np.array(precision_results[clf_name])[:,l]), 
                       np.mean(np.array(recall_results[clf_name])[:,l]), 
                       np.mean(np.array(fscore_results[clf_name])[:,l])])
        
        all_results.append(tmp)


    for clf_name in classifier_names:       
        #precision_results[clf_name] = [np.max(i) for i in precision_results[clf_name]]
        #recall_results[clf_name] = [np.max(i) for i in recall_results[clf_name]]
        #fscore_results[clf_name] = [np.max(i) for i in fscore_results[clf_name]]

        print(clf_name)
        print("precision: %s" % np.mean(np.array(precision_results[clf_name])))
        print("recall: %s" % np.mean(np.array(recall_results[clf_name])))
        print("fscore: %s" % np.mean(np.array(fscore_results[clf_name])))

    """