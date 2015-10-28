import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from string import punctuation
from sklearn import metrics

def determine_svm_performance(clf, phi, t, k=5):
    from sklearn import cross_validation
    skf = cross_validation.StratifiedKFold(t,k)
    avg_perfs = []
    avg_perf=np.float64()
    for train, test in skf :
        phi_train = phi[train]
        phi_test = phi[test]
        t_train = t[train]
        t_test = t[test]
        clf.fit(phi_train,t_train)
        t_pred = clf.decision_function(phi_test)
        t_pred_binary = [ np.sign(t_i) for t_i in t_pred ]
        perf = metrics.accuracy_score(t_test, t_pred_binary)
        avg_perfs.append(perf)
    avg_perf = np.mean(avg_perfs)
    return avg_perf

def select_svm_c(phi, t, k=5):
    cs = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
    accuracies = []
    for c in cs:
        clf = SVC(kernel='linear',C=c)
        acc = determine_svm_performance(clf,phi,t,k)
        accuracies.append((c,acc))
    best_c_acc = max(accuracies, key = lambda x: x[1])
    return best_c_acc
   
def main():
    fname_x_train = 'debug_x_train.dat'
    fname_t_train = 'debug_t_train.dat'
    phi_train = np.loadtxt(fname_x_train)
    t_train = np.loadtxt(fname_t_train)
    fname_x_test = 'debug_x_test.dat'
    fname_t_test = 'debug_t_test.dat'
    phi_test = np.loadtxt(fname_x_test)
    t_test = np.loadtxt(fname_t_test)

    k = 5
    best_c, best_acc = select_svm_c(phi_train, t_train, k)
    print "best C: %.4f, best acc: %.4f" %(best_c, best_acc)

    # now test the SVM
    clf = SVC(kernel='linear', C=best_c)
    clf.fit(phi_train,t_train)
    t_pred = clf.decision_function(phi_test)
    t_pred_binary = [ np.sign(t_i) for t_i in t_pred ]
    perf = metrics.accuracy_score(t_test, t_pred_binary)
    print "best perf: " + str(perf)

    
main()
