import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from string import punctuation
from sklearn import metrics
from sklearn.externals import joblib

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
        t_pred = clf.predict(phi_test)
        perf = metrics.accuracy_score(t_test, t_pred)
        avg_perfs.append(perf)
    avg_perf = np.mean(avg_perfs)
    return avg_perf

# def select_svm_hyperparameters(phi, t, k=5):
#     cs = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
#     gammas = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
#     accuracies = []
#     for gamma in gammas :
#         for c in cs:
#             clf = SVC(kernel='linear',C=c)
#             acc = determine_svm_performance(clf,phi,t,k)
#             accuracies.append((c,acc))
#     best_c_acc = max(accuracies, key = lambda x: x[1])
#     return best_c_acc
   
def select_hyperparam_rbf(phi, t, k=5):
    cs = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
    gammas = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
    accuracies = []
    for gamma in gammas :
        for c in cs:
            clf = SVC(kernel='rbf',C=c, gamma=gamma)
            acc = determine_svm_performance(clf,phi,t,k)
            print ("acc: %.4f, C: %.4f, gamma: %.4f" %(acc, c, gamma))
            accuracies.append((acc,c,gamma))
    best_acc_c_gamma = max(accuracies, key = lambda x: x[0])
    print best_acc_c_gamma
    return best_acc_c_gamma

def select_hyperparam_linear(phi, t, k=5):
    cs = [ 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2 ]
    accuracies = []
    for c in cs:
        clf = SVC(kernel='linear',C=c)
        acc = determine_svm_performance(clf,phi,t,k)
        print ("acc: %.4f, C: %.4f" %(acc, c))
        accuracies.append((acc,c))
    best_acc_c = max(accuracies, key = lambda x: x[0])
    return best_acc_c


def main():
    fname_x_train = 'dataset_5Genres_train.txt_matrix.npy'
    fname_t_train = 'tid_to_5Genres_train.txt_field_1s.npy'
    phi_train = np.load(fname_x_train)
    t_train = np.load(fname_t_train)
    
    fname_x_test = 'dataset_5Genres_test.txt_matrix.npy'
    fname_t_test = 'tid_to_5Genres_test.txt_field_1s.npy'
    phi_test = np.load(fname_x_test)
    t_test = np.load(fname_t_test)

    # fname_x_train = 'debug_x_train.dat'
    # fname_t_train = 'debug_t_train.dat'
    # phi_train = np.loadtxt(fname_x_train)
    # t_train = np.loadtxt(fname_t_train)
    
    # fname_x_test = 'debug_x_test.dat'
    # fname_t_test = 'debug_t_test.dat'
    # phi_test = np.loadtxt(fname_x_test)
    # t_test = np.loadtxt(fname_t_test)


    k = 5
    # best_acc, best_c, best_gamma = select_hyperparam_rbf(phi_train, t_train, k)
    # print ("best acc: %.4f, best C: %.4f, best_gamma: %.4f" %(best_acc, best_c, best_gamma))

    best_acc, best_c = select_hyperparam_linear(phi_train, t_train, k)
    print ("best acc: %.4f, best C: %.4f" %(best_acc, best_c))

    # now test the SVM
    clf = SVC(kernel='linear', C=best_c)
    clf.fit(phi_train,t_train)
    joblib.dump(clf, 'genre_fitted_svm.pkl')
    t_pred = clf.predict(phi_test)
    perf = metrics.accuracy_score(t_test, t_pred)
    print ("best perf: " + str(perf))

    
main()
