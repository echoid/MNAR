import numpy as np
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,f1_score



def imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Imputation error of missing data
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm

        if i % 100 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XM


def not_imputationRMSE(model, Xorg, Xz, X, S, L):
    """
    Imputation error of missing data, using the not-MIWAE
    """
    N = len(X)

    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / e_x.sum(axis=1)[:, None]

    def imp(model, xz, s, L):
        l_out, log_p_x_given_z, log_p_z, log_q_z_given_x, log_p_s_given_x  = model.sess.run(
            [model.l_out_mu, model.log_p_x_given_z, model.log_p_z, model.log_q_z_given_x, model.log_p_s_given_x],
            {model.x_pl: xz, model.s_pl: s, model.n_pl: L})

        wl = softmax(log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x)

        xm = np.sum((l_out.T * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)

        return l_out, wl, xm, xmix

    XM = np.zeros_like(Xorg)

    for i in range(N):

        xz = Xz[i, :][None, :]
        s = S[i, :][None, :]

        l_out, wl, xm, xmix = imp(model, xz, s, L)

        XM[i, :] = xm

        if i % 100 == 0:
            print('{0} / {1}'.format(i, N))

    return np.sqrt(np.sum((Xorg - XM) ** 2 * (1 - S)) / np.sum(1 - S)), XM


def downstream(Xtrain, Ytrain, Xtest,Ytest, dataset):
    result = []

    if dataset == "concrete":
        modellist = [svm.SVR(),tree.DecisionTreeRegressor(),linear_model.Lasso(alpha=0.1)]
        for regr in modellist:
            regr.fit(Xtrain, Ytrain)
            result.append(mean_squared_error(Ytest, regr.predict(Xtest)))
    else:
        modellist = [tree.DecisionTreeClassifier(),SGDClassifier(),svm.SVC()]
        for clf in modellist:
            clf.fit(Xtrain, Ytrain)
            result.append(f1_score(Ytest, clf.predict(Xtest), average='weighted'))

    return result