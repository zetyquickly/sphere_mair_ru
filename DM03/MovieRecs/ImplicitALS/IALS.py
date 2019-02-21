## -*- coding: utf-8 -*-

import numpy as np

class ImplicitALS:
    def __init__(self, max_epoch=10, embedding_size=15, alpha=10, l2reg=0.1, log_confidence=True, eps=0.1,
                 mean_decrease=0.85, use_test_to_init=True, random_state=42, show_every=1,
                 show_real_metric=False):
        '''
        :param max_epoch:       number of iterations
        :param embedding_size:  size of embedding vector for users and items
        :param alpha:           alpha in confidence level formula
        :param l2reg:           lambda in optimised metric
        :param log_confidence:  False: Cij = 1 + alpha * R
                                True: Cij = 1 + alpha * log(1 + R/eps)
        :param eps:             coefficient in log_confidence formula
        :param mean_decrease:   how much should we decrease init value for unlabeled pairs
        :param use_test_to_init if True, we use test matrix to calculate init values for unlabeled data
        :param random_state:    random state
        :param show_every:      show metrics each N steps, if == 0 - nothing printed
        :param show_real_metric: Shows real metric optimised by ALS(value must not increase)
        '''
        self.max_epoch = max_epoch
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.l2reg = l2reg
        self.random_state = random_state
        self.show_every = show_every
        self.show_real_metric = show_real_metric
        self.eps = eps
        self.log_confidence = log_confidence
        if self.log_confidence:
            self.confidence_func = self.__log_confidence
        else:
            self.confidence_func = self.__standard_confidence
        self.mean_decrease = mean_decrease
        self.use_test_to_init = use_test_to_init
        self.normalisation_func = self.__bias_normalisation
        np.random.seed(self.random_state)


    @staticmethod
    def __standard_confidence(R_train, R_test, alpha, eps):
        return np.ones(R_train.shape) + alpha * R_train


    @staticmethod
    def __log_confidence(R_train, R_test, alpha, eps):
        test_eq_mask = R_test == 0
        return (np.ones(R_train.shape) + alpha * np.log(np.ones(R_train.shape) + R_train / eps))


    def __bias_normalisation(self, R, R_test):
        gr_mask = R > 0
        eq_mask = R == 0
        mean = R[gr_mask].mean()
        R_unbiased = R * gr_mask - gr_mask * mean
        user_bias = (R_unbiased.sum(1) / gr_mask.sum(1)).reshape(-1, 1)
        R_unuserbiased = R_unbiased * gr_mask - gr_mask * user_bias
        gr_mask_sum_0 = gr_mask.sum(0)
        gr_mask_sum_0[gr_mask_sum_0 == 0] = 1
        item_bias = (R_unuserbiased.sum(0) / gr_mask_sum_0).reshape(1, -1)
        if self.use_test_to_init:
            gr_mask_test = R_test > 0
            P = R + eq_mask * (~gr_mask_test) * (user_bias + item_bias + mean) * self.mean_decrease + eq_mask * \
                gr_mask_test * (user_bias + item_bias + mean)
        else:
            P = R + eq_mask * (user_bias + item_bias + mean) * self.mean_decrease
        return P, mean, gr_mask, eq_mask


    def __mean_normalisation(self, R, R_test):
        gr_mask = R > 0
        eq_mask = R == 0
        mean = R[gr_mask].mean()
        P = R + eq_mask * mean * self.mean_decrease
        return P, mean, gr_mask, eq_mask


    def eval_metrics(self, epoch, result, R_train, R_test, X, Y, C, gamma, beta, gr_mask, show_every):
        str_out = "Epoch {}\ttrain\t".format(epoch)
        train_error = np.sqrt(((result * gr_mask - R_train) ** 2).sum() / gr_mask.sum())
        str_out += str(train_error)
        R = R_train.copy()
        if R_test is not None:
            test_error = np.sqrt(((result * (R_test > 0) - R_test) ** 2).sum() / (R_test > 0).sum())
            str_out += "\ttest\t{}".format(test_error)
            R += R_test
        else:
            test_error = None
        if self.show_real_metric:
            metric = 1.0/(R.shape[0] * R.shape[1]) * \
                     (C * (R - beta - gamma) ** 2).sum() + \
                     self.l2reg * ((X**2).sum() + (Y**2).sum() + (beta**2).sum() + (gamma**2).sum())
            str_out += "\tmetric\t{}".format(metric)
        if show_every > 0:
            print(str_out)
        return train_error, test_error

    def fit(self, R_train, R_test=None):
        P, mean, gr_mask, eq_mask = self.normalisation_func(R_train, R_test)
        # Precompute C and (C - 1) matrices
        C = self.confidence_func(R_train, R_test, self.alpha, self.eps)
        Cm1 = C - 1

        laI = np.eye(self.embedding_size + 1, self.embedding_size + 1) * self.l2reg  # fixed

        # Initialize embeddings
        X = np.hstack([np.ones((R_train.shape[0], 1)), np.random.random_sample((R_train.shape[0], self.embedding_size))])
        Y = np.hstack([np.ones((R_train.shape[1], 1)), np.random.random_sample((R_train.shape[1], self.embedding_size))])

        # Initialize biases
        beta = np.zeros((X.shape[0], 1))  # user bias
        gamma = np.zeros((X.shape[0], 1))  # item bias


        if self.use_test_to_init:
            n_user = (R_train + R_test > 0).sum(1)
            n_item = (R_train + R_test > 0).sum(0)
        else:
            n_user = (R_train > 0).sum(1)
            n_item = (R_train > 0).sum(0)

        for epoch in range(self.max_epoch):
            # User-step
            YtY = np.matmul(Y.T, Y)
            Pgamma = P - gamma
            Cp = C * Pgamma
            for i in range(X.shape[0]):  # user-loop
                to_inv = YtY + laI * (n_user[i]) + np.matmul(Y.T * Cm1[i, :], Y)
                inv_mat = np.linalg.inv(to_inv)
                inv_mat_to_y = np.matmul(inv_mat, Y.T)
                X[i, :] = np.matmul(inv_mat_to_y, Cp[i, :].reshape(-1, 1)).ravel()
            # Item-step
            beta = X[:, 0].copy().reshape(-1, 1)
            X[:, 0] = 1
            XtX = np.matmul(X.T, X)
            Pbeta = P - beta
            Cp = C * Pbeta
            for j in range(Y.shape[0]):  # item-loop
                to_inv = XtX + laI * (n_item[j]) + np.matmul(X.T * Cm1[:, j], X)
                inv_mat = np.linalg.inv(to_inv)
                inv_mat_to_x = np.matmul(inv_mat, X.T)
                Y[j, :] = np.matmul(inv_mat_to_x, Cp[:, j].reshape(-1, 1)).ravel()
            gamma = Y[:, 0].copy().reshape(1, -1)
            Y[:, 0] = 1
            epoch_result = np.matmul(X[:, 1:], Y[:, 1:].T) + beta + gamma
            epoch_result_fix = epoch_result.copy()
            epoch_result_fix[epoch_result_fix > 5] = 5
            epoch_result_fix[epoch_result_fix < 1] = 1
            if self.show_every > 0:
                if (epoch + 1) % self.show_every == 0:
                    self.eval_metrics(epoch, epoch_result_fix, R_train, R_test, X, Y, C,
                                      gamma, beta, gr_mask, self.show_every)
        train_error, test_error = self.eval_metrics(epoch, epoch_result_fix, R_train, R_test, X, Y, C,
                                      gamma, beta, gr_mask, self.show_every)

        return epoch_result_fix, train_error, test_error
