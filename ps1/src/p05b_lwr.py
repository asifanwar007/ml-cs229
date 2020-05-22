import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    # print(x_train[0], y_train.shape)
    clf = LocallyWeightedLinearRegression(0.05)
    clf.fit(x_train, y_train)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    pred=clf.predict(x_eval)
    # print(len(pred))
    # Get MSE value on the validation set
    mse = (pred-y_eval)
    mse = mse.T.dot(mse)/mse.shape
    print(mse)
    # Plot validation predictions on top of training set
    plt.scatter(x_train[:, 1], y_train, label='training')
    plt.scatter(x_eval[:,1], pred, label='predicted')
    plt.legend()
    plt.show()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # print(x[:,1])
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        w = lambda xi, xii:np.diag(np.exp(-((xi-xii)**2)/(2*(self.tau**2)))) 
        pred = []
        for i in range(m):
            W = w(self.x[:, 1], x[i][1])
            # print(np.diag(W).shape)
            t_inv = np.linalg.inv(self.x.T.dot(W).dot(self.x))
            theta = t_inv.dot(self.x.T).dot(W).dot(self.y)
            p = theta.dot(x[i])
            pred.append(p)
            # l.append(wi)
        # W = np.array(l)
        # pred = np.array([ele for ele in pred])
        pred = np.array(pred)
        return pred
        # *** END CODE HERE ***
