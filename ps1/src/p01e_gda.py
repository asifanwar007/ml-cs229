import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=False)
    y_pred= clf.predict(x_val)
    x_val = util.add_intercept(x_val)
    print(clf.theta_0)
    # Use np.savetxt to save outputs from validation set to pred_path
    # print(clf.theta_0)
    util.plot(x_val, y_val, clf.theta_0, '{}.png'.format(pred_path))
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    # theta = 0.0
    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # Find phi, mu_0, mu_1, and sigma
        # print(m,n)
        phi = 1/m * y[y>0].sum()
        mu_0 = x[y==0].sum(axis=0)/np.count_nonzero(y==0) #np.count_nonzero is faster than below one
        mu_1 = x[y==1].sum(axis=0)/(y==1).sum()
        x1 = x.copy()
        x1[y==0]-=mu_0
        x1[y==1]-=mu_1
        sigma = 1/m * x1.T.dot(x1)
        # sigma = 1/m * np.multiply(x1, x1).sum(axis=1)
        # sigma = sigma.sum(axis=0)
        print(sigma.shape)
        print("------------,l;")
        # mu_0 = s_mu_0/mu_0_c
        # mu_1 = s_mu_1/mu_1_c
        print(mu_0, mu_1, sigma)
        sigma_inv = np.linalg.inv(sigma)
        print(sigma_inv)
        global theta
        # theta = 1/2 *(mu_0.T.dot(mu_0)/sigma - mu_1.T.dot(mu_1)/sigma) - np.log((1-phi)/phi)
        theta = 1/2 *(mu_0.T.dot(sigma_inv).dot(mu_0) - mu_1.T.dot(sigma_inv).dot(mu_1)) - np.log((1-phi)/phi)

        # self.theta_0 = - (1/sigma)*(mu_0 - mu_1)
        self.theta_0 = - sigma_inv.dot(mu_0 - mu_1)

        # print(mu_0, mu_1, sigma)
        # Write theta in terms of the parameters

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        pred = 1/(1+np.exp(-(x.dot(self.theta_0.T)+theta)))
        theta0 = np.array([theta])
        self.theta_0 = np.hstack([ theta0, self.theta_0])
        return pred
        # *** END CODE HERE