import numpy as np
import util
import matplotlib.pyplot as plt
import time
# %matplotlib inline

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # print(x_train.shape)
    # for i in range(800):
    #     if x_train[i][0] == 0:
    #         print(x_train[i][0]);

    # plt.plot(x_train[0], y_train)
    # plt.show();
    # *** START CODE HERE ***
    # Train a logistic regression classifier
    l = LogisticRegression()
    l.fit(x_train, y_train)
    # Plot decision boundary on top of validation set set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = l.predict(x_val)
    print(l.theta_0)
    util.plot(x_val, y_val, l.theta_0, '{}.png'.format(pred_path))
    # Use np.savetxt to save predictions on eval set to pred_path
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # print(m)
        g = lambda x: 1/(1+np.exp(-x))

        #initialize thehta
        if self.theta_0 == None:
            self.theta_0 = np.zeros(n)

        # print(self.theta_0.dot(self.theta_0))
        # time.sleep(10)

        for i in range(self.max_iter):
            theta = self.theta_0
            g_x = g(x.dot(self.theta_0))
            # print(g_x)
            # time.sleep(4)
            g_x1 = g(1-x.dot(self.theta_0))
            # print(g_x1.shape)
            # time.sleep(6)
            # print((1-x.dot(self.theta_0)).shape)
            # time.sleep(10)
            # print((g_x.dot(g_x1)).shape)
            # time.sleep(10)
            ltheta = - 1/m*(np.transpose(x)).dot(y-g_x)
            
            H = 1/m*g_x.dot(g_x1)*(x.T.dot(x))
            H_inv = np.linalg.inv(H)

            #update thehta
            self.theta_0 = theta - H_inv.dot(ltheta)
            # print(g_x)
            # print(g_x1)
            # print(i)
            print("\nPrinting Theta\n")
            print(self.theta_0- theta)
            print()
            print(np.linalg.norm(self.theta_0- theta))
            time.sleep(1)
            if np.linalg.norm(self.theta_0 - theta, ord=1) < self.eps:
                break

        # print(self.theta_0[0])
        # print(self.theta_0[1])
        # print(self.theta_0[2])

                
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        g = lambda x: 1/(1+np.exp(-x))
        preds = g(x.dot(self.theta_0))
        return preds
        # *** END CODE HERE ***
