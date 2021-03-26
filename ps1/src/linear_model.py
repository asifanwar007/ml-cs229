import numpy as np
import util
import matplotlib.pyplot as plt

class LinearModel(object):
    """Base class for linear models."""
    # theta0, theta1, theta2 = 0,0,0
    # delta = 1e-5
    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta_0 = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run solver to fit linear model.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # dl = 1
        # while(dl > self.delta):
        #     jt = 0;
        #     for i in range(800):
        #         jt += (y[i]-self.theta0)*x[i][0]
        #     tmp = self.theta0 + self.step_size*jt
        #     dl = abs(self.theta0-tmp)
        #     self.theta0 = tmp
        # print(self.theta0)
        # dl = 1
        # while(dl > self.delta):
        #     jt = 0;
        #     for i in range(800):
        #         jt += (y[i]- self.theta0 - self.theta1*x[i][1])*x[i][1]
        #     tmp = self.theta1 + self.step_size*jt
        #     dl = abs(self.theta1-tmp)
        #     self.theta1 = tmp
        # print(self.theta1)
        # dl = 1
        # while(dl > self.delta):
        #     jt = 0;
        #     for i in range(800):
        #         jt += (y[i]- self.theta0- self.theta1*x[i][1]-self.theta2*x[i][2])*x[i][2]
        #     tmp = self.theta2 + self.step_size*jt
        #     dl = abs(self.theta2-tmp)
        #     self.theta2 = tmp
        # print(self.theta2)

        
        # for i in range(800):
        #     # self.theta1 = self.theta1 + self.step_size*(y[i]- self.theta0 - self.theta1*x[i][1])*x[i][1]
        # for i in range(800):
        #     self.theta2 = self.theta2 + self.step_size*(y[i]- self.theta0- self.theta1*x[i][1]-self.theta2*x[i][2])*x[i][2]

        # print(self.theta1, self.theta2, self.theta0)
        raise NotImplementedError('Subclass of LinearModel must implement fit method.')

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """

        raise NotImplementedError('Subclass of LinearModel must implement predict method.')

# def main(train_path='../data/ds1_train.csv',
#         eval_path='../data/ds1_valid.csv',
#         pred_path='output/linear_regression_1.txt'):
#     """Problem 1(b): Logistic regression with Newton's Method.

#     Args:
#         train_path: Path to CSV file containing dataset for training.
#         eval_path: Path to CSV file containing dataset for evaluation.
#         pred_path: Path to save predictions.
#     """
#     x_train, y_train = util.load_dataset(train_path, add_intercept=True)
#     # print(x_train.shape)
#     for i in range(800):
#         if y_train[i] == 0:
#             plt.scatter(x_train[i][1], x_train[i][2],color='c')
#         else:
#             plt.scatter(x_train[i][1], x_train[i][2],color='k')

#         plt.pause(0.05)
#         # if x_train[i][0] == 0:
#         #     print(x_train[i][0]);
#     # for i in range(800):
#     #     plt.scatter(x_train[i][2], y_train[i], color='c')
#     #     plt.pause(0.05)

#     # plt.plot(x_train[0], y_train)
#     # plt.show();
#     # *** START CODE HERE ***
#     l = LinearModel(step_size=0.00001, theta_0 = 0, )
#     l.fit(x_train, y_train)
#     plt.show()
#     # Train a logistic regression classifier
#     # Plot decision boundary on top of validation set set
#     # Use np.savetxt to save predictions on eval set to pred_path
#     # *** END CODE HERE ***
# if __name__ == '__main__':
#     main()
