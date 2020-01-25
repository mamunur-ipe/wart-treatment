"""
@author: Mamunur Rahman
"""

### create a function for F-score feature selection
def calculate_F_score (X, y):
    """
    Parameters
    ----------
        X : feature matrix (numpy 2D array)
        y : target binary class (numpy 1D array)
    Output
    ---------
        Returns a list containing F-score value of the features
    -------------------------------------------------------------------------------------------
    Original Article:
    Chen, Y. W., & Lin, C. J. (2006). Combining SVMs with various feature selection strategies.
    In Feature extraction (pp. 315-324). Springer, Berlin, Heidelberg.
    """

    import numpy as np
    #find the unique values of target classes
    unique = np.unique(y)  
    if unique.shape[0] != 2:
        print("Error: The target class is not binary")
        
    F_score = []
    # From the feature matrix, take one column at a time and calculate F-score
    for i in range(np.shape(X)[1]):
        x = X[:, i]
        #numerator
        x_bar = np.mean(x)
        x_0_bar = np.mean(x[y == unique[0]])
        x_1_bar = np.mean(x[y == unique[1]])

        numerator = (x_0_bar - x_bar)**2 + (x_1_bar - x_bar)**2

        #denominator
        x_0 = x[y == unique[0]]
        x_1 = x[y == unique[1]]

        denominator = np.var(x_0, ddof=1) + np.var(x_1, ddof=1)

        # F-score
        f = (numerator/denominator)
        F_score.append(np.round(f, 4))  # round upto four decimal points

    return F_score

