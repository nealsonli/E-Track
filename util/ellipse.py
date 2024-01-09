# Ben Hammel, & Nick Sullivan-Molina. (2020, March 21). 
# bdhammel/least-squares-ellipse-fitting: v2.0.0 (Version v2.0.0). 
# Zenodo. http://doi.org/10.5281/zenodo.3723294
import numpy as np
import numpy.linalg as la
from math import pi

__version__ = '2.0.0'


class LsqEllipse:
    """Lest Squares fitting of Elliptical data

    Attributes
    ----------
    coef_ : array
        Estimated coefficients for the Least squares fit to the elliptical data
        containing the values [a,b,c,d,f,g].T corresponding to
        ax**2 + 2bxy + cy**2 + 2dx + 2fy + g

    References
    ----------
    (*) Halir R., Flusser J. 'Numerically Stable Direct Least Squares
    Fitting of Ellipses'
    (**) Weisstein, Eric W. "Ellipse." From MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/Ellipse.html

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LsqEllipse
    >>> x = np.array([ 1.,  0., -1., -0.,  1.])
    >>> y = np.array([ 0. ,  0.5,  0. , -0.5, -0. ])
    >>> X = np.c_[x, y]
    >>> reg = LsqEllipse().fit(X)
    >>> reg.as_parameters()
    """
    ALLOWED_FEATURES = 2

    def __init__(self):
        self.coef_ = None

    def _check_data(self, X):

        n_samples, n_features = X.shape
        if not n_features == self.ALLOWED_FEATURES:
            raise ValueError("Incorrect number of features. "
                             f"Got {n_features} features, expected 2. ")

        if n_samples < 5:
            raise ValueError("Received too few samples"
                             f"Got {n_samples} features, 5 or more required. ")

        return X

    def fit(self, X):
        """Fit the data

        Parameters
        ----------
        X : array, shape (n_points, 2)
            Data values for the x-y data pairs to fit

        Returns
        -------
        self : returns an instance of self.
        """
        X = self._check_data(X)

        # extract x-y pairs
        x, y = X.T

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = np.vstack([x**2, x * y, y**2]).T
        # Linear part of design matrix [eqn. 16] from (*)
        D2 = np.vstack([x, y, np.ones_like(x)]).T

        # print(f'D1: {D1}')
        # print(f'D2: {D2}')

        # Forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # print(f'S1: {S1}')
        # print(f'S2: {S2}')
        # print(f'S3: {S3}')

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        # Reduced scatter matrix [eqn. 29]
        M = la.inv(C1) @ (S1 - S2 @ la.inv(S3) @ S2.T)

        # print(f'M: {M}')

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this
        # equation [eqn. 28]
        eigval, eigvec = np.linalg.eig(M)


        # print(f'eigval: {eigval}')
        # print(f'eigvec: {eigvec}')

        # Eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = (
            4*np.multiply(eigvec[0, :], eigvec[2, :])
            - np.power(eigvec[1, :], 2)
        )
        a1 = eigvec[:, np.nonzero(cond > 0)[0]]

        # |d f g> = -S3^(-1) * S2^(T)*|a b c> [eqn. 24]
        a2 = la.inv(-S3) @ S2.T @ a1

        # Eigenvectors |a b c d f g>
        # list of the coefficients describing an ellipse [a,b,c,d,f,g]
        # corresponding to ax**2 + 2bxy + cy**2 + 2dx + 2fy + g
        self.coef_ = np.vstack([a1, a2])

        return self

    @property
    def coefficients(self):
        """
        List of the coefficients describing the fitted ellipse

        Returns
        -------
        [a,b,c,d,f,g] corresponding to ax**2 + 2bxy + cy**2 + 2dx + 2fy + g
        """
        return np.asarray(self.coef_).ravel()

    def as_parameters(self):
        """Returns the definition of the fitted ellipse as localized parameters

        Returns
        _______
        center : list
            [x0, y0]
        width : float
            Semimajor axis
        height : float
            Semiminor axis
        phi : float
            The counterclockwise angle of rotation from the x-axis to the major
            axis of the ellipse
        """

        # Eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0
        # [eqn. 15) from (**) or (***)
        a = self.coefficients[0]
        b = self.coefficients[1] / 2.
        c = self.coefficients[2]
        d = self.coefficients[3] / 2.
        f = self.coefficients[4] / 2.
        g = self.coefficients[5]

        # print(f'a: {a:.3f}')
        # print(f'b: {b:.3f}')
        # print(f'c: {c:.3f}')
        # print(f'd: {d:.3f}')
        # print(f'f: {f:.3f}')
        # print(f'g: {g:.3f}')

        # Finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c*d - b*f) / (b**2. - a*c)
        y0 = (a*f - b*d) / (b**2. - a*c)
        center = [x0, y0]

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        denominator1 = (b * b - a * c) * (
            (c - a) * np.sqrt(1 + 4*b*b / ((a - c)*(a - c))) - (c + a)
        )
        denominator2 = (b*b - a*c) * (
            (a - c) * np.sqrt(1 + 4*b*b / ((a - c) * (a - c))) - (c + a)
        )
        temp_width = np.sqrt(numerator / denominator1)
        temp_height = np.sqrt(numerator / denominator2)

        # Angle of counterclockwise rotation of major-axis of ellipse to x-axis
        # [eqn. 23] from (**) or [eqn. 26] from (***).
        temp_phi = .5 * np.arctan((2.*b) / (a - c))

        # width, height, phi = temp_width, temp_height, temp_phi
        if temp_width >= temp_height:
            width, height, phi = temp_width, temp_height, temp_phi
        else:
            width, height = temp_height, temp_width
            if temp_phi >= 0:
                phi = temp_phi - pi / 2
            else:
                phi = temp_phi + pi / 2

        return center, width, height, phi

    def return_fit(self, n_points=None, t=None):
        """Return the X, Y values of the predicted ellipse

        Points are returned along the parametric curve of the ellipse as evenly
        spaced points starting at t=0 to t=2pi

        Parameters
        ---------
        n_points : int
            Number of points to return
        t : array
            Parametric points used to generate x-y pairs, If provided,
            `n_points` will be ignored

        Returns
        -------
        X : array, shape (n_points, 2)
            data values for the x-y data pairs
        """
        if self.coef_ is None:
            raise ValueError("Must call .fit() before using .return_fit()")

        if n_points is None and t is None:
            raise AttributeError("A value for `n_points` or `t` must be ",
                                 "provided")

        if t is None:
            t = np.linspace(0, 2 * np.pi, n_points)

        center, width, height, phi = self.as_parameters()

        x = (center[0]
             + width * np.cos(t) * np.cos(phi)
             - height * np.sin(t) * np.sin(phi))
        y = (center[1]
             + width * np.cos(t) * np.sin(phi)
             + height * np.sin(t) * np.cos(phi))

        return np.c_[x, y]
