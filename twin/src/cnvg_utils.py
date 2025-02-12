from pathlib import Path
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from numpy.linalg import lstsq
from scipy.optimize import minimize
from scipy.special import expit, logit

# Define the types of the data
NumpyFloatArray = npt.NDArray[np.float64]
NumpyInt16Array = npt.NDArray[np.int16]

# *************************************************************************************************
@dataclass
class SimOutput:
    """Numpy output of one simulation"""
    # The shape of the data grid on this level; (nx, ny, nz)
    grid_shape: tuple[int, int, int]
    # Number of levels of mesh refinement; shape = (nx, ny, nz)
    refinement: NumpyInt16Array
    # The type of each cell; shape = (nx, ny, nz)
    cell_type: NumpyFloatArray
    # The velocity field; shape = (nx, ny, nz, 3)
    velocity: NumpyFloatArray
    # The pressure at cell centers; shape = (nx, ny, nz)
    pressure: NumpyFloatArray
    # The state of charge at cell centers; shape = (nx, ny, nz)
    soc: NumpyFloatArray

# *************************************************************************************************
# Output directory for plots in this study
plot_dir = Path('figs', '01_cnvg')

# *************************************************************************************************
# Dictionary with key = geometry number and value = geometry name
geom_name_tbl: dict[int, str] = {
    1: 'One Wire',
    2: 'Two Wires',
    3: 'Lattice',
}

# *************************************************************************************************
# Dictionary with subdirectory for geometry; key = geometry number, value = subdirectory
subdir_tbl: dict[int, str] = {
    1: '01_OneWire',
    2: '02_TwoWire',
    3: '03_Lattice',
}

# *************************************************************************************************
def fit_error_linear(h: NumpyFloatArray, err: NumpyFloatArray) -> tuple[float, float]:
    """
    Fit a line to the error data of the form
    err(h) = C * h^p
    INPUTS:
        h: step size (1/nz); dimensionless
        err: L2 norm of velocity difference; dimensionless
    RETURNS:
        C: Coefficient of the fitted line
        p: Exponent of the fitted line
    """
    # Number of data points
    m: int = h.shape[0]
    # Assemble the design matrix x
    x: NumpyFloatArray = np.ones((m,2))
    x[:, 1] = np.log(h)
    # The right hand side is the log of the error
    y = np.log(err)

    # Solve least squares problem and extract the coefficients into beta
    res = lstsq(x, y, rcond=None)
    beta = res[0]
    C: float = np.exp(beta[0])
    p: float = float(beta[1])
    return C, p

# *************************************************************************************************
def fit_error_nonlinear(h: NumpyFloatArray, err: NumpyFloatArray, h_ref: float) -> tuple[float, float, float, float]:
    """
    Fit a nonlinear model to the error data of the form
    err(h) = C * h^p * (1 - (h_ref/h)^p)
    INPUTS:
        h: step size (1/nz); dimensionless
        err: L2 norm of velocity difference; dimensionless
    RETURNS:
        C: Coefficient of the fitted line
        p: Exponent of the fitted line
        alpha: Coefficient of the mesh refinement term
        rmse: Root mean square error of the fit in log units
    """
    # Ratio r of reference step size to step size
    r: NumpyFloatArray = h_ref / h
    # The log error
    log_err: NumpyFloatArray = np.log(err)

    # Objective function for nonlinear optimization
    def fun(x0: NumpyFloatArray) -> NumpyFloatArray:
        # Unpack C
        C: float = np.exp(x0[0])
        # Unpack p 
        p: float = np.exp(x0[1])
        # Unpack alpha
        alpha: float = expit(x0[2])
        # Predicted error
        err_pred: NumpyFloatArray = C * np.power(h, p) * (1.0 - alpha * np.power(r, p))
        log_err_pred: NumpyFloatArray = np.log(err_pred)
        # The cost function is the RMS difference between the predicted and actual error on a log scale
        return np.sqrt(np.mean(np.square(log_err_pred - log_err)))
    
    # Initial guess for the parameters
    C: float
    p: float
    C, p = fit_error_linear(h=h, err=err)
    C = max(C, 1.0E-6)
    p = max(p, 1.0E-6)
    alpha: float = 0.5
    x0: NumpyFloatArray = np.array([np.log(C), np.log(p), logit(alpha)])

    # Minimize the objective function
    res = minimize(fun, x0, method='Nelder-Mead')
    C = np.exp(res.x[0])
    p = np.exp(res.x[1])
    alpha = expit(res.x[2])
    # Calculate root mean squared error
    rmse: float = res.fun
    # Return the optimized parameters
    return (C, p, alpha, rmse)

# *************************************************************************************************
def fit_error_mesh(h: NumpyFloatArray, level: NumpyInt16Array, err: NumpyFloatArray, h_ref: float) \
        -> tuple[float, float, float, float, float]:
    """
    Fit a nonlinear model to the error data of the form
    err(h) = (C_c * h_c^p * + C_f * h_f^p) * (1 - (h_ref/h_c)^p)
    h_c and h_f are the step sizes on the coarse and fine levels, respectively
    INPUTS:
        h: step size (1/nz) at the coarse level; dimensionless
        level: level of mesh refinement; dimensionless
        err: L2 norm of velocity difference; dimensionless
        h_ref: step size (1/nz) at the reference level; dimension
    RETURNS:
        C_c: Prefactor - coarse
        C_f: Prefactor - fine
        p:  Exponent
        alpha: Mesh refinement coefficient
        rmse: Root mean square error of the fit in log units
    """
    # The coarse step size h_c is the same as h
    h_c: NumpyFloatArray = h
    # The fine step size h_f
    h_f: NumpyFloatArray = h_c / np.power(2.0, level)
    # Refinement ratio; measured vs. the coarse resolution
    rr: NumpyFloatArray = h_ref / h_c
    # The log error
    log_err: NumpyFloatArray = np.log(err)

    # Allowed range for C
    C_min: float = 1.0E-6
    C_max: float = 1.0E+6
    # Allowed range for p
    p_min: float = 1.0E-2
    p_max: float = 3.0

    # Allowed range for log(C)
    log_C_min: float = np.log(C_min)
    log_C_max: float = np.log(C_max)
    # Allowed range for log(p)
    log_p_min: float = np.log(p_min)
    log_p_max: float = np.log(p_max)

    # Objective function for nonlinear optimization
    def fun(x0: NumpyFloatArray) -> NumpyFloatArray:
        # Unpack C for coarse and fine levels
        C_c: float = np.exp(np.clip(x0[0], log_C_min, log_C_max))
        C_f: float = np.exp(np.clip(x0[1], log_C_min, log_C_max))
        # Unpack p
        p: float = np.exp(np.clip(x0[2], log_p_min, log_p_max))
        # Unpack alpha
        alpha: float = expit(x0[3])
        # Predicted error
        err_pred: NumpyFloatArray = \
            (C_c * np.power(h_c, p) + C_f * np.power(h_f, p)) * (1.0 - alpha * np.power(rr, p))
        # Log of the predicted error
        log_err_pred: NumpyFloatArray = np.log(err_pred)
        # The cost function is the RMS difference between the predicted and actual error on a log scale
        return np.sqrt(np.mean(np.square(log_err_pred - log_err)))
    
    # Initial guess for the parameters - linear model on finest level
    C_lin: float
    p_lin: float
    C_lin, p_lin = fit_error_linear(h=h_f, err=err)
    # Set minimium positive values for C and p
    C_lin = np.clip(C_lin, C_min, C_max)
    p_lin = np.clip(p_lin, p_min, p_max)
    # Assign the initial guess for C_c and C_f assuming half the error is on each level
    C_c: float = C_lin / 2.0
    C_f: float = C_lin / 2.0
    # Assign the initial guess for p
    p: float = p_lin
    # Assign a neutral initial guess that alpha is 0.5
    alpha: float = 0.5
    # Wrap the initial guesses into a numpy array
    x0: NumpyFloatArray = np.array([np.log(C_c), np.log(C_f), np.log(p), logit(alpha)])

    # Minimize the objective function
    method: str = 'BFGS'
    tol: float = 1.0E-9
    options = {'maxiter': 10000}
    res = minimize(fun=fun, x0=x0, method=method, tol=tol, options=options)
    # Unpack the optimized parameters
    C_c = np.exp(res.x[0])
    C_f = np.exp(res.x[1])
    p = np.exp(res.x[2])
    alpha = expit(res.x[3])
    # Calculate root mean squared error
    rmse: float = res.fun

   # Return the optimized parameters
    return (C_c, C_f, p, alpha, rmse)
