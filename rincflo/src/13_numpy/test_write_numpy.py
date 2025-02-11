import numpy as np
import pandas as pd

# Create DataFrame from CSV files
layout_df = pd.read_csv("csv/layout.csv")
idx_df = pd.read_csv("csv/idx.csv", dtype={'i':np.int32, 'j': np.int32, 'k': np.int32})
pos_df = pd.read_csv("csv/pos.csv", dtype=np.float64)
vel_df = pd.read_csv("csv/vel.csv", dtype=np.float64)

# Load numpy arrays from .npy files written
idx_np = np.load("numpy/idx.npy")
pos_np = np.load("numpy/pos.npy")
vel_np = np.load("numpy/vel.npy")

# Print Layout
print("layout:")
print(layout_df)

# Test index
i = 1
j = 2
# Output index
i_out = idx_np[i, j, 0]
j_out = idx_np[i, j, 1]
# Output position
x = pos_np[i, j, 0]
y = pos_np[i, j, 1]

# Compare index between CSV and numpy
print("********************************************************************************")
print("idx by source:")
print("DataFrame:")
print(idx_df)
# print(idx_df.dtypes)
print("\nFrom .npy file:")
print("Top 5x5 grid of i:")
print(idx_np[0:5,0:5,0])
print("Top 5x5 grid of j:")
print(idx_np[0:5,0:5,1])
print("(i,j) at selected test point:")
print(f"(i,j) at index [{i},{j}] = ({i_out}, {j_out}).")

# Compare position between CSV and numpy
print("********************************************************************************")
print("pos by source:")
print("DataFrame:")
print(pos_df)
print("from .npy file:")
print(f"shape = {pos_np.shape}.")
#print("Top 5x5 grid of x:")
#print(pos_np[0:5,0:5,0])
#print("Top 5x5 grid of y:")
#print(pos_np[0:5,0:5,1])
print(f"(x,y) at index [{i},{j}] = ({x:0.10f}, {y:0.10f}).")

# # Compare velocity between CSV and numpy
# print("********************************************************************************")

# *************************************************************************************************
# Report test results
is_ok_ij = (i==i_out) and (j==j_out)
dx = 100E-6/36.0
dy = 100E-6/36.0
x_ = (i+0.5)*dx
y_ = (j+0.5)*dy
tol = 1.0E-8
is_ok_xy = np.isclose(x, x_, tol) and np.isclose(y, y_, tol)
is_ok = is_ok_ij and is_ok_xy
msg = 'PASS' if is_ok else 'FAIL'
print("********************************************************************************")
print("Test results for WriteNumpy():")
print(f"*****{msg}*****")
