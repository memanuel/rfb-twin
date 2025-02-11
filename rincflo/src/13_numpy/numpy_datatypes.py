import numpy as np

def print_data_type(s1: str, s2: str):
    """Print the name of a data type and its description in .npy file headers"""
    print(f'{s1:16} : {s2:16}')

def main():
    """Print out string descriptors for numpy datatypes"""
    
    # See https://numpy.org/doc/stable/reference/arrays.scalars.htm for list of scalar data types
    # See https://numpy.org/devdocs/reference/generated/numpy.lib.format.dtype_to_descr.html#numpy.lib.format.dtype_to_descr
    # for how to get the descriptor of a data type to use in the .npy file header.

    # Signed integer data types
    bool = np.dtype('bool')

    # Signed integer data types
    i8  = np.dtype('int8')
    i16 = np.dtype('int16')
    i32 = np.dtype('int32')
    i64 = np.dtype('int64')

    # Unsigned integer data types
    u8  = np.dtype('uint8')
    u16 = np.dtype('uint16')
    u32 = np.dtype('uint32')
    u64 = np.dtype('uint64')

    # Floating point data types
    f32 = np.dtype('float32')
    f64 = np.dtype('float64')
    f128 = np.dtype('float128')

    # Complex data types
    c64  = np.dtype('complex64')
    c128 = np.dtype('complex128')
    c256 = np.dtype('complex256')

    # Get the header for each type
    s1 = 'Data Type'
    s2 = 'Header'
    print_data_type(s1, s2)

    # Bool
    print_data_type('bool', np.lib.format.dtype_to_descr(bool))

    # Signed integers
    print_data_type('i8',  np.lib.format.dtype_to_descr(i8))
    print_data_type('i16', np.lib.format.dtype_to_descr(i16))
    print_data_type('i32', np.lib.format.dtype_to_descr(i32))
    print_data_type('i64', np.lib.format.dtype_to_descr(i64))

    # Unsigned integers
    print_data_type('u8',  np.lib.format.dtype_to_descr(u8))
    print_data_type('u16', np.lib.format.dtype_to_descr(u16))
    print_data_type('u32', np.lib.format.dtype_to_descr(u32))
    print_data_type('u64', np.lib.format.dtype_to_descr(u64))

    # Floating point
    print_data_type('f32',  np.lib.format.dtype_to_descr(f32))
    print_data_type('f64',  np.lib.format.dtype_to_descr(f64))
    print_data_type('f128', np.lib.format.dtype_to_descr(f128))

    # Complex
    print_data_type('c64',  np.lib.format.dtype_to_descr(c64))
    print_data_type('c128', np.lib.format.dtype_to_descr(c128))
    print_data_type('c256', np.lib.format.dtype_to_descr(c256))

# **********************************************************************************************************************
if __name__ == '__main__':
    main()
