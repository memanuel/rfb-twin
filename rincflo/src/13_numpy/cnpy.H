//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php
#pragma once

#include <string>
#include <cstring>
#include <complex>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <stdexcept>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <typeinfo>
#include <cassert>
#include <numeric>
#include <regex>
#include <cstdint>
#include <zlib.h>

using std::size_t, std::string, std::complex, std::vector, std::shared_ptr, std::map, std::ofstream, std::to_string,
std::regex, std::smatch, std::stoi, std::runtime_error, std::accumulate, std::multiplies, std::type_info;

// ************************************************************************************************
namespace cnpy {

// *********************************************************************************************************************
// Functions used in implementation of numpy capabilities
// These need to be declared before they are used in gcc-12 (gcc-11 permitted this without complaint)

// Map from type_info to string describing that type, e.g. typeid(int) -> "i"
string map_type(const type_info& t);

// Test if system is little-endian or big-endian
char BigEndianTest();

// Create the header to use when writing this file to disk
template<typename T>
vector<char> create_npy_header(const vector<size_t>& shape);

// Parse a numpy file header given a file pointer
void parse_npy_header(FILE* fp,size_t& word_size, vector<size_t>& shape, bool& fortran_order);

// Parse a numpy file header given a character buffer
void parse_npy_header(unsigned char* buffer, size_t& word_size, vector<size_t>& shape, bool& fortran_order);

// Parse the footer to a numpy z-lib archive (.npz file)
void parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset);

// Append bytes to the back of a character vector
template<typename T>
vector<char>& operator+=(vector<char>& lhs, const T rhs) 
{
    //write in little endian
    for(size_t byte = 0; byte < sizeof(T); byte++) 
    {
        char val = *((char*)&rhs+byte); 
        lhs.push_back(val);
    }
    return lhs;
}

// Append a string to the back of a character vector
template<>
vector<char>& operator+=(vector<char>& lhs, const string rhs);

// Append a character buffer to the back of a character vector
template<>
vector<char>& operator+=(vector<char>& lhs, const char* rhs);

// Create the header to a numpy file
template<typename T>
vector<char> create_npy_header(const vector<size_t>& shape) 
{
    // The body of the header is a string encoding of a dictionary
    vector<char> dict;
    dict += "{'descr': '";
    dict += BigEndianTest();
    dict += map_type(typeid(T));
    dict += to_string(sizeof(T));
    dict += "', 'fortran_order': False, 'shape': (";
    dict += to_string(shape[0]);
    for(size_t i = 1;i < shape.size();i++) 
    {
        dict += ", ";
        dict += to_string(shape[i]);
    }
    if(shape.size() == 1) 
        {dict += ",";}
    dict += "), }";
    //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
    int remainder = 16 - (10 + dict.size()) % 16;
    dict.insert(dict.end(),remainder,' ');
    dict.back() = '\n';

    // The header itself is a short preamble followed by the dictionary
    vector<char> header;
    header += (char) 0x93;
    header += "NUMPY";
    header += (char) 0x01; //major version of numpy format
    header += (char) 0x00; //minor version of numpy format
    // Insert the dictionary at the end of the header after the preamble
    header += (uint16_t) dict.size();
    header.insert(header.end(),dict.begin(),dict.end());

    return header;
}

// ************************************************************************************************
// Save an array of data in numpy format to a filestream
template<typename T> 
void npy_save(ofstream& fs, const T* data, const vector<size_t>& shape) 
{
    // Build the Numpy head and compute number of elements
    vector<char> header = create_npy_header<T>(shape);
    size_t n_elt = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
    size_t n_bytes = n_elt * sizeof(T);

    // Open the file and write out the header and data
    fs.write(&header[0], header.size());
    fs.write(reinterpret_cast<const char*>(data), n_bytes);
    fs.flush();
}

// ************************************************************************************************
// Save an array of data in numpy format to a named file
// Revised MSE version - uses C++ ofstream instead of fwrite
// Motivation: avoid data race.  WriteCSV() using C++ ofstream seems to work, while fwrite has problem.
template<typename T> 
void npy_save(string fname, const T* data, const vector<size_t>& shape) 
{
    // Open the file and write out the header and data
    ofstream fs(fname.c_str(), ofstream::binary);

    // Delegate to fstream version
    npy_save(fs, data, shape);
}

// ************************************************************************************************
template<typename T> 
void npz_save(string zipname, string fname, const T* data, const vector<size_t>& shape, string mode = "w")
{
    //first, append a .npy to the fname
    fname += ".npy";

    //now, on with the show
    FILE* fp = NULL;
    uint16_t nrecs = 0;
    size_t global_header_offset = 0;
    vector<char> global_header;

    if(mode == "a") 
        {fp = fopen(zipname.c_str(),"r+b");}

    if(fp) 
    {
        //zip file exists. we need to add a new npy file to it.
        //first read the footer. this gives us the offset and size of the global header
        //then read and store the global header.
        //below, we will write the the new data at the start of the global header then append the global header and footer below it
        size_t global_header_size;
        parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
        fseek(fp,global_header_offset,SEEK_SET);
        global_header.resize(global_header_size);
        size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
        if(res != global_header_size)
        {
            throw runtime_error("npz_save: header read error while adding to existing zip");
        }
        fseek(fp,global_header_offset,SEEK_SET);
    }
    else 
    {
        fp = fopen(zipname.c_str(),"wb");
    }

    vector<char> npy_header = create_npy_header<T>(shape);

    size_t nels = accumulate(shape.begin(),shape.end(),1,multiplies<size_t>());
    size_t nbytes = nels*sizeof(T) + npy_header.size();

    //get the CRC of the data to be added
    uint32_t crc = crc32(0L,(uint8_t*)&npy_header[0],npy_header.size());
    crc = crc32(crc,(uint8_t*)data,nels*sizeof(T));

    //build the local header
    vector<char> local_header;
    local_header += "PK"; //first part of sig
    local_header += (uint16_t) 0x0403; //second part of sig
    local_header += (uint16_t) 20; //min version to extract
    local_header += (uint16_t) 0; //general purpose bit flag
    local_header += (uint16_t) 0; //compression method
    local_header += (uint16_t) 0; //file last mod time
    local_header += (uint16_t) 0; //file last mod date
    local_header += (uint32_t) crc; //crc
    local_header += (uint32_t) nbytes; //compressed size
    local_header += (uint32_t) nbytes; //uncompressed size
    local_header += (uint16_t) fname.size(); //fname length
    local_header += (uint16_t) 0; //extra field length
    local_header += fname;

    //build global header
    global_header += "PK"; //first part of sig
    global_header += (uint16_t) 0x0201; //second part of sig
    global_header += (uint16_t) 20; //version made by
    global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
    global_header += (uint16_t) 0; //file comment length
    global_header += (uint16_t) 0; //disk number where file starts
    global_header += (uint16_t) 0; //internal file attributes
    global_header += (uint32_t) 0; //external file attributes
    global_header += (uint32_t) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
    global_header += fname;

    //build footer
    vector<char> footer;
    footer += "PK"; //first part of sig
    footer += (uint16_t) 0x0605; //second part of sig
    footer += (uint16_t) 0; //number of this disk
    footer += (uint16_t) 0; //disk where footer starts
    footer += (uint16_t) (nrecs+1); //number of records on this disk
    footer += (uint16_t) (nrecs+1); //total number of records
    footer += (uint32_t) global_header.size(); //nbytes of global headers
    footer += (uint32_t) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
    footer += (uint16_t) 0; //zip file comment length

    //write everything
    fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
    fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
    fwrite(data,sizeof(T),nels,fp);
    fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
    fwrite(&footer[0],sizeof(char),footer.size(),fp);
    fclose(fp);
}

// ************************************************************************************************
struct NpyArray 
{
    // Explicit constructor
    NpyArray(const vector<size_t>& _shape, size_t _word_size, bool _fortran_order) :
        shape(_shape), word_size(_word_size), fortran_order(_fortran_order)
    {
        // Accumulate the number of values by multiplying out the sizes along each dimension
        // num_vals = 1;
        // for(size_t i = 0; i < shape.size(); i++) 
        //     {num_vals *= shape[i];}
        num_vals = accumulate(shape.begin(), shape.end(), 1, multiplies<size_t>());
        // Initialize the data_holder with the required memory
        data_holder = shared_ptr<vector<char>>(new vector<char>(num_vals * word_size));
    }

    // Build an empty numpy array
    NpyArray() : 
        shape(0), word_size(0), fortran_order(0), num_vals(0) 
    { }

    // Modifiable pointer to the data
    template<typename T>
    T* data() 
    {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }

    // Constant pointer to the data
    template<typename T>
    const T* const_data() const 
    {
        return reinterpret_cast<const T*>(&(*data_holder)[0]);
    }

    // Vector wrapping the constant data
    template<typename T>
    vector<T> as_vec() const 
    {
        const T* p = const_data<T>();
        return vector<T>(p, p+num_vals);
    }

    // Return a copy of the scalar data
    template<typename T>
    T as_scalar() const
    {
        const T* p = const_data<T>();
        return p[0];
    }

    // Number of bytes
    size_t num_bytes() const 
    {
        return data_holder->size();
    }

    // Save this numpy array to disk
    template<typename T>
    void save(string fname)
    {
        const T* data_ptr = const_data<T>();
        npy_save<T>(fname, data_ptr, shape);
    }

    // Set the array to a constant
    template<typename T>
    void set_value(T x)
    {
        T* data_array = data<T>();
        for (int i=0; i < num_vals; ++i)
            {data_array[i] = x;}
    }

    // Multiply all the data by a constant
    template<typename T>
    void rescale(T scale_factor)
    {
        T* data_array = data<T>();
        for (int i=0; i < num_vals; ++i)
            {data_array[i] *= scale_factor;}
    }

    // Take the minimum of all the data with a constant
    template<typename T>
    void minimum(T x)
    {
        T* data_array = data<T>();
        for (int i=0; i < num_vals; ++i)
            {data_array[i] = std::min(data_array[i], x);}
    }

    // Take the maximum of all the data with a constant
    template<typename T>
    void maximum(T x)
    {
        T* data_array = data<T>();
        for (int i=0; i < num_vals; ++i)
            {data_array[i] = std::max(data_array[i], x);}
    }

    // Members

    // The data stored by the numpy array as a vector of bytes
    shared_ptr<vector<char>> data_holder;

    // The shape of the numpy array
    vector<size_t> shape;

    // The size of one entry in the array in bytes
    size_t word_size;

    // Is the array stored in fortran (column major) order? Default is false for C style (row major) ordering.
    bool fortran_order;

    // How many data elements are in the array? Cumulative produce of size of each axis
    size_t num_vals;
};

// *********************************************************************************************************************
// Factory to build a numpy array with a known type and the given size
template<typename T>
inline NpyArray make_npy_array(const vector<size_t>& shape)
{
    return NpyArray(shape, sizeof(T), false);
}

// *********************************************************************************************************************
// Basic API for working with numpy files in .npy and .npz format

// An npz data type is a map where the key is a variable name and the value is a NpyArray object
using npz_t = map<string, NpyArray>; 

// Load a numpy file and return a NpyArray object
NpyArray npy_load(string fname);

// Load a numpy z-lib archive (.npz file) and return an npz data map
npz_t npz_load(string fname);

// Load a numpy array from a numpy z-lib archive and the name of the variable
NpyArray npz_load(string fname, string varname);

}   // namespace cnpy
