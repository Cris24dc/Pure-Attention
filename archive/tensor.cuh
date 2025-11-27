#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <stdexcept>
#include <iostream>
#include <vector>
#include <cuda_runtime.h> 
#include <cuda_fp16.h>
#include <memory>

using namespace std;

typedef __half float16_t;
typedef float  float32_t;
typedef double float64_t;

struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
            // cout<<"S a eliberat\n";
        }
    }
};

template<typename T>
class Tensor {
private:
    vector<int32_t> shape;
    vector<int32_t> strides;
    u_int32_t rank;
    size_t size;
    unique_ptr<T,CudaDeleter> data_ptr; 


public:
    using value_type = T;
    Tensor(const vector<int>& shape);

    T* data();
    const T* data() const;
    size_t get_size() const;
    const vector<int>& get_shape() const;
    void from_cpu(const std::vector<T>& host_data);
    vector<T> to_cpu()const;
};


template<typename T> 
Tensor<T>::Tensor(const std::vector<int>& shape) : shape(shape) {
    this->rank = this->shape.size();

    this->size=1;
    for (int dim : this->shape)
        this->size*=dim;

    cout<<this->size*sizeof(T)<<'\n';

    this->strides.resize(this->rank);

    int stride_value = 1;
    for (int i=this->rank-1;i>=0; i-=1) {
        this->strides[i] = stride_value;
        stride_value*=this->shape[i];
    }

    T* raw_ptr_data=nullptr;
    size_t bytes=this->size*sizeof(T);

    cudaError_t err=cudaMalloc((void**)&raw_ptr_data,bytes);

    if (err != cudaSuccess) {
        string error_msg = "CUDA Error: " + string(cudaGetErrorString(err));
        throw runtime_error(error_msg);
    }

    this->data_ptr.reset(raw_ptr_data);
}

template<typename T>
T* Tensor<T>::data() { return data_ptr.get(); }

template<typename T>
const T* Tensor<T>::data() const { return data_ptr.get(); }

template<typename T>
size_t Tensor<T>::get_size() const { return this->size; }

template<typename T>
const vector<int>& Tensor<T>::get_shape() const { 
    return this->shape; 
}

template<typename T>
void Tensor<T>::from_cpu(const std::vector<T>& host_data) {
    if (host_data.size() != this->size) {
        throw std::length_error("Size mismatch: vectorul CPU are alta marime decat Tensorul GPU.");
    }
    cudaError_t err = cudaMemcpy(data_ptr.get(), host_data.data(), this->size * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw runtime_error("Eroare la from_cpu (Memcpy H2D)");
}

template<typename T>
vector<T> Tensor<T>::to_cpu() const {
    std::vector<T> host_result(this->size);
    cudaError_t err = cudaMemcpy(host_result.data(), data_ptr.get(), this->size * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw runtime_error("Eroare la to_cpu (Memcpy D2H)");
    return host_result;
}

#endif