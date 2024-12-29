#include <cudnn.h>
#include <iostream>
#include <vector>

#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    std::cout << "cuDNN initialized successfully!" << std::endl;

    // Tensor dimensions: batch = 1, channels = 1, height = 5, width = 5
    int batch_size = 1, channels = 1, height = 5, width = 5;

    // Create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        batch_size, channels, height, width));

    // Create output tensor (after convolution)
    cudnnTensorDescriptor_t output_descriptor;
    int output_height = 3, output_width = 3; // Assuming valid padding
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        output_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        batch_size, channels, output_height, output_width));

    // Create convolution descriptor
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        filter_descriptor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC,
        channels, channels, 3, 3)); // Filter size = 3x3

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor,
        0, 0,  // Padding
        1, 1,  // Stride
        1, 1,  // Dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // Allocate memory for input, filter, and output
    size_t input_size = batch_size * channels * height * width * sizeof(float);
    size_t filter_size = channels * channels * 3 * 3 * sizeof(float);
    size_t output_size = batch_size * channels * output_height * output_width * sizeof(float);

    float* d_input;
    float* d_filter;
    float* d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_filter, filter_size);
    cudaMalloc(&d_output, output_size);

    // Initialize input and filter values
    std::vector<float> input(batch_size * channels * height * width, 1.0f);
    std::vector<float> filter(channels * channels * 3 * 3, 0.5f);
    cudaMemcpy(d_input, input.data(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter.data(), filter_size, cudaMemcpyHostToDevice);

    // Perform forward convolution
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor, d_input,
        filter_descriptor, d_filter,
        convolution_descriptor,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        nullptr, 0,
        &beta,
        output_descriptor, d_output));

    std::cout << "Forward convolution completed successfully!" << std::endl;

    // Perform backward pass (gradient wrt filter)
    float* d_grad_filter;
    cudaMalloc(&d_grad_filter, filter_size);
    checkCUDNN(cudnnConvolutionBackwardFilter(
        cudnn,
        &alpha,
        input_descriptor, d_input,
        output_descriptor, d_output,
        convolution_descriptor,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        nullptr, 0,
        &beta,
        filter_descriptor, d_grad_filter));

    std::cout << "Backward convolution (filter gradient) completed successfully!" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaFree(d_grad_filter);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    std::cout << "cuDNN training test completed successfully!" << std::endl;
    return 0;
}
