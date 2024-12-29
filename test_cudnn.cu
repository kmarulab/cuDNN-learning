#include <cudnn.h>
#include <iostream>

#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

int main() {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    std::cout << "cuDNN initialized successfully!" << std::endl;

    // Create and describe a tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        1, 1, 5, 5));

    std::cout << "Tensor descriptor created and set successfully!" << std::endl;

    // Cleanup
    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
    checkCUDNN(cudnnDestroy(cudnn));

    std::cout << "cuDNN test completed successfully!" << std::endl;
    return 0;
}
