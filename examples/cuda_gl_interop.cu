#include <iostream>
#include <cmath>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA kernel to calculate vertex positions
__global__ void updateVertices(float2* vertices, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float amplitude = 0.5f;
    float frequency = 1.0f;
    vertices[idx].x = idx / 3.0f + sinf(frequency * time + idx) * amplitude;
    vertices[idx].y = cosf(frequency * time + idx) * amplitude;
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW to use the core profile and set required version
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL + CUDA", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Create a CUDA-OpenGL interop buffer
    GLuint vbo;
    float2* d_vertices; // Device pointer for CUDA

    // Generate and bind a vertex buffer object (VBO)
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float2), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA
    cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard);

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate time for animation
        float time = glfwGetTime();

        // Map the CUDA-OpenGL interop buffer
        cudaGraphicsMapResources(1, &cuda_vbo_resource);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource);

        // Launch the CUDA kernel to update vertices
        updateVertices<<<1, 3>>>(d_vertices, time);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        // Unmap the buffer
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource);

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT);

        // Render the triangle
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Specify the vertex attribute pointers
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        // Draw the triangle
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Disable the vertex attribute pointer
        glDisableVertexAttribArray(0);

        // Swap front and back buffers
        glfwSwapBuffers(window);
        // Poll for and process events
        glfwPollEvents();
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);
    glfwTerminate();

    return 0;
}
