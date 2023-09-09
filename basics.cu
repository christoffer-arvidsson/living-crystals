#include <stdio.h>
#include <cuda_runtime.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
}

// CUDA kernel to change positions
__global__ void colorPositions(float4* colors, int numColors, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numColors) {
        float t = idx + time;
        colors[idx] = make_float4(0.5f + 0.5f * sinf(t), 0.5f + 0.5f * sinf(t + 2.0f), 0.5f + 0.5f * sinf(t + 4.0f), 1.0f);
    }
}

float4* d_colors; // GPU buffer for colors
float currentTime = 0.0f;

GLfloat vertices[] = {
    // first triangle
     0.5f,  0.5f, 0.0f,  // top right
     0.5f, -0.5f, 0.0f,  // bottom right
    -0.5f, -0.5f, 0.0f,  // bottom left
    -0.5f,  0.5f, 0.0f   // top left
};

GLuint indices[] = {
    0, 2, 3, // top left triangle
    0, 1, 2 // bottom right triangle
};

const char* vertexShaderSource = "#version 330 core\n"
    "layout(location = 0) in vec3 aPos;"
    "void main() {"
    "   gl_Position = vec4(aPos, 1.0);"
    "}";

const char* fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;"
    "void main() {"
    "   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);"
    "}";

void compile_shaders(GLuint vertexShader, GLuint fragmentShader, GLuint shaderProgram) {
    int success;
    char infoLog[512];
    // Compile vertex shader
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        fprintf(stderr, "vertex shader compilation failed: %s\n", infoLog);
    }

    // Compile fragment shader
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        fprintf(stderr, "fragment shader compilation failed: %s\n", infoLog);
    }

    // Shader program
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void display() {

}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800U, 600U, "opengl", NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        fprintf(stderr, "Failed to initialize GLAD\n");
        return -1;
    }

    glViewport(0, 0, 800U, 600U);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Compile shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint shaderProgram = glCreateProgram();
    compile_shaders(vertexShader, fragmentShader, shaderProgram);

    GLuint vao, vbo, ebo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);


    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Draw wireframe mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);


    // Create and initialize GPU buffer for colors
    int numColors = 4;  // Number of vertices
    cudaMalloc((void**)&d_colors, numColors * sizeof(float4));

    const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString(GL_VERSION); // version as a string
    printf("Renderer: %s\n", renderer);
    printf("OpenGL version supported %s\n", version);

    float currentTime = 0.0f;

    while(!glfwWindowShouldClose(window)) {
        // input
        process_input(window);

        // Update the triangle
        currentTime += 0.01f;
        colorPositions<<<1, numColors>>>(d_colors, 4, currentTime);
        cudaMemcpy(vertices, d_colors, 3 * sizeof(float4), cudaMemcpyDeviceToHost);

        // render stuff
        glClear(GL_COLOR_BUFFER_BIT);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // swap and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);
    glfwTerminate();

    cudaFree(d_colors);

    return 0;
}
