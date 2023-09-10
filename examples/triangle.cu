#include <cuda_runtime.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA kernel to change color
__global__ void colorChange(float4* colors, int numColors, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numColors) {
        float t = idx + time;
        colors[idx] = make_float4(0.5f + 0.5f * sinf(t), 0.5f + 0.5f * sinf(t + 2.0f), 0.5f + 0.5f * sinf(t + 4.0f), 1.0f);
    }
}

// Vertex and color data for a triangle
GLfloat vertices[] = {
    -0.6f, -0.6f, 0.0f,
    0.6f, -0.6f, 0.0f,
    0.0f,  0.6f, 0.0f
};

float4* d_colors;  // GPU buffer for colors

float currentTime = 0.0f;

void initGL() {
    // Initialize GLEW
    glewInit();

    // Create and bind a vertex buffer object (VBO)
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Create a shader program and link it
    GLuint shaderProgram = glCreateProgram();
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    const char* vertexShaderSource = "#version 330 core\n"
        "layout(location = 0) in vec3 aPos;"
        "void main() {"
        "   gl_Position = vec4(aPos, 1.0);"
        "}";
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glAttachShader(shaderProgram, vertexShader);
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Specify vertex attribute pointers
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);

    // Create and initialize GPU buffer for colors
    int numColors = 3;  // Number of vertices
    cudaMalloc((void**)&d_colors, numColors * sizeof(float4));

    // Set OpenGL clear color
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Call the CUDA kernel to change colors
    colorChange<<<1, 3>>>(d_colors, 3, currentTime);
    cudaMemcpy(vertices, d_colors, 3 * sizeof(float4), cudaMemcpyDeviceToHost);

    // Draw the triangle
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glutSwapBuffers();
}

void idle() {
    currentTime += 0.01f;
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(800, 600);
    glutCreateWindow("CUDA + OpenGL Example");

    initGL();

    glutDisplayFunc(display);
    glutIdleFunc(idle);

    glutMainLoop();

    // Clean up
    cudaFree(d_colors);

    return 0;
}
