#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "particle.h"
#include "constants.h"

bool pause = true;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(width/2 - SCREEN_WIDTH / 2,
               height/2 - SCREEN_HEIGHT / 2,
               SCREEN_WIDTH,
               SCREEN_HEIGHT);
}

void process_input(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        pause = !pause;
    }
}

void message_callback(GLenum source,
                      GLenum type,
                      GLuint id,
                      GLenum severity,
                      GLsizei length,
                      const GLchar* message,
                      const void* userParam)
{
    (void) source;
    (void) id;
    (void) length;
    (void) userParam;
    fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
            (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
            type, severity, message);
}

const char* vertex_shader_source =
    "#version 330 core\n"
    "\n"
    "uniform vec2 resolution;\n"
    "uniform float radius;\n"
    "\n"
    "layout(location = 0) in vec2 position;\n"
    "layout(location = 1) in vec3 color;\n"
    "\n"
    "out vec3 particle_color;\n"
    "\n"
    "vec2 screen_to_ndc(vec2 pos) {\n"
    "    return (pos - resolution * 0.5) / (resolution * 0.5);\n"
    "}\n"
    "\n"
    "void main() {\n"
    "    vec2 uv = vec2(\n"
    "        float(gl_VertexID & 1),\n"
    "        float((gl_VertexID >> 1) & 1));\n"
    "    gl_Position = vec4(\n"
    "       screen_to_ndc(position + uv * radius * 2),\n"
    "       0.0,\n"
    "       1.0);\n"
    "    particle_color = color;\n"
    "}\n";

const char* frag_shader_source =
    "#version 330 core\n"
    "in vec3 particle_color;\n"
    "void main() {\n"
    "   gl_FragColor = vec4(particle_color, 1.0);\n"
    "}\n";

const char *shader_type_as_cstr(GLuint shader)
{
    switch (shader) {
    case GL_VERTEX_SHADER:
        return "GL_VERTEX_SHADER";
    case GL_FRAGMENT_SHADER:
        return "GL_FRAGMENT_SHADER";
    default:
        return "(Unknown)";
    }
}

bool compile_shader_source(const GLchar *source, GLenum shader_type, GLuint *shader) {
    *shader = glCreateShader(shader_type);
    glShaderSource(*shader, 1, &source, NULL);
    glCompileShader(*shader);


    GLint compiled = 0;
    glGetShaderiv(*shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLchar message[1024];
        GLsizei message_size = 0;
        glGetShaderInfoLog(*shader, sizeof(message), &message_size, message);
        fprintf(stderr, "ERROR: shader compilation failed: %s\n", shader_type_as_cstr(shader_type));
        fprintf(stderr, "%.*s\n", message_size, message);
        return false;
    }

    return true;
}

bool link_program(GLuint vert_shader, GLuint frag_shader, GLuint *program) {
    *program = glCreateProgram();
    glAttachShader(*program, vert_shader);
    glAttachShader(*program, frag_shader);
    glLinkProgram(*program);

    GLint linked = 0;
    glUseProgram(*program);
    glGetProgramiv(*program, GL_LINK_STATUS, &linked);

    if (!linked) {
        GLchar message[1024];
        GLsizei message_size = 0;
        glGetProgramInfoLog(*program, sizeof(message), &message_size, message);
        fprintf(stderr, "ERROR: failed linking shader program: %.*s\n", message_size, message);
        return false;
    }

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    return true;

}

GLuint program = 0;
GLint resolution_uniform = 0;
GLint radius_uniform = 0;

void init_shaders(void) {
    // Compile vertex shader

    GLuint vert_shader = 0;
    if (!compile_shader_source(vertex_shader_source, GL_VERTEX_SHADER, &vert_shader)) {
        exit(1);
    }

    GLuint frag_shader = 0;
    if (!compile_shader_source(frag_shader_source, GL_FRAGMENT_SHADER, &frag_shader)) {
        exit(1);
    }

    if (!link_program(vert_shader, frag_shader, &program)) {
        exit(1);
    }

    glUseProgram(program);
    resolution_uniform = glGetUniformLocation(program, "resolution");
    radius_uniform = glGetUniformLocation(program, "radius");

}

typedef enum {
    POSITION_ATTRIB = 0,
    COLOR_ATTRIB,
} Attribs;

typedef struct {
    GLfloat x;
    GLfloat y;
    GLfloat r;
    GLfloat g;
    GLfloat b;
} Vert;

Vert verts[PARTICLES_CAPACITY];
size_t verts_count = 0;

GLuint vao = 0;
GLuint vbo = 0;

void init_buffers(void) {
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(POSITION_ATTRIB);
    glVertexAttribPointer(POSITION_ATTRIB, 2, GL_FLOAT, GL_FALSE, sizeof(verts[0]), (GLvoid*) 0);
    glVertexAttribDivisor(POSITION_ATTRIB, 1);

    glEnableVertexAttribArray(COLOR_ATTRIB);
    glVertexAttribPointer(COLOR_ATTRIB, 3, GL_FLOAT, GL_FALSE, sizeof(verts[0]), (GLvoid*) (sizeof(GLfloat) * 2));
    glVertexAttribDivisor(COLOR_ATTRIB, 1);
}

void clear_verts(void) {
    verts_count = 0;
}

void push_vert(float x, float y, float r, float g, float b) {
    assert(verts_count < PARTICLES_CAPACITY);
    verts[verts_count].x = x;
    verts[verts_count].y = y;
    verts[verts_count].r = r;
    verts[verts_count].g = g;
    verts[verts_count].b = b;
    verts_count += 1;
}

void sync_buffers(void) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, verts_count * sizeof(verts[0]), verts);
}

float3 particle_type_to_color(ParticleType type) {
    switch (type) {
    case ACTIVE:
        return make_float3(0.8f, 0.2f, 0.4f);
    case PASSIVE:
        return make_float3(0.3f, 0.3f, 0.3f);
    default:
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

void setup_particles(size_t n_particles) {
    clear_particles();

    // PASSIVE
    for (size_t p=0; p<n_particles; ++p) {
        float pos_x = (float)rand()/(float)(RAND_MAX/SCREEN_WIDTH);
        float pos_y = (float)rand()/(float)(RAND_MAX/SCREEN_HEIGHT);
        float speed = (float)rand()/(float)(RAND_MAX);
        float orient = (float)rand()/((float)(RAND_MAX)/(3.14f * 2.0f));
        ParticleType type = PASSIVE;
        if ((float)rand()/((float)(RAND_MAX)) < (float)ACTIVE_FRACTION) {
            speed += PARTICLE_SPEED;
            type = ACTIVE;
        }
        else {
            speed = 0.0f;
        }
        push_particle(make_float2(pos_x, pos_y), speed, orient, type, PARTICLE_RADIUS);
    }
}

void particles_to_vert(void) {
    clear_verts();

    for (size_t p = 0; p < get_num_particles(); ++p) {
        Particle* part = get_particle(p);

        float3 color = particle_type_to_color(part->charge);
        push_vert(part->pos.x, part->pos.y, color.x, color.y, color.z);
    }
}

void render_particles(void) {
    particles_to_vert();
    sync_buffers();

    // render stuff
    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, verts_count);
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "opengl", NULL, NULL);

    if (window == NULL) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetKeyCallback(window, process_input);

    if (GLEW_OK != glewInit()) {
        fprintf(stderr, "ERROR: Could not initialize GLEW!\n");
        exit(1);
    }

    if (!GLEW_EXT_draw_instanced) {
        fprintf(stderr, "ERROR: Support for EXT_draw_instanced is required!\n");
        exit(1);
    }

    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(message_callback, 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    /* glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT); */

    const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
    const GLubyte* version = glGetString(GL_VERSION); // version as a string
    printf("Renderer: %s\n", renderer);
    printf("OpenGL version supported %s\n", version);

    // Compile shader
    init_shaders();
    init_buffers();

    // Draw wireframe mode
    /* glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); */

    glUniform2f(resolution_uniform, SCREEN_WIDTH, SCREEN_HEIGHT);
    glUniform1f(radius_uniform, PARTICLE_RADIUS);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);


    setup_particles(PARTICLES_CAPACITY);
    init_simulation();
    while(!glfwWindowShouldClose(window)) {
        // input
        if (!pause) {
            tick_simulation();
            render_particles();
        }

        // swap and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);
    glfwTerminate();

    return 0;
}
