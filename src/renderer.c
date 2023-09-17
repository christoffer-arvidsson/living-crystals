#define GLEW_STATIC
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "renderer.h"
#include "constants.h"

#define VERT_SRC "src/shaders/main.vert"
#define FRAG_SRC "src/shaders/main.frag"

typedef enum {
    POSITION_ATTRIB = 0,
    COLOR_ATTRIB,
} Attribs;

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
    if (*shader == 0) {
        fprintf(stderr, "Could not create shader: %s\n", shader_type_as_cstr(shader_type));
    }
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

const char* read_shader_content(const char* file_name) {
    FILE *fp;
    long size = 0;
    char* shader_content;

    fp = fopen(file_name, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed reading shader file: %s", file_name);
        return NULL;
    }

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp) + 1;
    fclose(fp);

    fp = fopen(file_name, "r");
    shader_content = (char*) memset(malloc(size), '\0', size);
    fread(shader_content, 1, size-1, fp);
    fclose(fp);

    return shader_content;
}

void init_shaders(Renderer* renderer) {
    // Compile vertex shader

    const char* vertex_shader_source = read_shader_content(VERT_SRC);
    const char* frag_shader_source = read_shader_content(FRAG_SRC);

    GLuint vert_shader = 0;
    if (!compile_shader_source(vertex_shader_source, GL_VERTEX_SHADER, &vert_shader)) {
        exit(1);
    }

    GLuint frag_shader = 0;
    if (!compile_shader_source(frag_shader_source, GL_FRAGMENT_SHADER, &frag_shader)) {
        exit(1);
    }

    if (!link_program(vert_shader, frag_shader, &renderer->program)) {
        exit(1);
    }

}

void init_buffers(Renderer* renderer) {
    glGenVertexArrays(1, &renderer->vao);
    glBindVertexArray(renderer->vao);

    glGenBuffers(1, &renderer->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->vbo);
    Entity* entities = renderer->entity_container.entities;
    glBufferData(GL_ARRAY_BUFFER, MAX_ENTITIES * sizeof(entities[0]), entities, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(POSITION_ATTRIB);
    glVertexAttribPointer(POSITION_ATTRIB, 2, GL_FLOAT, GL_FALSE, sizeof(entities[0]), (GLvoid*) 0);
    glVertexAttribDivisor(POSITION_ATTRIB, 1);

    glEnableVertexAttribArray(COLOR_ATTRIB);
    glVertexAttribPointer(COLOR_ATTRIB, 3, GL_FLOAT, GL_FALSE, sizeof(entities[0]), (GLvoid*) (sizeof(GLfloat) * 2));
    glVertexAttribDivisor(COLOR_ATTRIB, 1);
}

Renderer* renderer_alloc(void) {
    Renderer* renderer = calloc(1, sizeof(Renderer));
    renderer->entity_container.size = 0;
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    renderer->window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "opengl", NULL, NULL);

    if (renderer->window == NULL) {
        fprintf(stderr, "Failed to create GLFW renderer->window\n");
        glfwTerminate();
        exit(1);
    }
    glfwMakeContextCurrent(renderer->window);
    glfwSetFramebufferSizeCallback(renderer->window, framebuffer_size_callback);
    glfwSetKeyCallback(renderer->window, process_input);

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

    init_shaders(renderer);
    init_buffers(renderer);

    glUseProgram(renderer->program);
    renderer->resolution_uniform = glGetUniformLocation(renderer->program, "resolution");
    renderer->radius_uniform = glGetUniformLocation(renderer->program, "radius");

    glUniform2f(renderer->resolution_uniform, SCREEN_WIDTH, SCREEN_HEIGHT);
    glUniform1f(renderer->radius_uniform, PARTICLE_RADIUS);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    return renderer;
}

void renderer_free(Renderer* renderer) {
    glDeleteVertexArrays(1, &renderer->vao);
    glDeleteBuffers(1, &renderer->vbo);
    glDeleteProgram(renderer->program);
    glfwTerminate();
    free(renderer);
}

void renderer_sync_buffers(Renderer* renderer) {
    glBindVertexArray(renderer->vao);
    glBindBuffer(GL_ARRAY_BUFFER, renderer->vbo);
    size_t n_entities = renderer->entity_container.size;
    Entity* entities = renderer->entity_container.entities;
    glBufferSubData(GL_ARRAY_BUFFER, 0, n_entities * sizeof(entities[0]), (GLvoid*) entities);
}

void renderer_add_entity(Renderer* renderer, const Entity* entity) {
    size_t size = renderer->entity_container.size;
    assert(size < PARTICLES_CAPACITY);
    renderer->entity_container.entities[size] = *entity;
    renderer->entity_container.size += 1;
}

void renderer_clear_entities(Renderer* renderer) {
    renderer->entity_container.size = 0;
}

void renderer_render(Renderer* renderer) {
    renderer_sync_buffers(renderer);

    glClear(GL_COLOR_BUFFER_BIT);
    glBindVertexArray(renderer->vao);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, renderer->entity_container.size);
}

void renderer_poll(Renderer* renderer) {
    glfwSwapBuffers(renderer->window);
    glfwPollEvents();
}
