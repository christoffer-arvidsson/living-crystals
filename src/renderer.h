#ifndef RENDERER_H
#define RENDERER_H

#define VERT_SHADER_SRC "src/shaders/main.vert"
#define FRAG_SHADER_SRC "src/shaders/main.frag"

#include <stdbool.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "constants.h"

#define MAX_ENTITIES PARTICLES_CAPACITY

typedef struct {
    GLfloat x;
    GLfloat y;
    GLfloat r;
    GLfloat g;
    GLfloat b;
} Entity;

typedef struct {
    Entity entities[MAX_ENTITIES];
    size_t size;
} EntityContainer;

typedef struct {
    GLFWwindow* window;
    GLuint program;
    GLint resolution_uniform;
    GLint radius_uniform;
    bool pause;
    GLuint vao;
    GLuint vbo;
    EntityContainer entity_container;
} Renderer;

Renderer* renderer_alloc(void);
void renderer_free(Renderer* renderer);
void renderer_render(Renderer* renderer);
void renderer_add_entity(Renderer* renderer, const Entity* entity);
void renderer_poll(Renderer* renderer);
void renderer_clear_entities(Renderer* renderer);

#endif /* RENDERER_H */
