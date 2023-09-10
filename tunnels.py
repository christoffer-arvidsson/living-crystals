#!/usr/bin/env python3
#
import sys, pygame
import numpy as np
import numpy.ma as ma
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt


def torque(sZ, dZ, φ, params):
    sZ = sZ.view(complex)[:, 0] # Convert 2-vector to complex
    dZ = dZ.view(complex)[:, 0] # Convert 2-vector to complex

    dist = sZ[:,None] - dZ[None,:]
    dist = ma.array(dist)
    dist[dist==0] = ma.masked
    dist[np.abs(dist)>params['r_c']] = ma.masked

    r = np.abs(dist)
    rH = dist / np.abs(dist)
    v = np.exp(φ*1j)
    v = v / np.abs(v)

    dot   = lambda a, b: a.real*b.real+a.imag*b.imag
    cross = lambda a, b: a.real*b.imag-a.imag*b.real

    t = dot(v[:, None], rH) / (r**2) * v[:, None]
    t = cross(t, rH)
    # t = dot(t, params.ez)
    t = np.sum(t, axis=1).filled(0)
    return t

class Model:
    def _init_positions(self, num):
        """ Generate random (x,y) pairs for each particles. """
        grid_size = self.params['grid_size']
        pos = self.rng.uniform(0, grid_size, (num, 2))
        return pos

    def __init__(self, params):
        self.step = 0
        self.params = params
        self.rng = np.random.default_rng()
        self.speeds = np.repeat(self.params['initial_speed'], self.params['num_particles'])
        # self.speeds = np.array([1,2,3,4])
        self.positions = self._init_positions(self.params['num_particles'])
        self.passive = self._init_positions(self.params['num_passive'])
        self.orientations = np.pi * 2 * self.rng.random(params['num_particles'])

        # Stats
        self.pos_history = np.zeros((self.params['steps'], self.params['num_particles'], 2))
        # self.msd = np.zeros((self.params['steps'], self.params['num_particles']))

        # Constants
        self.cphi = np.sqrt(2 * self.params['D_R'] * self.params['delta_t'])
        self.cpos = np.sqrt(2 * self.params['D_T'] * self.params['delta_t'])

    def calc_unit_v_vecs(self):
        thetas = self.orientations
        return np.array([np.cos(thetas), np.sin(thetas)]).swapaxes(0,1)

    def calc_torque2(self, sources, destinations, kdtree):
        vn_hats = self.calc_unit_v_vecs()
        torques = np.zeros_like(self.orientations)
        for s, src in enumerate(sources):
            vn_hat = vn_hats[s]
            indxs, rni_vecs, rni_dists = self.calc_inter_dist_opt(kdtree, src, self.params['r_c'], destinations)
            torque = 0

            for d, rni_vec in enumerate(rni_vecs):
                rni = rni_dists[d]
                if(rni != 0):
                    torque += (np.dot(vn_hat, rni_vec) / (rni**2))*np.cross(vn_hat, rni_vec)

            torques[s] = torque

        return torques


    def move_overlap(self, kdtree, sources, destinations):
        # Vectors and distance between source and every destination particle
        radius = self.params['radius']

        for s, src in enumerate(sources):
            idxs, rni_vec, rni_dist = self.calc_inter_dist_opt(kdtree, src, 2 * radius, destinations)
            for d, dest in enumerate(rni_vec):
                dist = rni_dist[d]
                if dist != 0:
                    overlap = (2 * radius - dist + 0.1)/2
                    sources[s] += rni_vec[d] * overlap
                    destinations[idxs[d]] -= rni_vec[d] * overlap
    def calc_inter_dist_opt(self, kdtree, source, rc, destinations):
        nidxs = kdtree.query_ball_point(source, rc)
        neighbors = destinations[nidxs]
        vecs = source - neighbors
        normed = np.sqrt(np.einsum('...i,...i', vecs, vecs))
        # normed[np.isnan(normed)] = np.finfo(neighbors.dtype).max

        return (nidxs, vecs / normed.reshape(-1, 1), normed)

    def move_particles(self):
        """ Move a particle according to non-chiral brownian motion."""
        num_particles = self.params['num_particles']
        num_passive = self.params['num_passive']
        D_R = self.params['D_R']
        D_T = self.params['D_T']
        delta_t = self.params['delta_t']

        # Random variables
        mu, sigma = 0, 1 # mean and standard deviation
        w_phi = np.random.normal(mu, sigma, num_particles)
        w_x = np.random.normal(mu, sigma, num_particles)
        w_y = np.random.normal(mu, sigma, num_particles)

        cphi = np.multiply(self.cphi, w_phi)
        cx = np.multiply(self.cpos, w_x)
        cy = np.multiply(self.cpos, w_y)

        kdtree_pos = KDTree(self.positions)
        kdtree_pas = KDTree(self.passive)
        # self.calc_inter_dist_opt(kdtree_pos, [0,0])

        # Torque vector
        # attT = self.params['T_0a'] * self.calc_torque2(self.positions, self.positions, kdtree_pos)
        # repT = self.params['T_0r'] * self.calc_torque2(self.positions, self.passive, kdtree_pas)

        attT = self.params['T_0a'] * torque(self.positions, self.positions, self.orientations, self.params)
        repT = self.params['T_0r'] * torque(self.positions, self.passive, self.orientations, self.params)

        # Move overlapping particles
        self.move_overlap(kdtree_pos, self.positions, self.positions)
        self.move_overlap(kdtree_pas, self.positions, self.passive)
        # self.move_overlap(kdtree_pas, self.passive, self.passive)

        self.orientations += attT * delta_t + cphi
        self.orientations -= repT * delta_t
        self.positions[:,0] += self.speeds * np.cos(self.orientations) * delta_t + cx
        self.positions[:,1] += self.speeds * np.sin(self.orientations) * delta_t + cy

        w_x = np.random.normal(mu, sigma, num_passive)
        w_y = np.random.normal(mu, sigma, num_passive)
        cx = np.multiply(np.sqrt(2 * D_T * delta_t), w_x)
        cy = np.multiply(np.sqrt(2 * D_T * delta_t), w_y)

        self.passive[:,0] += cx
        self.passive[:,1] += cy

        if params['loop_around']:
            self.positions %= self.params['grid_size']


    def tick(self):
        # Record positions history
        self.pos_history[self.step] = self.positions
        self.move_particles()

        self.step += 1

    def draw_task1(self, particle_surf, line_surf):
        gs = self.params['grid_size'] / 4
        if self.step == 0:
            self.positions = np.array([[3*gs, 3*gs], [gs, 3 * gs], [gs,gs], [3*gs,gs]])
            self.speeds = [0,1,2,3]
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(self.params['T_0a'] / (4 * self.params['radius']**2)), False, (255, 255, 255))
        colors = [(255,0,0,255), (0,255,0,255), (0,0,255,255), (255,255,0,255)]
        for p in self.passive:
            pygame.draw.circle(particle_surf, (50,50,200,100), p, params['r_c'])
        for p in self.positions:
            pygame.draw.circle(particle_surf, (200,0,100,100), p, params['r_c'])
        for i,p in enumerate(self.positions):
            pygame.draw.circle(line_surf, colors[i], p, 2)
            pygame.draw.circle(particle_surf, (255,255,255,255), p, params['radius'])

        for p in self.passive:
            pygame.draw.circle(particle_surf, (40,40,70,255), p, params['radius'])


        # particle_surf.fill((200, 200, 200, 200), special_flags=pygame.BLEND_RGBA_MULT)
        # if self.step % 100:
        #     screen.fill((255,255,255,0.000001), special_flags=pygame.BLEND_RGBA_MULT)

        screen.blit(line_surf, (0,0), special_flags=0)
        particle_surf.fill((255, 255, 255,250), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(particle_surf, (0,0), special_flags=0)
        particle_surf.fill((0,0,0, 0))
        # screen.blit(textsurface,(0,0))
        pygame.display.flip()

        screen.fill((0,0,0,0))
    def draw_tunnels(self, particle_surf, line_surf):
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(self.params['T_0a'] / (4 * self.params['radius']**2)), False, (255, 255, 255))
        for p in self.passive:
            pygame.draw.circle(particle_surf, (50,50,200,100), p, params['r_c'])
        for p in self.positions:
            pygame.draw.circle(particle_surf, (200,0,100,100), p, params['r_c'])
        for i,p in enumerate(self.positions):
            pygame.draw.circle(line_surf, (115, 129, 189, 200), p, 2)
            pygame.draw.circle(particle_surf, (255,255,255,255), p, params['radius'])

        for p in self.passive:
            pygame.draw.circle(particle_surf, (40,40,70,255), p, params['radius'])


        # particle_surf.fill((200, 200, 200, 200), special_flags=pygame.BLEND_RGBA_MULT)
        # if self.step % 100:
        #     screen.fill((255,255,255,0.000001), special_flags=pygame.BLEND_RGBA_MULT)

        screen.blit(line_surf, (0,0), special_flags=0)
        particle_surf.fill((255, 255, 255,250), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(particle_surf, (0,0), special_flags=0)
        particle_surf.fill((0,0,0, 0))
        # screen.blit(textsurface,(0,0))
        pygame.display.flip()

        screen.fill((0,0,0,0))

    def draw_cluster(self, particle_surf, line_surf):
        myfont = pygame.font.SysFont('Comic Sans MS', 30)
        textsurface = myfont.render(str(self.params['T_0a'] / (4 * self.params['radius']**2)), False, (255, 255, 255))
        screen.fill((0,0,0,0))
        # for p in self.passive:
        #     pygame.draw.circle(particle_surf, (50,50,200,100), p, params['r_c'])
        for p in self.positions:
            pygame.draw.circle(particle_surf, (200,0,100,100), p, params['r_c'])
        for i,p in enumerate(self.positions):
            pygame.draw.circle(line_surf, (115, 129, 189, 80), p, 4)
            pygame.draw.circle(particle_surf, (255,255,255,255), p, params['radius'])

        for p in self.passive:
            pygame.draw.circle(particle_surf, (40,40,70,255), p, params['radius'])


        # particle_surf.fill((200, 200, 200, 200), special_flags=pygame.BLEND_RGBA_MULT)
        # if self.step % 100:
        #     screen.fill((255,255,255,0.000001), special_flags=pygame.BLEND_RGBA_MULT)

        screen.blit(particle_surf, (0,0), special_flags=0)
        # screen.blit(line_surf, (0,0), special_flags=0)
        particle_surf.fill((0,0,0, 0))
        # screen.blit(textsurface,(0,0))
        pygame.display.flip()


    def run(self, screen):
        window = (params['grid_size'], params['grid_size'])
        particle_surf = pygame.surface.Surface(window, pygame.SRCALPHA, 32)
        line_surf = pygame.surface.Surface(window, pygame.SRCALPHA, 32)
        screen.fill((0,0,0))

        screenshot_region = pygame.Rect(0, 0, params['grid_size'], params['grid_size'])
        scr = line_surf.subsurface(screenshot_region)

        # WAit for keypress
        wait = False
        DRAW = True
        while wait:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN: wait = False

        for step in range(self.params['steps']):

            # Handle exit
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

            if self.step == 1000:
                print("Screenshot!")
                pygame.image.save(scr, "screenshot.jpg")


            if DRAW:
                self.draw_cluster(particle_surf, line_surf)

            self.tick()

pygame.init()

params = {
    'steps': 1000,
    'num_particles': 4,
    'num_passive': 0,
    'grid_size': 256,
    'initial_speed': 1,
    'delta_t': 0.01,
    'D_T': 0.5,
    'D_R': 0.5,
    'nabla':0,
    'T_0r': 0,
    'T_0a': 0,
    'r_c': 0,
    'radius': 6,
    'color_radius': 20, #Radius for coloring clusters
    'loop_around': False,
}
tunnels = {
    'steps': 100000,
    'num_particles': 100,
    'num_passive': 1000,
    'grid_size': 512,
    'initial_speed': 4,
    'delta_t': 1,
    'D_T': 0.01,
    'D_R': 0.01,
    'nabla':0,
    'T_0r': 20,
    'T_0a': 20,
    'r_c': 25,
    'radius': 6,
    'color_radius': 20, #Radius for coloring clusters
    'loop_around': True,
}

clusters = {
    'steps': 100000,
    'num_particles': 200,
    'num_passive': 0,
    'grid_size': 512,
    'initial_speed': 20,
    'delta_t': 0.1,
    'D_T': 0.22,
    'D_R': 0.16,
    'nabla':0,
    'T_0r': 0,
    'T_0a': 1500,
    'r_c': 80,
    'radius': 6,
    'color_radius': 50, #Radius for coloring clusters
    'loop_around': True,
}

p = tunnels
screen = pygame.display.set_mode((p['grid_size'], p['grid_size']))

model = Model(p)
model.run(screen)

# STATS

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel(r'$\log{\tau}$')
# ax.set_ylabel(r'$\log{MSD(\tau)}$')


# def msd_tau(trajectory, tau):
#     diffx = trajectory[0, tau:] - trajectory[0,:-tau]
#     diffy = trajectory[1, tau:] - trajectory[1,:-tau]
#     square = diffx**2 + diffy**2

#     s = np.mean(square)
#     return s


# def MSD(trajectory):
#     steps = trajectory.shape[1]
#     msds = np.zeros(steps)
#     taus = np.arange(1, steps)
#     t = trajectory.copy()
#     for tau in taus:
#         msd = msd_tau(t, tau)
#         msds[tau-1] = msd

#     return taus, msds


# def calc_msds_iteration(params, pos_history):
#     MSDS = np.zeros((params['num_particles'], params['steps']))
#     for pi, p in enumerate(pos_history):
#         taus, msds = MSD(p)
#         MSDS[pi] = msds

#     return MSDS

# iterations = 5
# MSDS = np.zeros((params['num_particles'], params['steps']))
# for i in range(iterations):
#     model = Model(params)
#     model.run(screen)

#     ph = model.pos_history.copy()
#     pos_history = ph.swapaxes(0,2)
#     pos_history = pos_history.swapaxes(0,1)

#     msds = calc_msds_iteration(params, pos_history)
#     MSDS += msds

# average = MSDS / iterations
# print(average.shape)
# speeds = [0,1,2,3]
# colors = ['red', 'green', 'blue', 'yellow']
# for pi, p in enumerate(average):
#     ax.plot(np.log(np.arange(average.shape[1]//2)), np.log(p[:len(p)//2]), c=colors[pi])

# ax.set_title(f"{iterations} iterations of averaging for MSD")
# plt.savefig('msd.png')
# plt.show()




print("Finished")
