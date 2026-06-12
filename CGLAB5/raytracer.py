import taichi as ti
from config import *


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c

    if delta > 0:
        sqrt_delta = ti.sqrt(delta)
        t1 = (-b - sqrt_delta) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)

    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])

    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1

    return t, normal


@ti.func
def get_ground_color(p):
    grid_scale = GROUND_CHECKER_SIZE
    ix = ti.floor(p.x * grid_scale)
    iz = ti.floor(p.z * grid_scale)

    color = ti.Vector([0.0, 0.0, 0.0])
    if (int(ix) + int(iz)) % 2 == 0:
        color = ti.Vector(GROUND_COLOR1)
    else:
        color = ti.Vector(GROUND_COLOR2)
    return color


@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 红色漫反射球
    t, n = intersect_sphere(ro, rd, ti.Vector(RED_SPHERE_CENTER), RED_SPHERE_RADIUS)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector(RED_SPHERE_COLOR)
        hit_mat = MAT_DIFFUSE

    # 2. 银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector(MIRROR_SPHERE_CENTER), MIRROR_SPHERE_RADIUS)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector(MIRROR_SPHERE_COLOR)
        hit_mat = MAT_MIRROR

    # 3. 地面
    t, n = intersect_plane(ro, rd, GROUND_Y)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = get_ground_color(ro + rd * t)
        hit_mat = MAT_DIFFUSE

    return min_t, hit_n, hit_c, hit_mat