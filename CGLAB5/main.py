import taichi as ti
from config import *
from raytracer import normalize, reflect, scene_intersect

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES_X, RES_Y))

# UI 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())


@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        u = (i - RES_X / 2.0) / RES_Y * 2.0
        v = (j - RES_Y / 2.0) / RES_Y * 2.0

        ro = ti.Vector(CAMERA_POS)
        rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        for bounce in range(max_bounces[None]):
            t, N, obj_color, mat_id = scene_intersect(ro, rd)

            if t > 1e9:
                final_color += throughput * bg_color
                break

            p = ro + rd * t

            if mat_id == MAT_MIRROR:
                ro = p + N * EPSILON
                rd = normalize(reflect(rd, N))
                throughput *= 0.8 * obj_color
            else:
                L = normalize(light_pos - p)
                shadow_ray_orig = p + N * EPSILON
                shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
                dist_to_light = (light_pos - p).norm()

                ambient = 0.2 * obj_color
                direct_light = ambient

                if shadow_t > dist_to_light or shadow_t < 0:
                    diff = ti.max(0.0, N.dot(L))
                    diffuse = 0.8 * diff * obj_color
                    direct_light += diffuse

                final_color += throughput * direct_light
                break

        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)


def main():
    light_pos_x[None] = LIGHT_POS_DEFAULT[0]
    light_pos_y[None] = LIGHT_POS_DEFAULT[1]
    light_pos_z[None] = LIGHT_POS_DEFAULT[2]
    max_bounces[None] = MAX_BOUNCES_DEFAULT

    window = ti.ui.Window("光线追踪 - Whitted-Style Ray Tracing", (RES_X, RES_Y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    print("=" * 50)
    print("光线追踪程序已启动")
    print("=" * 50)

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.22):
            light_pos_x[None] = gui.slider_float("Light X", light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float("Light Y", light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float("Light Z", light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 5)

        window.show()

    print("程序已退出")


if __name__ == "__main__":
    main()