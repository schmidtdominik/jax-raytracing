import jax.numpy as jnp


# coordinates are (x, y, z)
# x is forward backward
# y is left right
# z is up down

def stack_dict_list(l):
    keys = l[0].keys()
    print(keys)
    return [jnp.stack([jnp.array(elm[k]).astype(float) for elm in l]) for k in keys]


def get_init(res_x, res_y, x_persp, y_persp, camera_persp, x_offset, y_offset):
    ray_dirs = jnp.ones((res_x, res_y, 3))
    x_grid, y_grid = jnp.meshgrid(jnp.linspace(-x_persp, x_persp, res_x), jnp.linspace(-y_persp, y_persp, res_y))
    ray_dirs = ray_dirs.at[:, :, 0].set(camera_persp)
    ray_dirs = ray_dirs.at[:, :, 1].set(x_grid.T+x_offset)
    ray_dirs = ray_dirs.at[:, :, 2].set(y_grid.T+y_offset)
    ray_dirs = ray_dirs / jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    ray_pos = jnp.zeros((res_x, res_y, 3))
    return ray_pos, ray_dirs