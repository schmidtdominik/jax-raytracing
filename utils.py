import jax.numpy as jnp
from jax import jit, vmap

# coordinates are (x, y, z)
# x is forward backward
# y is left right
# z is up down


def stack_dict_list(l):
    keys = l[0].keys()
    print(keys)
    return [jnp.stack([jnp.array(elm[k]).astype(float) for elm in l]) for k in keys]

#@jit
def get_init(res_x, res_y, x_persp, y_persp, camera_persp, x_offset, y_offset):
    camera_pos = jnp.array([-camera_persp, 0, 0]) # from above: set last to -2

    focal_plane = jnp.zeros((res_x, res_y, 3))
    x_grid, y_grid = jnp.meshgrid(
        jnp.linspace(-x_persp, x_persp, res_x),
        jnp.linspace(-y_persp, y_persp, res_y)
    )
    focal_plane = focal_plane.at[:, :, 1].set(x_grid.T + x_offset)
    focal_plane = focal_plane.at[:, :, 2].set(y_grid.T + y_offset)
    ray_dirs = focal_plane - camera_pos
    ray_dirs = ray_dirs / jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)

    ray_origin = jnp.empty((res_x, res_y, 3))
    ray_origin = ray_origin.at[:, :].set(camera_pos)

    return ray_origin, ray_dirs
