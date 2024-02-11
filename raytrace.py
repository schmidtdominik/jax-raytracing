from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap


def ray_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    # ray_dir should be normalized
    offset_ray_origin = ray_origin - sphere_center
    a = jnp.dot(ray_dir, ray_dir)
    b = 2 * jnp.dot(offset_ray_origin, ray_dir)
    c = jnp.dot(offset_ray_origin, offset_ray_origin) - sphere_radius**2

    discriminant = b**2 - 4 * a * c
    dist = (-b - jnp.sqrt(discriminant)) / (2 * a)

    # sphere was hit if (discriminant >= 0) & (dist >= 0)
    # if discriminant < 0, then dist is nan already, if dist < 0,
    # meaning hitpoint is against ray direction, set dist to nan
    dist = jnp.where(dist < 0, jnp.nan, dist)

    return dist


def ray_intersect_target_batch(ray_origin, ray_dir, sphere_center, sphere_radius):
    dist = ray_intersect(ray_origin, ray_dir, sphere_center, sphere_radius)
    closest_hit = jnp.nanargmin(dist)
    dist = dist[closest_hit]

    return dist, closest_hit


@jit
@partial(vmap, in_axes=(1, 1, 1, None, None, None, None, None, None))
@partial(vmap, in_axes=(0, 0, 0, None, None, None, None, None, None))
def full_ray_trace(
    ray_origin,
    ray_dir,
    key,
    sphere_center,
    sphere_radius,
    mat_color,
    em_color,
    em_strength,
    mat,
):
    def ray_trace_single_hit(args):
        inc_light, ray_color, ray_origin, ray_dir, key, i, done = args
        dist, closest_hit = ray_intersect_target_batch(
            ray_origin, ray_dir, sphere_center, sphere_radius
        )
        did_hit = ~jnp.isnan(dist)  # or: closest_hit != -1
        # done = done | (did_hit & (em_strength[closest_hit] == 1) & (jnp.arange(10)[i] == 0))
        # did_hit = did_hit & ~done

        hit_point = ray_origin + ray_dir * dist
        normal = hit_point - sphere_center[closest_hit]
        normal = normal / jnp.linalg.norm(normal)

        emitted_light = em_color[closest_hit] * em_strength[closest_hit]
        light_strength = jnp.dot(normal, -ray_dir)
        light_strength = jnp.where(jnp.isnan(light_strength), 0, light_strength)
        inc_light += did_hit * (emitted_light * ray_color)
        # *light_strength
        ray_color = (
            did_hit * ray_color * mat_color[closest_hit] + (~did_hit) * ray_color
        )

        key, subkey = jax.random.split(key, 2)
        random_dir = jax.random.normal(subkey, (3,))
        random_dir = random_dir / jnp.linalg.norm(random_dir)
        diffuse_reflect = random_dir * jnp.sign(jnp.dot(random_dir, normal))
        diffuse_reflect = diffuse_reflect / jnp.linalg.norm(diffuse_reflect)

        specular_reflect = (
            ray_dir - 2 * jnp.dot(ray_dir, normal) * normal
        )  # maybe should be -raydir too?
        specular_reflect = specular_reflect / jnp.linalg.norm(specular_reflect)

        alpha = mat[closest_hit]
        reflect_dir = alpha * diffuse_reflect + (1 - alpha) * specular_reflect

        return inc_light, ray_color, hit_point, reflect_dir, key, i + 1, done | ~did_hit

    inc_light = jnp.zeros((3,))
    ray_color = jnp.ones((3,))

    # inc_light, ray_color, _, _, _ = ray_trace_single_hit(inc_light, ray_color, ray_origin, ray_dir, key)

    def cond_fun(args):
        i = args[-2]
        done = args[-1]
        return (i < 8) | done

    inc_light, ray_color, _, _, _, _, _ = jax.lax.while_loop(
        cond_fun,
        ray_trace_single_hit,
        (inc_light, ray_color, ray_origin, ray_dir, key, 0, False),
    )
    return inc_light


# coordinates are (x, y, z)
# x is forward backward
# y is left right
# z is up down


def stack_dict_list(l):
    keys = l[0].keys()
    return [jnp.stack([jnp.array(elm[k]).astype(float) for elm in l]) for k in keys]


# @jit
def get_init(res_x, res_y, x_persp, y_persp, camera_persp, x_offset, y_offset):
    camera_pos = jnp.array([-camera_persp, 0, 0])  # from above: set last to -2

    focal_plane = jnp.zeros((res_x, res_y, 3))
    x_grid, y_grid = jnp.meshgrid(
        jnp.linspace(-x_persp, x_persp, res_x), jnp.linspace(-y_persp, y_persp, res_y)
    )
    focal_plane = focal_plane.at[:, :, 1].set(x_grid.T + x_offset)
    focal_plane = focal_plane.at[:, :, 2].set(y_grid.T + y_offset)
    ray_dirs = focal_plane - camera_pos
    ray_dirs = ray_dirs / jnp.linalg.norm(ray_dirs, axis=-1, keepdims=True)

    ray_origin = jnp.empty((res_x, res_y, 3))
    ray_origin = ray_origin.at[:, :].set(camera_pos)

    return ray_origin, ray_dirs
