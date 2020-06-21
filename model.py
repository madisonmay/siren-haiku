import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import optix
import haiku as hk
from haiku.initializers import RandomUniform
from matplotlib import pyplot as plt
import imageio


class Config:
    w0 = 30.
    lr = 1e-4
    n_steps = 10000
    hidden_size = 512
    n_layers = 5


def rgba_to_rgb(rgba):
    """
    O - 1 normalized rgba --> rgb
    """
    rgb = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    rgb[:,:,0] = r 
    rgb[:,:,1] = g 
    rgb[:,:,2] = b 
    rgb *= np.expand_dims(a, -1)
    return rgb


def load_image(path):
    """
    Load img and convert to 0-1 normalized RGB
    """
    img = imageio.imread(path).astype(np.float32)
    img /= 255.
    if img.shape[-1] == 4:
        img = rgba_to_rgb(img)
    return img


def init_scale(n, w0=1.):
    """
    See SIREN paper (https://arxiv.org/pdf/2006.09661.pdf) for initialization rationale
    """
    return w0 * np.sqrt(6. / n)


@jax.jit
def model_fn(coords):
    layers = []
    input_dim = 2

    # Hidden Layers
    for i in range(Config.n_layers):
        w0 = Config.w0 if i == 0 else 1.
        layers.extend([
            hk.Linear(
                Config.hidden_size, 
                w_init=RandomUniform(
                    minval=-init_scale(n=input_dim, w0=w0),
                    maxval=init_scale(n=input_dim, w0=w0)
                ),
                # Not sure if biases also have to use this initialization
                b_init=RandomUniform(
                    minval=-init_scale(n=input_dim, w0=1.),
                    maxval=init_scale(n=input_dim, w0=1.)
                )
            ), 
            jnp.sin,
        ])
        input_dim = Config.hidden_size
    
    # Final down-projection
    layers.extend([
        hk.Linear(
            3,
            w_init=RandomUniform(
                minval=-init_scale(n=input_dim),
                maxval=init_scale(n=input_dim)
            ),
            b_init=RandomUniform(
                minval=-init_scale(n=input_dim),
                maxval=init_scale(n=input_dim)
            )
        )
    ])
    mlp = hk.Sequential(layers)
    reconstruction = mlp(coords.astype(np.float32))
    return reconstruction


def main(image_filepath, upsample=False, upsample_ratio=2):
    img = load_image(image_filepath)
    basename = os.path.splitext(os.path.basename(image_filepath))[0]
    key = jax.random.PRNGKey(42)

    flat_coords = np.indices(img.shape[:2]).reshape(2, -1).T
    flat_img = img.reshape(-1, 3)

    upsampled_x = int(img.shape[0] * upsample_ratio)
    upsampled_y = int(img.shape[1] * upsample_ratio)
    upsampled_indices = np.indices((upsampled_x, upsampled_y)) / upsample_ratio
    upsampled_coords = upsampled_indices.reshape(2, -1).T

    model = hk.transform(model_fn)
    params = model.init(key, flat_coords)
    
    opt = optix.adam(learning_rate=Config.lr)
    opt_state = opt.init(params)

    @jax.jit
    def loss(params, coords, rgbs):
        reconstruction = model.apply(
            params, coords
        )
        loss = np.mean((reconstruction - rgbs) ** 2)
        return loss

    @jax.jit
    def reconstruct(params, coords):
        return model.apply(
            params, coords
        )

    @jax.jit
    def update(params, opt_state, coords, rgbs):
        batch_loss, grads = jax.value_and_grad(loss)(
            params, coords, rgbs
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optix.apply_updates(params, updates)
        return new_params, opt_state, batch_loss

    for i in range(Config.n_steps):
        params, opt_state, batch_loss = update(
            params, opt_state, flat_coords, flat_img
        )

        if i % 10 == 0:
            flat_recon = reconstruct(
            	params, flat_coords
            )
            recon = np.clip( 
                np.reshape(flat_recon, (img.shape[0], img.shape[1], 3)),
                a_min=0, a_max=1
            ) 
            plt.imsave(f'results/{basename}-{i}.png', recon)

            if upsample:
                upsampled_recon = reconstruct(
            	    params, upsampled_coords
                )
                recon = np.clip( 
                    np.reshape(upsampled_recon, (upsampled_x, upsampled_y, 3)),
                    a_min=0, a_max=1
                ) 
                plt.imsave(f'upsampled_results/{basename}-{i}.png', recon)


        print(f"Loss at step {i}: {batch_loss}")


if __name__ == '__main__':
    main('data/1.jpg')

