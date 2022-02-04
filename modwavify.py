"""
The modwave stores *up to* 64 waves per table
The two supported formats for individual waves are:
32-bit floating point, exactly 2048 samples long
16-bit linear data, exactly 512 samples long

Vital exports 16-bit, 1048576 bytes = 524288 samples = 256 waves x 2048 samples/wave
wavedit 1.1 exports 16-bit, 32768 bytes = 16384 samples = 64x256 samples OR 32x512 samples?
wavedit Modwave branch exports 32-bit, 524288 bytes = 131072 samples = 64x2048 samples

Author: Gary Ruben
License: MIT

"""
from pathlib import Path
import click
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def waterfall(filepath, waves):
    """Creates and saves a png thumbnail of the wavetable

    Args:
        filepath (Path): path object with png filename
        waves (ndarray): n x 64 float32 ndarray of wavetable data

    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(
        box_aspect=(1, 1, 0.1),
        yticklabels=[],
        zticklabels=[],
        yticks=[],
        zticks=[],
        xmargin=0,
        ymargin=0,
        frame_on=False,
    )
    ax.azim = 200
    ax.elev = 35
    ax.autoscale(tight=True)
    ax.grid(visible=False)
    plt.title(filepath.stem, y=0.9, fontsize=30)

    n, wavelength = waves.shape
    X, Y = np.mgrid[:n, :wavelength]

    ax.plot_wireframe(X, Y, waves, cstride=2048, lw=0.5, color='k')
    fig.subplots_adjust(top=1.1, bottom=-0.15, left=-0.1, right=1.1)
    plt.savefig(filepath)


@click.command()
@click.option('--local/--no-local', default=True, help='Normalize each wave to local (default) or global max.')
@click.option('--flip', is_flag=True, default=False, help='Reverses the wave order in the wavetable.')
@click.option('--no-thumb', is_flag=True, default=False, help='Set this to skip the thumbnail plot.')
@click.argument('filepath', type=click.Path(exists=True))
def main(filepath, local, flip, no_thumb):
    """Downsamples a 2048-sample-per-wave wavetable to be modwave compatible."""

    x, Fs = sf.read(filepath)
    print(f'{click.format_filename(filepath)}: {Fs} samples, Max excursion {np.amax(np.abs(x))}')

    path = Path(filepath)
    output_filepath = path.with_name(path.stem + '_mw' + path.suffix)

    wavelength = 2048
    # Vital wavetables contain 256 waves. The following code should always reduce this
    # by a factor of 4, but the code supports other factors for other wavetable sources
    for i in range(1, 10):
        try:
            x.shape = (-1, i, wavelength)
            if x.shape[0] <= 64:
                break
        except:
            pass

    waves, factor, _ = x.shape
    print(f"{output_filepath}: reduced by factor {factor} to {waves} waves")

    x = x.sum(axis=1)
    if flip:
        x = x[::-1]
    if local:
        max_xs = np.amax(np.abs(x), axis=1)
        x = x / max_xs[:, np.newaxis]
        waves = x[:]
        x = x.flatten()
    else:
        waves = x[:]
        x = x.flatten()
        max_x = np.amax(np.abs(x))
        x = x / max_x

    sf.write(
        output_filepath,
        x,
        samplerate=44100,
        subtype='FLOAT'
    )

    if not no_thumb:
        waterfall(path.with_name(path.stem + '.png'), waves)


if __name__ == "__main__":
    main()
