import matplotlib.pyplot as plt
import matplotlib.patches as patches


def display_image(img, ann, figure_size=(16, 9), color='red', fill=False,
                  alpha=1.0):
    r"""
    Display image with overlaid bounding box.

    Parameters
    ----------
    img : np.ndarray
        ``(height, width, 3)`` array containing the image to be displayed.
    ann : np.ndarray
        ``(4, )`` array containing the top left vertex and the size of the
        bounding box to be displayed.
    figure_size : tuple(int), optional
        ``(2,)`` tuple defining the figure size.
    color : str, optional
        The color of the bounding box.
    fill : bool, optional
        Whether to display a filled bounding box or not.
    alpha: float, optional
        The value of the alpha channel of the bounding box.
    """
    fig, ax = plt.subplots(1, figsize=figure_size)
    ax.set_axis_off()
    ax.imshow(img)
    box = patches.Rectangle((ann[0], ann[1]), ann[2], ann[3],
                            color=color, fill=fill, alpha=alpha)
    ax.add_patch(box)
    plt.show()
