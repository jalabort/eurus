import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def crop(img, center, size):
    r"""


    Parameters
    ----------
    img : np.ndarray
        The image to be cropped.
    center: np.ndarray

    size:

    Returns
    -------
    cropped_img: PIL.Image.Image
        The cropped image.
    """
    x, y = center
    w, h = size
    x1 = int(round(x - w / 2.))
    y1 = int(round(y - h / 2.))
    return img.crop((x1, y1, x1 + w, y1 + h))


def display_image(img, box, figure_size=(16, 9), color='red', fill=False,
                  alpha=1.0):
    r"""
    Display image with overlaid bounding box.

    Parameters
    ----------
    img : np.ndarray
        ``(height, width, 3)`` array containing the image to be displayed.
    box : np.ndarray
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
    box = patches.Rectangle((box[0], box[1]), box[2], box[3],
                            color=color, fill=fill, alpha=alpha)
    ax.add_patch(box)
    plt.show()


def display_search_image(img, box, context_box, figure_size=(16, 9),
                         color_box='red', color_context='blue', fill=False,
                         alpha=1.0):
    r"""
    Display image with overlaid bounding box.

    Parameters
    ----------
    img : np.ndarray
        ``(height, width, 3)`` array containing the image to be displayed.
    box : np.ndarray
        ``(4, )`` array containing the top left vertex and the size of the
        bounding box to be displayed.
    context_box : np.ndarray

    figure_size : tuple(int), optional
        ``(2,)`` tuple defining the figure size.
    color_box : str, optional
        The color of the bounding box.
    color_context : str, optional

    fill : bool, optional
        Whether to display a filled bounding box or not.
    alpha: float, optional
        The value of the alpha channel of the bounding box.
    """
    fig, ax = plt.subplots(1, figsize=figure_size)
    ax.set_axis_off()
    ax.imshow(img)
    ax.scatter(int(img.size[1] / 2), int(img.size[0] / 2), marker='x')
    box = patches.Rectangle((box[0], box[1]), box[2], box[3],
                            color=color_box, fill=fill, alpha=alpha)
    ax.add_patch(box)
    box = patches.Rectangle((context_box[0], context_box[1]),
                            context_box[2], context_box[3],
                            color=color_context, fill=fill, alpha=alpha)
    ax.add_patch(box)
    plt.show()


# TODO: Expose more matplotlib histogram options
def display_1d_histogram(data, figure_size=(16, 9)):
    r"""
    Display the histogram of the given data.
    
    Parameters
    ----------
    data : list | np.ndarray
       ``(n,)`` list or array containing the data for which the histogram 
       needs to be generated.
    figure_size : tuple(int), optional
        ``(2,)`` tuple defining the figure size.
    """
    fig, ax = plt.subplots(1, figsize=figure_size)
    ax.hist(data, 'auto')
    plt.show()
