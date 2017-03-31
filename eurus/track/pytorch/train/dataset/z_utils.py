import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path


def crop(img, center, size):
    x, y = center
    h, w = size
    x1 = int(round(x - w / 2.))
    y1 = int(round(y - h / 2.))
    return img.crop((x1, y1, x1 + w, y1 + h))


def view_pair(img_pair, box_pair, labels, stride):
    fig = plt.figure(figsize=(32, 24))

    ax = fig.add_subplot(131, aspect='equal')
    ax.set_axis_off()

    img = img_pair[0]
    box = box_pair[0]

    ax.imshow(img)

    verts = [
        (box[0], box[1]),
        (box[2], box[3]),
        (box[4], box[5]),
        (box[6], box[7]),
        (box[0], box[1]),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, color='blue', fill=False)
    ax.add_patch(patch)

    ax = fig.add_subplot(132, aspect='equal')
    ax.set_axis_off()

    img = img_pair[1]
    box = box_pair[1]

    ax.imshow(img)

    verts = [
        (box[0], box[1]),
        (box[2], box[3]),
        (box[4], box[5]),
        (box[6], box[7]),
        (box[0], box[1]),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, color='blue', fill=False)
    ax.add_patch(patch)

    ax = fig.add_subplot(133, aspect='equal')
    ax.set_axis_off()

    ax.imshow(labels, cmap='jet')

    verts = [
        (box[0] / stride, box[1] / stride),
        (box[2] / stride, box[3] / stride),
        (box[4] / stride, box[5] / stride),
        (box[6] / stride, box[7] / stride),
        (box[0] / stride, box[1] / stride),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY
    ]
    path = Path(verts, codes)
    patch = patches.PathPatch(path, color='blue', fill=False)
    ax.add_patch(patch)


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
