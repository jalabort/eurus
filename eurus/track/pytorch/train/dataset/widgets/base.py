import ipywidgets as widgets

from PIL import Image

from eurus.track.pytorch.train.dataset.utils import display_image


def notebook_view_sequence(img_list, ann_list):
    r"""

    Parameters
    ----------
    img_list :

    ann_list :

    """

    def _view_image(sequence_index, index):
        img = Image.open(img_list[sequence_index][index])
        ann = ann_list[sequence_index][index]
        display_image(img, ann)

    sequence_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(img_list) - 1,
        step=1,
        description='sequence: \t',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='i',
        slider_color='white'
    )

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(img_list[0]) - 1,
        step=1,
        description='image: \t',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='i',
        slider_color='white'
    )

    def update_sequence_range(*args):
        i = sequence_slider.value
        slider.max = len(img_list[i]) - 1
        slider.value = 0

    sequence_slider.observe(update_sequence_range)

    widgets.interact(_view_image,
                     sequence_index=sequence_slider,
                     index=slider)


def notebook_view_group(img_list, ann_list):
    r"""

    Parameters
    ----------
    img_list :

    ann_list :

    """
    def _view_image(group_index, sequence_index, index):
        img = Image.open(img_list[group_index][sequence_index][index])
        ann = ann_list[group_index][sequence_index][index]
        display_image(img, ann)

    group_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(img_list) - 1,
        step=1,
        description='group:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='i',
        slider_color='white'
    )

    sequence_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(img_list[0]) - 1,
        step=1,
        description='sequence:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='i',
        slider_color='white'
    )

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(img_list[0][0]) - 1,
        step=1,
        description='image:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='i',
        slider_color='white'
    )

    def update_sequence_range(*args):
        i = group_slider.value
        j = sequence_slider.value
        slider.max = len(img_list[i][j]) - 1
        slider.value = 0

    sequence_slider.observe(update_sequence_range)

    def update_group_range(*args):
        i = group_slider.value
        j = sequence_slider.value
        sequence_slider.max = len(img_list[i]) - 1
        sequence_slider.value = 0
        slider.max = len(img_list[i][j]) - 1
        slider.value = 0

    group_slider.observe(update_group_range)

    widgets.interact(_view_image,
                     group_index=group_slider,
                     sequence_index=sequence_slider,
                     index=slider)
