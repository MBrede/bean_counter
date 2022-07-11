import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import inspect
import os
import numpy as np


def build_function_row(fun, prefix, button_label="Add"):
    """Build a GUI-row for a given function.

    This function uses a functions docs to build a PySimpleGUI-row
    with all necessary arguments.
    Each row starts with a button.

    Args:
        fun (function): function to create the row for.
        prefix (str): prefix to add to the button key. Necessary for the GUI-loop.
        button_label (str): label of the button at the row's start.

    Returns:
        list: list of PySimpleGUI elements to be treated as a GUI-row
    """
    row = []
    label = inspect.getdoc(fun).split(":")[0]
    row.append(sg.Text("%s:" % label, size=(18, 1)))
    args = dict(inspect.signature(fun)._parameters)
    row.append(
        sg.Button(button_label, key="%s.%s" % (prefix, fun.__name__.split("_")[2]))
    )
    for arg in args:
        if arg != "self":
            arg = str(args[arg]).split("=")
            row.append(sg.Text(arg[0], size=(12, 1)))
            if len(arg) > 1:
                value = arg[1]
            else:
                value = ""
            row.append(
                sg.Input(
                    value,
                    key="%s.%s:%s" % (prefix, fun.__name__.split("_")[2], arg[0]),
                    size=(6, 10),
                )
            )
    if len(row) > 8:
        dummy = [[] if i == 0 else [sg.Text("", size=(27, 1))] 
                 for i in range(len(row) // 8 + int(len(row) % 8 > 0))]
        for i in range(len(row)):
            dummy[i // 8].append(row[i])
        row = dummy
    return row


def build_dropdown(options, prefix, button_label="Run"):
    """Build GUI-row containing a Dropdown selection.

    Builds a GUI row with a dropdown selection, especially usefull for
    Model selections.

    Args:
        options (dict): Options for the dropdown-menu.
        prefix (str): prefix to add to the button key. Necessary for the GUI-loop.
        button_label (str): label of the button at the row's start.

    Returns:
        list: list of PySimpleGUI elements to be treated as a GUI-row

    """
    row = []
    row.append(sg.Combo(list(options.keys()),
                        size=(30, 5),
                        enable_events=False,
                        key='%s_dropdown' % prefix,
                        default_value=list(options.keys())[0]))
    if button_label:
        row.append(sg.Button(button_label, key='do_%s_dropdown' % (prefix)))
    return(row)


def render_figure(img, cm="gray"):
    """Renders a grain picture.

    Uses the luminocity values of the grain picture to render a 2D image.

    Args:
        img (numpy.array): Matrix of luminocity values.
        cm (str): Color mapping.

    Returns:
        matplotlib.figure.Figure: rendered grain image
    """
    fig = plt.figure(figsize=(10, 10), dpi=50)
    im = plt.imshow(img, cmap=cm, aspect="equal")
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    return [fig, im]


def update_figure(fig, img, cm=None):
    """
    Update grain-figure.

    Updates the grain depiction with new data or color scheme if
    it already exists, initializes it otherwise.

    Args:
        fig (matplotlib.pyplot.figure): Figure to update or None if a new figure is to be created.
        img (numpy.array): Values for the figure
        cm (str): colormap to use.

    Returns:
        matplotlib.pyplot.figure: updated figure

    """

    if fig is not None:
        if cm is not None:
            if cm != 'gray':
                img = np.multiply(img % 16, 16)
            fig[1].set_cmap(cm)
        elif fig[1].get_cmap().name != 'gray':
            img = np.multiply(img % 16, 16)
        fig[1].set_data(img)
    else:
        cm = 'gray'
        fig = render_figure(img, cm)
    return fig


def render_histogram(data, dists, x_text='Area in nm²'):
    """Renders a histogram.

    Uses the data given to render a histogram.

    Args:
        data (numpy.array): data to display.

    Returns:
        matplotlib.figure.Figure: rendered histogram.

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), dpi=50)
    if not data:
        return [fig]

    count, bins, _ = ax[0].hist(data, 30, density=True)

    # histogram with both pdfs
    x = np.linspace(0, np.max(bins), 1000)
    ax[0].set_xlabel(x_text)
    ax[0].set_ylabel('Density')
    for dist in dists:
        ax[0].plot(x, dists[dist]['function'](x), lw=2, label=dist)
    ax[0].legend()

    # cdf-diagram
    for dist in dists:
        s = np.not_equal(dists[dist]['emp cdf'][1]-
                         np.roll(dists[dist]['emp cdf'][1],1),
                         0) * 3
        ax[1].scatter(dists[dist]['emp cdf'][0][np.where(s)],
                      dists[dist]['emp cdf'][1][np.where(s)],
                      label=dist)
        xmax = np.roll(dists[dist]['emp cdf'][0][np.where(s)], 1)
        xmax[0] = 0
        ax[1].hlines(y=dists[dist]['emp cdf'][1][np.where(s)],
                     xmin=dists[dist]['emp cdf'][0][np.where(s)],
                     xmax=xmax)
        theor_dist = dists[dist]['dist'].cdf(x, **dists[dist]['pars'])
        ax[1].plot(x, theor_dist, lw=2, label=dist, alpha=.5)
    ax[1].legend()
    ax[1].set_xlabel(x_text)
    ax[1].set_ylabel('P(X<=x)')
    ax[1].set_xlim(0, np.max(data) * 1.1)

    # KS-Distance in cdf:
    ks = np.argmax(np.abs(dists[dist]['dist'].cdf(dists[dist]['emp cdf'][0],
                                                  **dists[dist]['pars']) -
                          dists[dist]['emp cdf'][1]))
    ax[1].vlines(x=dists[dist]['emp cdf'][0][ks],
                 ymin=dists[dist]['dist'].cdf(dists[dist]['emp cdf'][0][ks],
                                              **dists[dist]['pars']),
                 ymax=dists[dist]['emp cdf'][1][ks])
    ax[1].text(x=1.1 * dists[dist]['emp cdf'][0][ks],
               y=np.mean([dists[dist]['dist'].cdf(dists[dist]['emp cdf'][0][ks],
                                                  **dists[dist]['pars']),
                          dists[dist]['emp cdf'][1][ks]]),
               s='D = %f' % np.max(np.abs(dists[dist]['dist']
                                          .cdf(dists[dist]['emp cdf'][0],
                                               **dists[dist]['pars'])
                                          - dists[dist]['emp cdf'][1])))

    plt.tight_layout()
    return([fig])


def draw_figure(canvas, figure):
    """Draws a figure.

    Adds a histogram or grain image to a canvas.

    Args:
        canvas (tkinter.Canvas): canvas to draw on
        figure (matplotlib.figure.Figure): figure to draw

    Returns:
        tkinter.Canvas: canvas with added figure.
    """
    if canvas.children:
        for child in canvas.winfo_children():
            child.forget()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.get_tk_widget().pack()
    figure_canvas_agg.get_tk_widget().config(width=500, height=500)
    figure_canvas_agg.draw()
    return figure_canvas_agg


def refresh_images(grain_picture, window, cm=None):
    """Refreshes GUI-Canvasses

    Deletes all existing plots, resets the canvases and
    redraws all images.

    Args:
        grain_picture (GrainImage): grain image model
        window (PySimpleGUI.Window): Window to update.
        imgs (list): Optional list of existing images.

    """

    grain_picture.imgs['modded_image'] = update_figure(grain_picture.imgs['modded_image'],
                                                       grain_picture.modImg,
                                                       cm)

    grain_picture.imgs['default_image'] = update_figure(grain_picture.imgs['default_image'],
                                                        grain_picture.origImage)
    if grain_picture.toDisplay == 'diameter':
        x_text = 'Approximated diameter in nm'
    else:
        x_text = 'Area in nm²'

    if grain_picture.existing_areas[grain_picture.toDisplay]:
        try:
            distros = {'existing': grain_picture.distros['existing']}
        except KeyError:
            distros = {}
        grain_picture.imgs['default_hist'] = render_histogram(grain_picture.existing_areas[grain_picture.toDisplay],
                                                              distros,x_text)
    else:
        grain_picture.imgs['default_hist'] = render_histogram([],
                                                              {},x_text)

    if grain_picture.generated_areas[grain_picture.toDisplay]:
        distros = {k:grain_picture.distros[k] for k in grain_picture.distros
                   if 'dist' in grain_picture.distros[k].keys()}
        grain_picture.imgs['modded_hist'] = render_histogram(grain_picture.generated_areas[grain_picture.toDisplay],
                                                             distros, x_text)
    else:
       grain_picture.imgs['modded_hist'] = render_histogram([],
                                                            {},x_text)

    for k in grain_picture.imgs:
        if grain_picture.imgs[k] is not None:
            draw_figure(window[k].TKCanvas, grain_picture.imgs[k][0])
            window[k].set_size((500, 500))

    return None


def handle_files(file_name='', save_type='folder'):
    """Path finder.

    Opens a file handling dialogue to help the user find relevant paths.

    Args:
        file_name (str): name of file to add to path.
        save_type (str): choose folder or file?

    Returns:
        str: selected path
    """
    try:
        if save_type == 'folder':
            browser = sg.FolderBrowse(key="-IN-")
        else:
            browser = sg.FileBrowse(key="-IN-")
        layout = [[sg.T("")], [sg.Text("Choose a %s: " % save_type),
                               sg.Input(key="-IN2-"),
                               browser],
                  [sg.Button('Submit', key="Submit")]]
        dialogue = sg.Window('I/O Dialogue', layout, size=(600, 150))

        while True:
            event, values = dialogue.read()
            if event == sg.WIN_CLOSED or event == "Exit":
                break
            elif event == 'Submit':
                break
        dialogue.close()
        if file_name:
            return(os.path.join(values["-IN-"], file_name))
        else:
            return(values["-IN-"])
    except TypeError:
        return None


def parse_action(event, values):
    """Parser for GUI events.

    Turns GUI events into interpretable actions.

    Args:
        event (str): Name of the button pressed.
        values (dict): dictionary of values in GUI.

    Returns:
        str: parsed action

    """
    string = "%s.(" % event
    for k in values:
        if isinstance(k, str):
            if k.split(":")[0] == event:
                if string[-1] != "(":
                    string += ", "
                string += k.split(":")[1]
                string += "=" + values[k]
    string += ")"
    return string
