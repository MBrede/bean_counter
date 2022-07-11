import PySimpleGUI as sg
from model import GrainImage
import _preprocess
import _analysis
import _gui_Helper as gh
import inspect
import threading
import queue
import json
import scipy.stats
import pandas as pd
from matplotlib import cm

def collapse_rows(rows):
     dummy = []
     for f in rows:
         if isinstance(f[0], list):
            dummy += f
         else:
             dummy.append(f)
     return dummy

def build_app(distros):
    precs = [f[1] for f in inspect.getmembers(_preprocess) if f[0][0:6] == "_prep_"]
    analysis_funs = [
        f[1] for f in inspect.getmembers(_analysis) if f[0][0:9] == "_analyse_"
    ]

    sg.theme("Material2")

    import_block = [
        sg.Frame("File-Handling", [[sg.Text('Import')],
                                   [sg.Input(), sg.FileBrowse()],
                                   [sg.Button("Open")],
                                   [sg.Text('Export')],
                                   gh.build_dropdown({'JSON with complete information': 'json',
                                                      'grain summary in csv': 'csv'},
                                                     'outputFormat', 'Export')])
    ]

    filter_rows = collapse_rows([gh.build_function_row(fun, "prep") for fun in precs])
    filter_block = [sg.Frame("Filtering", filter_rows, element_justification="l")]

    analysis_rows = collapse_rows([gh.build_function_row(fun, "analyse", "Run") for fun in analysis_funs])
    analysis_block = [sg.Frame("Analysis", analysis_rows, element_justification="l")]

    batch_block = [
        sg.Frame(
            "Batch",
            [
                [sg.Text("Batch stack:")],
                [sg.Multiline(s=(45, 10), key="batch_stack")],
                [
                    sg.Button("Run Batch", key="batch.run"),
                    sg.Button("Clear Batch", key="batch.clear"),
                    sg.Button("Load Batch", key="batch.load"),
                    sg.Button("Save Batch", key="batch.save"),
                ],
            ],
        )
    ]

    setting_rows = [gh.build_dropdown(distros, 'distro', 'Fit distribution'),
                    [sg.Text('Calculate ellipses'),
                     sg.Checkbox('', key='ellipseCheck', enable_events=False)],
                    gh.build_dropdown({'size': 0,
                                       'diameter': 0},
                                      'displayedStatistic', 'Change statistic'),
                    [sg.Text("Max. grain side ratio:"),
                     sg.Input('', key='max_ratio', size=(6, 10))]]

    setting_block = [sg.Frame("Settings",
                              setting_rows,
                              element_justification="l")]

    input_bar = [import_block, filter_block, batch_block, analysis_block, setting_block]

    display_bar = [
        [
            sg.Graph(
                canvas_size=(500, 500),
                graph_bottom_left=(0, 0),
                graph_top_right=(500, 500),
                key="default_image",
                tooltip="Un-modified image",
            ),
            sg.Graph(
                canvas_size=(500, 500),
                graph_bottom_left=(0, 0),
                graph_top_right=(500, 500),
                key="default_hist",
                tooltip="Manually extracted grain size (if existant)",
            ),
        ],
        [
            sg.Graph(
                canvas_size=(500, 500),
                graph_bottom_left=(0, 0),
                graph_top_right=(500, 500),
                key="modded_image",
                tooltip="Modified image",
            ),
            sg.Graph(
                canvas_size=(500, 500),
                graph_bottom_left=(0, 0),
                graph_top_right=(500, 500),
                key="modded_hist",
                tooltip="Extracted grain sizes",
            ),
        ],
    ]

    layout = [
        [
            sg.Column(input_bar, element_justification="l"),
            sg.VSeperator(),
            sg.Column(display_bar, element_justification="l"),
        ]
    ]

    window = sg.Window("Bean counter - Studio", layout)
    return window


def run_app(distros):
    viridis = cm.get_cmap('viridis', 16)
    running = False
    gui_queue = queue.Queue()
    window = build_app(distros)
    imgs = []
    while True:  # Event Loop
        event, values = window.read(timeout=100)
        if event in (sg.WIN_CLOSED, "Exit", None):
            break
        if not running:
            if event == "Open":
                try:
                    if values["Browse"][-4:] != '.tif':
                        sg.Popup("Please load tif-picture!")
                    else:
                        grain_picture = GrainImage(values["Browse"])
                        gh.refresh_images(grain_picture, window)
                except FileNotFoundError:
                    sg.Popup("File not found!")
            elif event != "__TIMEOUT__" and "grain_picture" not in locals():
                sg.Popup("Please load a picture first!")
            else:
                if event == 'do_distro_dropdown':
                    grain_picture.change_distro(distros[values["distro_dropdown"]])
                    gh.refresh_images(grain_picture, window)
                if event[0:4] == "prep":
                    try:
                        string = gh.parse_action(event, values)
                        grain_picture.add_batch_op(string)
                        window["batch_stack"].print(grain_picture.printable_batch(-1))
                    except SyntaxError as e:
                        sg.Popup("Invalid Input! Did you misstype?")
                if event[0:7] == "analyse":
                    grain_picture.run_batch()
                    gh.refresh_images(grain_picture, window, 'gray')
                    grain_picture.set_max_ratio(values['max_ratio'])
                    grain_picture.calcElipses = values["ellipseCheck"]
                    string = gh.parse_action(event, values)
                    thread_id = threading.Thread(target=grain_picture.run_analysis,
                                                 args=(string, gui_queue,),
                                                 daemon=True)
                    thread_id.start()
                    running = True
                if event == "batch.run":
                    grain_picture.run_batch()
                    gh.refresh_images(grain_picture, window, 'gray')
                if event == "batch.clear":
                    window["batch_stack"]("")
                    grain_picture.batch_stack = []
                if event == "batch.save":
                    path = gh.handle_files('batch_stack.json')
                    if path:
                        with open(path, 'w') as f:
                            json.dump(grain_picture.batch_stack, f)
                        sg.Popup('Batch saved!')
                if event == "batch.load":
                    path = gh.handle_files('', 'file')
                    if path:
                        with open(path, 'r') as f:
                            batch = json.load(f)
                        for b in batch:
                            grain_picture.add_batch_op(b)
                            window["batch_stack"].print(grain_picture.printable_batch(-1))
                if event == 'do_outputFormat_dropdown':
                    if values["outputFormat_dropdown"] == 'JSON with complete information':
                        out = {'preparation': grain_picture.batch_stack,
                               'fitted function': {'distribution': values["distro_dropdown"],
                                                   'parameters': grain_picture.distros['generated']['pars']},
                               'grains': grain_picture.generated_areas}
                        file = '%s_results.json' % values["Browse"][:-4]\
                               .split('/')[-1]
                        path = gh.handle_files(file)
                        if path:
                            with open(path, 'w') as f:
                                json.dump(out, f, indent=4)
                            sg.Popup('Results exported!')
                    else:
                        file = '%s_results.csv' % values["Browse"][:-4]\
                               .split('/')[-1]
                        path = gh.handle_files(file)
                        if path:
                            pd.DataFrame(grain_picture.generated_areas)\
                              .to_csv(path, index=False)
                            sg.Popup('Results exported!')
                if event == 'do_displayedStatistic_dropdown':
                    grain_picture.toDisplay = values['displayedStatistic_dropdown']
                    grain_picture.change_distro(grain_picture.distro)
                    gh.refresh_images(grain_picture, window)
        else:
            sg.PopupAnimated('../scripts/rsc/marvin.gif',
                             message='Processing',
                             no_titlebar=False,
                             title='Processing',
                             background_color='white',
                             time_between_frames=10)
        try:
            message = gui_queue.get_nowait()
        except queue.Empty:
            message = None
        if message is not None:
            sg.PopupAnimated(None)
            running = False
            sg.Popup(message)
            gh.refresh_images(grain_picture, window, viridis)
    window.close()

