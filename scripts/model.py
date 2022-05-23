import cv2
import regex as re
import ast
import _preprocess
import _analysis
from numpy import median
from scipy.stats import lognorm
import numpy as np
from scipy.stats import ks_2samp
from random import sample

class GrainImage:
    """Class to read and Preprocess GrainImages.

    Attributes:
        img (numpy.array): Matrix-representation of the image
                           (this one is getting pre-processed).
        origImage (numpy.array): Matrix-representation of the original image.
        len_pro_pixel (float): mm pro pixel.
        existing_areas (list): List of measured grain sizes. Empty if none were
                               found.
        generated_areas (dict): Dict of calculated grains. Initialized as
                                dict with empty list fields `size`,
                                `center`, `angle`, `major` and `minor`.
        distros (dict): Dict of distributions fit to data.
    """

    @staticmethod
    def parsenumkey(key, path_to_image):
        """Extract meta-info from `.tif`-file.

        Args:
            key (str): Label of meta-information to extract.
            fname (str): Path to file.

        Returns:
            float: Extracted meta-information.

        """
        with open(path_to_image, "rb") as f:
            for line in f:
                match = re.search(b"".join([b"^", key, b"=([-.0-9e]*)"]), line)
                if match is not None:
                    break
        return float(match.group(1))

    def __init__(self, path):
        """Init Image with old pars, image and scale.

        Args:
            path (str): Path to Image. Will try to open results-csv as well.

        """
        self.origImage = cv2.imread(path, 0)
        dbheight = int(self.parsenumkey(b"DatabarHeight", path))
        self.origImage = self.origImage[: len(self.origImage) - dbheight]
        self.modImg = self.origImage.copy()
        self.len_pro_pixel = self.parsenumkey(b"PixelWidth", path) * 10 ** 9
        self.existing_areas = {'size': [],
                               'diameter': []}
        self.generated_areas = {"pixel": [],
                                "size": [],
                                "center_x": [],
                                "center_y": [],
                                "color_variance": [],
                                "angle": [],
                                "major": [],
                                "minor": [],
                                "diameter": []}
        self.batch_stack = []
        self.distro = lognorm
        self.distros = {'existing': {},
                        'generated': {}}
        self.metric = 'euclidean'
        self.imgs = {'default_image': None,
                     'default_hist': None,
                     'modded_image': None,
                     'modded_hist': None}
        self.calcElipses = False
        self.toDisplay = 'size'
        self.max_ratio = 7
        possible_names = ["%s results.csv", "%s-results.csv", "%s_results.csv"]
        for p in possible_names:
            try:
                with open(p % path[:-4], "r") as f:
                    for line in f:
                        try:
                            size = float(line.split(",")[1])
                            self.existing_areas['size'].append(size)
                        except ValueError:
                            pass
                if median(self.existing_areas['size']) < 10 ** -2: # korrigiert csv
                    self.existing_areas['size'] = [c * (10 ** 6) for c in self.existing_areas['size']]
                self.existing_areas['diameter'] = [2 * np.sqrt(size/np.pi) for size in self.existing_areas['size']]
                self.distros['existing'] = _analysis.fit_distro(self.distro,
                                                                self.existing_areas[self.toDisplay],
                                                                self.len_pro_pixel)
                break
            except FileNotFoundError:
                continue

    @staticmethod
    def parse_batch_op(op):
        """Parse operation to usable dict

        Args:
            op (Union[dict, str]): Operation to add to batch-stack.
                                   Can be one of the two inputs:
                                   * a dictionary containing the
                                   'category' of the operation
                                   (preprocessing/analysis) as str: str-pair
                                   and the type of operation to run in the same
                                   format in addition to the arguments to use
                                   * a string in the format
                                   `<category>.<type>.(<argument>=<value>, ...)`

        Returns:
            dict: Interpretable operation-dict.

        """
        if isinstance(op, str):
            op = op.split(".")
            dummy = ".".join(op[2:])
            op = {"category": op[0], "type": op[1]}
            if len(dummy) > 4:
                to_sub = {
                    "=": '": ',
                    r"^\(": '{"',
                    r"\)$": "}",
                    r", (?=[^0-9])": ', "',
                }
                for k in to_sub:
                    dummy = re.sub(k, to_sub[k], dummy)
                dummy = ast.literal_eval(dummy)
                for k in dummy:
                    op[k] = dummy[k]
        return op

    def add_batch_op(self, op):
        """Wrapper to add operation to batch-stack

        Args:
            op (Union[dict, str]): see `parse_batch_op`

        Returns:
            type: Description of returned object.

        """
        op = self.parse_batch_op(op)
        self.batch_stack.append(op)

    def printable_batch(self, i=-1):
        """Get Ith element of batch-stack as printable string.

        Args:
            i (int): Place in the stack to get.

        Returns:
            type: Printable description of Batch operation.

        """
        op = self.batch_stack[i]
        if any([k in ["category", "type"] for k in op]):
            out = "%s.%s with the following settings:" % (op["category"], op["type"])
            for k in op:
                if k not in ["category", "type"]:
                    out += "\n"
                    out += "%s: %s" % (k, str(op[k]))
        else:
            out = "%s.%s" % (op["category"], op["type"])
        out += "\n" + "===========" + "\n"
        return out

    def run_batch(self, message_stack=None):
        """Process batch-stack"""
        try:
            self.modImg = self.origImage.copy()
            for b in self.batch_stack:
                if b["category"][0:4] == "prep":
                    dummy = {k: b[k] for k in b if k != "category"}
                    _preprocess.preprocess(self, **dummy)
                else:
                    raise ValueError("%s is not implemented yet" % b["category"])
        except KeyError as e:
            message_stack.put("%s ::: exited with an error:\n%s" % (self.batch_stack, e))

    def run_analysis(self, string, message_stack=None):
        """Process analysis call"""
        try:
            b = self.parse_batch_op(string)
            dummy = {k: b[k] for k in b if k != "category"}
            _analysis.analyse(self, **dummy)
            if message_stack is not None:
                message_stack.put("{} ::: done".format(string))
        except KeyError as e:
            if message_stack is not None:
                message_stack.put("%s ::: exited with an error:\n%s" % (string,
                                                                        e))

    def change_distro(self, distro):
        """Recalculate grain distribution"""
        self.distro = distro
        if self.existing_areas[self.toDisplay]:
            self.distros['existing'] = _analysis.fit_distro(self.distro,
                                                            self.existing_areas[self.toDisplay],
                                                            self.len_pro_pixel)
        if self.generated_areas[self.toDisplay]:
            self.distros['generated'] = _analysis.fit_distro(self.distro,
                                                             self.generated_areas[self.toDisplay],
                                                             self.len_pro_pixel)

    def calc_KS_gen(self):
        """Return KS-Distance between cdfs of generated and pre-generated grainsizes"""
        try:
            ks = ks_2samp(self.generated_areas['size'], self.existing_areas['size'])[0]
        except (ValueError, KeyError):
            print(self.generated_areas['size'], self.existing_areas['size'])
            ks = 99
        return ks

    def sample_window(self, size=1.0):
        """Cuts random area from image, good for hyper-optimization"""
        if 1 - size >  1/np.min(np.shape(self.origImage)):
            shape = np.shape(self.origImage)
            starts = [sample(range(int(s - np.ceil(s * size))),1)[0] for s in shape]
            ranges = [range(s, s + int(np.floor(shape[i] * size))) for i,s in enumerate(starts)]
            self.origImage = self.origImage[np.ix_(*ranges)]

    def set_max_ratio(self, ratio):
        """Changes maximum acceptable grain-side-ratio"""
        try:
            ratio = float(ratio)
        except ValueError:
            ratio = 9999
        self.max_ratio = ratio


if __name__ == "__main__":
    path = "../data/17 0 650 cbs 01_002.tif"
    bild = GrainImage(path)
    self = bild
