import gui
import scipy.stats

if __name__ == "__main__":
    distros = {'lognorm': scipy.stats.lognorm,
               'norm': scipy.stats.norm,
               'f': scipy.stats.f,
               'uniform': scipy.stats.uniform}
    gui.run_app(distros)
