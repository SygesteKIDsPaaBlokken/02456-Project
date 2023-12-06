import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.evaluation import ScoringEvaluation

# For the radar chart
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

DIR_PATH = 'data'
FILE_NAME = 'ft_RankingResults.csv'
TEST_SET = '0-3scoringTestSet.txt'

qrels = pd.read_csv(DIR_PATH + '/' + TEST_SET, sep=' ')
ft = pd.read_csv(DIR_PATH + '/' + 'ft_RankingResults.csv', index_col=0)
topK = [1, 2, 3, 4, 5, 10, ]




def load_data(ranking_paths):
    for path in ranking_paths:
        yield pd.read_csv(DIR_PATH + '/' + path + '.csv', index_col=0)


def render_bar_chart(metric:str='score',*RankingResultsFileNames:str):
    """
    Plots bar chart
    """
    print(RankingResultsFileNames)
    models = [df for df in load_data(RankingResultsFileNames)]
    models_count = len(models)

    x = np.arange(len(topK))

    width=0.2

    bar_positions = np.arange(len(topK)) - width * (models_count - 1) / 2

    for i, model in enumerate(models):
        plt.bar(bar_positions + i * width, model[metric]/model['max_'+metric]*100, width=width)

    plt.ylim(0,100)
    plt.xticks(x, map(str,topK)) 
    plt.title(f'{metric.capitalize()} for top K evaluations across models')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Top K') 
    plt.legend([*RankingResultsFileNames]) 
    plt.show()


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


if __name__ == '__main__':
    render_bar_chart('score','Fasttext', 'fuzzy', 'sbert_1epoch')
    render_bar_chart('count','Fasttext', 'fuzzy', 'sbert_1epoch')
    

    ft_evaluation = pd.read_csv(DIR_PATH + '/' + 'Fasttext.csv')
    fz_evaluation = pd.read_csv(DIR_PATH + '/' + 'fuzzy.csv')
    sbert_evaluation = pd.read_csv(DIR_PATH + '/' + 'sbert_1epoch' + '.csv') # 1epoch

    N = len(topK)
    theta = radar_factory(N, frame='polygon')

    data = [
        list(map(str,topK)),
        ('Score', [
            (ft_evaluation['score']/ft_evaluation['max_score']*100).to_list(),
            (fz_evaluation['score']/fz_evaluation['max_score']*100).to_list(),
            (sbert_evaluation['score']/sbert_evaluation['max_score']*100).to_list()]),
        ('Count', [
            (ft_evaluation['count']/ft_evaluation['max_count']*100).to_list(),
            (fz_evaluation['count']/fz_evaluation['max_count']*100).to_list(),
            (sbert_evaluation['count']/sbert_evaluation['max_count']*100).to_list()]),
    ]

    spoke_labels = data.pop(0)

    fig, axs = plt.subplots(figsize=(9, 9), nrows=1, ncols=2,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    colors = ['r','g','b']#, 'r', 'g', 'm', 'y']
    labels = ('FastText','Fuzzy','SBERT')
    
    
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_rgrids(np.round(np.linspace(0,np.max([np.max(scores) for scores in case_data]), 5 ), decimals=1))
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)


    legend = axs[0].legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')
    fig.text(0.5, 0.965, 'Score and count for top K evaluations across models',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()