import seaborn as sns
import matplotlib.pyplot as plt

def draw_heatmap(x, xlabels, ylabels, x_top=False):
    """
    draw matrix heatmap with matplotlib
    :param x:
    :param xlabels:
    :param ylabels:
    :param x_top:
    :return:
    """
    # Plot it out
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(x, cmap=plt.cm.Blues, alpha=0.8)

    # Format
    fig = plt.gcf()
    fig.set_size_inches(8, 11)

    # turn off the frame
    ax.set_frame_on(False)
    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(x.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(x.shape[1]) + 0.5, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    if x_top:
        ax.xaxis.tick_top()

    ax.set_xticklabels(xlabels, minor=False)
    ax.set_yticklabels(ylabels, minor=False)

    # rotate the
    plt.xticks(rotation=90)

    ax.grid(False)

    # Turn off all the ticks
    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False


def draw_heatmap_sea(x, xlabels, ylabels, answer, save_path, inches=(11, 3), bottom=0.45, linewidths=0.2):
    """
    draw matrix heatmap with seaborn
    :param x:
    :param xlabels:
    :param ylabels:
    :param answer:
    :param save_path:
    :param inches:
    :param bottom:
    :param linewidths:
    :return:
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=bottom)
    plt.title('Answer: ' + answer)
    sns.heatmap(x, linewidths=linewidths, ax=ax, cmap='Blues', xticklabels=xlabels, yticklabels=ylabels)
    fig.set_size_inches(inches)
    fig.savefig(save_path)
