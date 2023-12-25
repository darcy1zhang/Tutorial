import matplotlib.pyplot as plt

def npl(nbrow):
    """
    Description:
        Create a multi-subplot figure with specified number of rows.

    Params:
        nbrow: Number of rows for the subplots.

    Returns:
        fig: The figure object.
        axes: A list of subplot axes.
    """
    fig, axes = plt.subplots(nbrow, 1, figsize=(6, 4 * nbrow))
    return fig, axes

if __name__ == "__main__":
    nbrow = 3
    fig, axes = npl(nbrow)

    for i, ax in enumerate(axes):
        ax.plot([0, 1, 2, 3], [i, i+1, i+2, i+3], label=f'Subplot {i+1}')
        ax.set_title(f'Subplot {i+1}')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()

    plt.show()

