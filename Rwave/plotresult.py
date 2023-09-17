import numpy as np
import matplotlib.pyplot as plt

def plotresult(result, original, maxresoln, scale=False, yaxtype="s"):
    plt.rcParams.update({'figure.figsize': (8, 6)})

    if not scale:
        # plt.subplot(maxresoln + 1, 1, 1)
        plt.plot(original)
        plt.title("Original Signal")
        plt.show()

        for j in range(maxresoln):
            # plt.subplot(maxresoln + 1, 1, j + 2)
            plt.plot(result[:, j])
            plt.title(f'Resolution {j + 1}')
            plt.show()
    else:
        ymin = np.min(result)
        ymax = np.max(result)

        # plt.subplot(maxresoln + 1, 1, 1)
        plt.plot(original)
        plt.title("Original Signal")
        plt.show()

        for j in range(maxresoln):
            # plt.subplot(maxresoln + 1, 1, j + 2)
            plt.plot(result[:, j])
            plt.ylim(ymin, ymax)
            plt.title(f'Resolution {j + 1}')
            plt.show()

    # plt.tight_layout()
    # plt.show()

# 示例用法
if __name__ == "__main__":
    # 构造一个示例的result和original
    original = np.random.rand(256)
    maxresoln = 4
    result = np.random.rand(256, maxresoln)

    plotresult(result, original, maxresoln, scale=True)
