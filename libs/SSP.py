import numpy as np


def integral_trapz(y):
    """
    trapezoidal rule to approximate a definite integral numerically
    :param y: graph of a funktion
    :return: approximated integral under graph
    """
    return (y[0] + y[-1]) / 2.0 + np.sum(y[1:-1])


def integral_trapz_2d(y):           # extra integral definieren mit sclicen durch tensor, da es trapz nicht für tf gibt
    tr1 = (y[ 0, 0] + y[0, -1] + y[-1, 0] + y[ -1, -1]) / 4.0
    tr2 = (np.sum(y[0, 1:-1]) + np.sum(y[-1, 1:-1]) + np.sum(
        y[1:-1, 0]) + np.sum(y[1:-1, -1])) / 2.0
    tr3 = np.sum(y[1:-1, 1:-1])

    return tr1 + tr2 + tr3


def SSP(y_true, y_pred):
    """
    function calculates the Surface Similarity Parameter [PerlinBustamante2015] between two 1D surfaces
    :param y_true: true surface elevation
    :param y_pred: predicted surface elevation
    :return: SSP in [0, 1], whereas 0 represents perfect agreement between y_true and y_pred
    """

    # Fourier transforms
    spec1 = np.fft.fft(y_true)
    spec2 = np.fft.fft(y_pred)

    # normalized error
    nominator = np.sqrt(integral_trapz(np.square(np.abs(spec1 - spec2))))
    denominator = np.sqrt(integral_trapz(np.square(np.abs(spec1)))) + np.sqrt(integral_trapz(np.square(np.abs(spec2))))

    SSP = np.divide(nominator, denominator)

    return SSP


def SSP_2D (y_true, y_pred):
    """
    This method returns the Surface Similarity parameter for 2-dimensional signals
    :param y_true: tensor of shape (batch, x, y)
    :param y_pred: tensor of shape (batch, x, y)
    :return: SSP
    """
    spec1 = np.fft.fft2(y_true)  # FFTs
    spec2 = np.fft.fft2(y_pred)

    nominator = np.sqrt(integral_trapz_2d(np.square(np.abs(spec1 - spec2))))
    denominator = np.sqrt(integral_trapz_2d(np.square(np.abs(spec1)))) + np.sqrt(
        integral_trapz_2d(np.square(np.abs(spec2))))

    SSP = np.divide(nominator, denominator)

    return SSP


if __name__ == "__main__":

    x = np.linspace(0, 60, 100)     # 100 x-values from 0 to 60

    # 1-D surfaces of shape (vals,) = (100,)
    f = np.sin(x)
    h = -np.sin(x)
    g = np.sin(x+0.1*np.pi)



    print(f'The SSP between slightly different surfaces is:{SSP(f, g)}')
    print(f'The SSP between different surfaces is:{SSP(f, h)}')
    print(f'The SSP between identical surfaces is:{SSP(f, f)}')
