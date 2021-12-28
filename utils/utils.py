from scipy.interpolate import UnivariateSpline


def get_lut(x, y):
    spline_interpolate = UnivariateSpline(x, y)
    return spline_interpolate(range(256))