import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from HM1_Convolve import convolve, padding
from utils import read_img, write_img


def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float)
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the
            gradient at each pixel
    """

    magnitude_grad = np.sqrt(np.square(x_grad) + np.square(y_grad))
    direction_grad = np.arctan2(y_grad, x_grad)

    return magnitude_grad, direction_grad


def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float)
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """

    h, w = grad_mag.shape

    grad_dir = np.around(grad_dir / (np.pi / 4)) * np.pi / 4

    grad_pad = np.zeros((h + 2, w + 2))
    grad_pad[1:-1, 1:-1] = grad_mag

    dx = np.int32(np.around(np.sin(grad_dir)))
    dy = np.int32(np.around(np.cos(grad_dir)))

    hidx, widx = np.meshgrid(np.arange(1, h + 1), np.arange(1, w + 1))

    pr = grad_pad[widx + dx, hidx + dy]
    pl = grad_pad[widx - dx, hidx - dy]

    nms = (grad_mag > pl) & (grad_mag > pr)

    NMS_output = np.where(nms, grad_mag, 0)

    return NMS_output


def hysteresis_thresholding(img):
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float)
        Outputs:
            output: array(float)
    """

    # you can adjust the parameters to fit your own implementation

    low_ratio = 0.10
    high_ratio = 0.40

    mask = img > low_ratio
    active = np.where(img > high_ratio, 1., 0.)

    # use convolution to disperse the active pixels
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    output = np.zeros_like(img, dtype=bool)

    while np.sum(active) > 0:
        tmp = active > 0
        output = output | tmp
        mask = mask & (~tmp)
        active = padding(active, 1, "zeroPadding")
        active = np.where((convolve(active, kernel) > 0) & mask, 1., 0.)

    return output * 1.


if __name__ == "__main__":

    # Load the input images
    input_img = read_img("lenna.png") / 255

    # Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    # Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(
        x_grad, y_grad)

    # NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    # Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)

    write_img("result/HM1_Canny_result.png", output_img * 255)
