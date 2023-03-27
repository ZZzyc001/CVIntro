import numpy as np
from utils import read_img, write_img


def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """
    imgW, imgH = img.shape
    psize = padding_size

    padding_img = np.zeros((imgW + 2 * psize, imgH + 2 * psize))
    padding_img[psize:-psize, psize:-psize] = img

    if type == "zeroPadding":
        return padding_img
    elif type == "replicatePadding":
        padding_img[:, :psize] = padding_img[:, psize].reshape(-1, 1)
        padding_img[:, -psize:] = padding_img[:, -psize - 1].reshape(-1, 1)
        padding_img[:psize, :] = padding_img[psize, :].reshape(1, -1)
        padding_img[-psize:, :] = padding_img[-psize - 1, :].reshape(1, -1)
        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # zero padding
    padding_img = padding(img, 1, "zeroPadding")

    # build the Toeplitz matrix and compute convolution

    h, w = img.shape
    hk, wk = kernel.shape
    hp, wp = padding_img.shape

    tmp = np.block([kernel, np.zeros((hk, wp - wk))])
    tmp = np.concatenate([tmp.reshape(-1), [0]])
    tmp = np.concatenate([tmp] * w)[:w * hk * wp].reshape(w, hk * wp)

    tmp = np.block([tmp, np.zeros((w, (hp - hk + 1) * wp))])
    tmp = np.block([tmp] * h)[:, :w * hp * wp]
    tmp = np.stack(np.hsplit(tmp, h)).reshape(h * w, hp * wp)

    T_matrix = tmp

    output = T_matrix @ padding_img.reshape(-1)
    output = output.reshape(h, w)

    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """

    # build the sliding-window convolution here
    h, w = img.shape
    hk, wk = kernel.shape
    hr = h - hk + 1
    wr = w - wk + 1

    widx, hidx = np.meshgrid(np.arange(wr), np.arange(hr))
    widx = np.repeat(widx, hk * wk)
    hidx = np.repeat(hidx, hk * wk)
    wkidx, hkidx = np.meshgrid(np.arange(wk), np.arange(hk))
    wkidx = np.concatenate([wkidx.reshape(-1)] * hr * wr)
    hkidx = np.concatenate([hkidx.reshape(-1)] * hr * wr)

    imgr = img[hidx + hkidx, widx + wkidx].reshape((hr * wr, hk * wk))

    output = np.sum(imgr * kernel.reshape(-1), axis=1).reshape(hr, wr)

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8],
                                [1 / 16, 1 / 8, 1 / 16]])
    output = convolve(padding_img, gaussian_kernel)
    return output


def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output


def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output


if __name__ == "__main__":

    np.random.seed(111)
    input_array = np.random.rand(6, 6)
    input_kernel = np.random.rand(3, 3)

    # task1: padding
    zero_pad = padding(input_array, 1, "zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt", zero_pad)

    replicate_pad = padding(input_array, 1, "replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt", replicate_pad)

    # task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    # task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png") / 255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x * 255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y * 255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur * 255)
