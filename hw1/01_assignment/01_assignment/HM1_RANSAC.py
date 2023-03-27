import numpy as np
from utils import draw_save_plane_with_points

if __name__ == "__main__":
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    noise_points = np.loadtxt("HM1_ransac_points.txt")

    # RANSAC
    # we recommend you to formulate the palnace function as:
    # A*x+B*y+C*z+D=0

    # calculate lsm
    def lsm(points):
        avg = np.mean(points, axis=1, keepdims=True)
        points_ = points - avg
        _, _, v = np.linalg.svd(points_)
        norm = v[list(range(points.shape[0])), -1, :]
        bias = -np.sum(norm * np.mean(points, axis=1), axis=1, keepdims=True)
        return np.concatenate([norm, bias], axis=1)

    # (1 - (100 / 130) ** 3) ** 12 < 0.001
    sample_time = 12
    # more than 99.9% probability at least one hypothesis
    # does not contain any outliers

    distance_threshold = 0.05

    # sample points group

    idx = np.random.choice(noise_points.shape[0], (sample_time, 3),
                           replacement=False)

    # estimate the plane with sampled points group

    pfs = lsm(noise_points[idx, :]).reshape(sample_time, 4, 1)

    # evaluate inliers (with point-to-plance distance < distance_threshold)

    dis = np.abs(noise_points @ pfs[:, :3, :] + pfs[:, 3:, :])
    valid = dis.reshape(sample_time, -1) < distance_threshold

    cnt = np.sum(valid, axis=1)

    inliers = np.expand_dims(noise_points[valid[np.argmax(cnt), :]], axis=0)

    # minimize the sum of squared perpendicular distances of all inliers
    # with least-squared method

    pf = lsm(inliers).reshape(-1)

    # draw the estimated plane with points and save the results
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of
    # palnace function  A*x+B*y+C*z+D=0
    draw_save_plane_with_points(pf, noise_points, "result/HM1_RANSAC_fig.png")
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
