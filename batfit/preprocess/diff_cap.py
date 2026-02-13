import numpy as np
import scipy


def grad_hand(x, y, winsize=35, poly=1, t_tmp=None):

    if len(x) > 1500:
        winsize = max(len(x) // 25, 25)
    else:
        winsize = max(len(x) // 40, 7)
    xold = x
    yold = y
    x = scipy.signal.savgol_filter(x, winsize, poly)
    y = scipy.signal.savgol_filter(y, winsize, poly)
    assert len(x) == len(y)
    assert len(x) == len(t_tmp)
    n = len(x)
    grads = np.zeros(n)
    for i in range(1, n - 1):
        if abs(x[i + 1] - x[i]) < 1e-8:
            grads[i] = np.nan
            t_tmp[i] = np.nan
        else:
            try:
                grads[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            except ZeroDivisionError:
                grads[i] = np.nan
                t_tmp[i] = np.nan
    grads[0] = np.nan
    grads[1] = np.nan
    t_tmp[0] = np.nan
    t_tmp[1] = np.nan
    grads[-1] = np.nan
    grads[-2] = np.nan
    t_tmp[-1] = np.nan
    t_tmp[-2] = np.nan

    ind = np.argwhere(~np.isnan(grads))[:, 0]
    x = x[ind]
    y = y[ind]
    grads = grads[ind]
    t_tmp = t_tmp[ind]
    ind = np.argwhere(abs(grads) < np.inf)[:, 0]
    x = x[ind]
    y = y[ind]
    grads = grads[ind]
    t_tmp = t_tmp[ind]

    # plt.plot(x,grads)
    # plt.draw()
    # plt.pause(0.1)
    # plt.close()
    # plt.show(block=False)

    # if np.amax(abs(grads)>1e6):

    return x, y, grads, t_tmp


def calc_dqdv_dvdq(t, phis_c):
    # breakpoint()
    t_new, phis_c_new, dvdq, t_tmp = grad_hand(t, phis_c, t_tmp=t)
    dqdv = 1 / dvdq
    if np.mean(dvdq) < 0:
        # Discharge
        ind = np.argwhere(abs(dvdq) < np.percentile(abs(dvdq), 95))[:, 0]
        t_new_crop = t_new[ind]
        phis_c_new_crop = phis_c_new[ind]
        dvdq_crop = dvdq[ind]
        dqdv_crop = 1 / dvdq_crop
    if np.mean(dvdq) > 0:
        # Charge
        ind = np.argwhere(abs(dvdq) < np.percentile(abs(dvdq), 95))[:, 0]
        t_new_crop = t_new[ind]
        phis_c_new_crop = phis_c_new[ind]
        dvdq_crop = dvdq[ind]
        dqdv_crop = 1 / dvdq_crop
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.plot(t_new_crop, dvdq_crop, color='b', label='smooth')
    # plt.plot(t, np.gradient(phis_c,t), color='k', label='raw')
    # plt.legend()

    # fig = plt.figure()
    # plt.plot(t_new_crop, dqdv_crop, color='b', label='smooth')
    # plt.plot(t, 1/np.gradient(phis_c,t), color='k', label='raw')
    # plt.legend()

    # fig = plt.figure()
    # plt.plot(t_new_crop, phis_c_new_crop, color='b', label='smooth')
    # plt.plot(t, phis_c, color='k', label='raw')
    # plt.legend()
    # plt.show()
    return {
        "t_diff_crop": t_new_crop,
        "phis_c_diff_crop": phis_c_new_crop,
        "dvdq_crop": dvdq_crop,
        "dqdv_crop": dqdv_crop,
        "t_diff": t_new,
        "phis_c_diff": phis_c_new,
        "dvdq": dvdq,
        "dqdv": dqdv,
    }
