import numpy as np
import torch


def degrees(n):
    return n * (180.0 / np.pi)


def radians(n):
    return n * (np.pi / 180.0)


def hpf_diff(x, y):
    mask1 = ((x == 0) * (y == 0)).float()
    mask1_no = 1 - mask1

    tmphp = degrees(torch.atan2(x * mask1_no, y * mask1_no))
    tmphp1 = tmphp * (tmphp >= 0).float()
    tmphp2 = (360 + tmphp) * (tmphp < 0).float()

    return tmphp1 + tmphp2


def dhpf_diff(c1, c2, h1p, h2p):

    mask1 = ((c1 * c2) == 0).float()
    mask1_no = 1 - mask1
    res1 = (h2p - h1p) * mask1_no * (torch.abs(h2p - h1p) <= 180).float()
    res2 = ((h2p - h1p) - 360) * ((h2p - h1p) > 180).float() * mask1_no
    res3 = ((h2p - h1p) + 360) * ((h2p - h1p) < -180).float() * mask1_no

    return res1 + res2 + res3


def ahpf_diff(c1, c2, h1p, h2p):

    mask1 = ((c1 * c2) == 0).float()
    mask1_no = 1 - mask1
    mask2 = (torch.abs(h2p - h1p) <= 180).float()
    mask2_no = 1 - mask2
    mask3 = (torch.abs(h2p + h1p) < 360).float()
    mask3_no = 1 - mask3

    res1 = (h1p + h2p) * mask1_no * mask2
    res2 = (h1p + h2p + 360.0) * mask1_no * mask2_no * mask3
    res3 = (h1p + h2p - 360.0) * mask1_no * mask2_no * mask3_no
    res = (res1 + res2 + res3) + (res1 + res2 + res3) * mask1
    return res * 0.5


def ciede2000_diff(lab1, lab2, device):
    """
    CIEDE2000 metric to calculate the color distance map for a batch of image tensors defined in CIELAB space
    """

    lab1 = lab1.to(device)
    lab2 = lab2.to(device)

    L1 = lab1[:, 0]
    A1 = lab1[:, 1]
    B1 = lab1[:, 2]
    L2 = lab2[:, 0]
    A2 = lab2[:, 1]
    B2 = lab2[:, 2]
    kL = 1
    kC = 1
    kH = 1

    mask_value_0_input1 = ((A1 == 0) * (B1 == 0)).float()
    mask_value_0_input2 = ((A2 == 0) * (B2 == 0)).float()
    mask_value_0_input1_no = 1 - mask_value_0_input1
    mask_value_0_input2_no = 1 - mask_value_0_input2
    B1 = B1 + 0.0001 * mask_value_0_input1
    B2 = B2 + 0.0001 * mask_value_0_input2

    C1 = torch.sqrt((A1**2.0) + (B1**2.0))
    C2 = torch.sqrt((A2**2.0) + (B2**2.0))

    aC1C2 = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - torch.sqrt((aC1C2**7.0) / ((aC1C2**7.0) + (25**7.0))))
    a1P = (1.0 + G) * A1
    a2P = (1.0 + G) * A2
    c1P = torch.sqrt((a1P**2.0) + (B1**2.0))
    c2P = torch.sqrt((a2P**2.0) + (B2**2.0))

    h1P = hpf_diff(B1, a1P)
    h2P = hpf_diff(B2, a2P)
    h1P = h1P * mask_value_0_input1_no
    h2P = h2P * mask_value_0_input2_no

    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2.0 * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.0)
    mask_0_no = 1 - torch.max(mask_value_0_input1, mask_value_0_input2)
    dHP = dHP * mask_0_no

    aL = (L1 + L2) / 2.0
    aCP = (c1P + c2P) / 2.0
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = (
        1.0
        - 0.17 * torch.cos(radians(aHP - 39))
        + 0.24 * torch.cos(radians(2.0 * aHP))
        + 0.32 * torch.cos(radians(3.0 * aHP + 6.0))
        - 0.2 * torch.cos(radians(4.0 * aHP - 63.0))
    )
    dRO = 30.0 * torch.exp(-1.0 * (((aHP - 275.0) / 25.0) ** 2.0))
    rC = torch.sqrt((aCP**7.0) / ((aCP**7.0) + (25.0**7.0)))
    sL = 1.0 + (
        (0.015 * ((aL - 50.0) ** 2.0)) / torch.sqrt(20.0 + ((aL - 50.0) ** 2.0))
    )

    sC = 1.0 + 0.045 * aCP
    sH = 1.0 + 0.015 * aCP * T
    rT = -2.0 * rC * torch.sin(radians(2.0 * dRO))

    res_square = (
        ((dLP / (sL * kL)) ** 2.0)
        + ((dCP / (sC * kC)) ** 2.0) * mask_0_no
        + ((dHP / (sH * kH)) ** 2.0) * mask_0_no
        + rT * (dCP / (sC * kC)) * (dHP / (sH * kH)) * mask_0_no
    )
    mask_0 = (res_square <= 0).float()
    mask_0_no = 1 - mask_0
    res_square = res_square + 0.0001 * mask_0
    res = torch.sqrt(res_square)
    res = res * mask_0_no

    return res
