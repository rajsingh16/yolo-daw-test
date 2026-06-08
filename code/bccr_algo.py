import cv2
import numpy as np
import torch
from guided_filter_pytorch.guided_filter import GuidedFilter


def dark_channel(image, size=15):
    b, g, r = cv2.split(image)
    min_img = cv2.min(cv2.min(r, g), b)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (size, size)
    )

    return cv2.erode(min_img, kernel)


def atmospheric_light(image, dark_channel_img):

    h, w = image.shape[:2]

    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))

    dark_vec = dark_channel_img.ravel()
    image_vec = image.reshape(num_pixels, 3)

    indices = dark_vec.argsort()[::-1][:num_brightest]

    return np.mean(image_vec[indices], axis=0)


def transmission_estimate(
        image,
        atmo_light,
        size=15,
        omega=0.95):

    norm_image = image / atmo_light

    return 1 - omega * dark_channel(
        norm_image,
        size
    )


def boundary_constrained_context_regularization(
        trans_map,
        image,
        alpha=0.5,
        beta=0.85):

    trans_tensor = torch.tensor(
        trans_map
    ).unsqueeze(0).unsqueeze(0).float()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    gray_tensor = torch.tensor(
        gray
    ).unsqueeze(0).unsqueeze(0).float() / 255.0

    guided_filter = GuidedFilter(
        r=60,
        eps=1e-3
    )

    refined = guided_filter(
        gray_tensor,
        trans_tensor
    )

    refined = refined.squeeze().numpy()

    refined = (
        alpha * trans_map +
        (1 - alpha) * refined
    )

    refined = np.clip(
        refined,
        beta,
        1
    )

    return refined


def recover(
        image,
        trans_map,
        atmo_light,
        t0=0.1):

    trans_map = np.maximum(
        trans_map,
        t0
    )

    result = np.empty_like(image)

    for i in range(3):
        result[:, :, i] = (
            (image[:, :, i] - atmo_light[i])
            / trans_map
            + atmo_light[i]
        )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def dehaze(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    dark = dark_channel(image_rgb)

    A = atmospheric_light(
        image_rgb,
        dark
    )

    t = transmission_estimate(
        image_rgb,
        A
    )

    t_refined = boundary_constrained_context_regularization(
        t,
        image_rgb
    )

    output = recover(
        image_rgb,
        t_refined,
        A
    )

    return cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )
