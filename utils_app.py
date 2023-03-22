import cv2
import dlib
import math
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from time import time


def start_timer():
    return time()


def stop_timer(start_time):
    return round(time() - start_time, 2)


@st.cache(suppress_st_warning=True)
def get_face_detector():
    # HOG based detector from dlib
    return dlib.get_frontal_face_detector()


def detect_faces(image, detector):
    face_rectangles = detector(image, 1)
    return face_rectangles


def get_wider_rectangle(rect):
    width = rect.right() - rect.left()
    height = rect.bottom() - rect.top()
    new_rect = dlib.rectangle(
        max(0, int(rect.left() - 0.05 * width)),
        max(0, int(rect.top() - 0.2 * height)),
        int(rect.right() + 0.05 * width),
        int(rect.bottom() + 0.2 * height),
    )
    return new_rect


def read_image_from_streamlit_cv2(streamlit_image):
    image = cv2.imdecode(np.frombuffer(streamlit_image.read(), np.uint8), 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def choose_faces(images, columns):
    image_number = 0
    chosen_ones = []
    for _ in range(int(len(images) / columns) + 1):
        cols = st.columns(columns)
        for column in range(0, columns):
            if image_number >= len(images):
                break
            else:
                with cols[column]:
                    st.image(images[image_number], use_column_width=True)
                    chosen = st.checkbox(f"{image_number}", value=True)
                    if chosen:
                        chosen_ones.append(image_number)
                    image_number += 1

    return chosen_ones


def motif_maker(images, chosen_ones, symetry):
    height, width, _ = images[0].shape
    dimensions = (width, height)

    images_frieze = tuple(
        (
            cv2.resize(images[i], dimensions, interpolation=cv2.INTER_LINEAR)
            for i in chosen_ones
        )
    )

    if symetry:
        images_frieze_flipped = tuple(
            (
                cv2.flip(
                    cv2.resize(images[i], dimensions, interpolation=cv2.INTER_LINEAR), 1
                )
                for i in chosen_ones  # [::-1]
            )
        )

        images_frieze = images_frieze + images_frieze_flipped

    return images_frieze


def add_frame(image, images_frieze, repetition):
    motif_top = np.concatenate(images_frieze * repetition, axis=1)
    motif_b = np.concatenate(images_frieze[::-1] * repetition, axis=1)

    height_original, width_original, _ = image.shape
    height_motif_top, width_motif_top, _ = motif_top.shape
    ratio_motif_top = width_motif_top / height_motif_top

    dimensions = (width_original, round(width_original / ratio_motif_top))
    frise_top = cv2.resize(motif_top, dimensions, interpolation=cv2.INTER_LINEAR)
    frise_b = cv2.resize(motif_b, dimensions, interpolation=cv2.INTER_LINEAR)

    fresque_top_bottom = np.concatenate((frise_top, image, frise_b), axis=0)

    height_fresque_tb, width_fresque_tb, _ = fresque_top_bottom.shape
    height_frise_top, width_frise_top, _ = frise_top.shape

    nb_motif_v = round((height_fresque_tb / height_frise_top) / len(images_frieze))

    motif_left = np.concatenate(images_frieze[::-1] * nb_motif_v, axis=0)
    motif_right = np.concatenate(images_frieze * nb_motif_v, axis=0)

    height_motif_v, width_motif_v, _ = motif_left.shape
    ratio_motif_v = width_motif_v / height_motif_v

    dimensions = (round(height_fresque_tb * ratio_motif_v), height_fresque_tb)
    frise_left = cv2.resize(motif_left, dimensions, interpolation=cv2.INTER_LINEAR)
    frise_right = cv2.resize(motif_right, dimensions, interpolation=cv2.INTER_LINEAR)

    fresque_totale = np.concatenate(
        (frise_left, fresque_top_bottom, frise_right), axis=1
    )
    return fresque_totale


def image_to_square(image):
    height, width, _ = image.shape
    square_dim = min(height, width)

    image_square = image[
        int((height - square_dim) / 2) : int(height - (height - square_dim) / 2),
        int((width - square_dim) / 2) : int(width - (width - square_dim) / 2),
    ]

    return image_square, square_dim


def calculate_circle_points(r, nb_pts, coeff, center, mini_h, mini_w):
    points = []
    for i in range(nb_pts):
        points.append(
            [
                int(coeff * r * math.sin((i * 2 * math.pi) / nb_pts)) + center[1],
                int(coeff * r * math.cos((i * 2 * math.pi) / nb_pts)) + center[1],
            ]
        )
    return points


def add_frame_round(image, images_frieze, repetition):
    image_square, square_dim = image_to_square(image)

    mini_h, mini_w, _ = images_frieze[0].shape
    big_dim = square_dim + mini_h * 2

    background = Image.new("RGB", (big_dim, big_dim), "white")

    offset = ((big_dim - square_dim) // 2, (big_dim - square_dim) // 2)
    background.paste(Image.fromarray(image_square), offset)

    center = (offset[0] + square_dim // 2, offset[1] + square_dim // 2)

    angles = [int(i / repetition * 360) for i in range(repetition)]
    rotate = st.sidebar.checkbox("Rotate pictures in frame ?", value=True)
    coeff = st.sidebar.slider("Size of circle", 0, 100, 75)
    size = st.sidebar.slider(
        "Size of the picture", min_value=10, max_value=100, value=85
    )
    radius = (big_dim) // 2
    width = st.sidebar.slider(
        "Horizontal shifting", min_value=-100, max_value=100, value=0
    )
    height = st.sidebar.slider(
        "Vertical shifting", min_value=-100, max_value=100, value=0
    )
    coordinates = calculate_circle_points(
        radius, repetition, coeff / 100, center, mini_h // 100, size * mini_w // 100
    )

    friezes = [
        Image.fromarray(images_frieze[i % len(images_frieze)]).convert("RGBA")
        for i in range(repetition)
    ]

    for i in range(repetition):
        img = friezes[i].resize(
            (size * mini_w // 100, size * mini_h // 100), Image.LANCZOS
        )
        if rotate:
            img = img.rotate(angles[i], expand=True)
        coordinates_modif = (
            coordinates[i][0] + width * 5,
            coordinates[i][1] + height * 5,
        )
        background.paste(img, coordinates_modif, img)

    lum_img = Image.new("L", [big_dim, big_dim], 0)

    draw = ImageDraw.Draw(lum_img)
    draw.pieslice(
        [(0, 0), (big_dim, big_dim)],
        0,
        360,
        fill=255,
        # outline="white"
    )

    img_arr = np.array(background)
    lum_img_arr = np.array(lum_img)
    round_img_arr = np.dstack((img_arr, lum_img_arr))

    return round_img_arr


footer = """<style>
a:link , a:visited{
color: red;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: gray;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with 💖 by <a style='display: block; text-align: center;' href="https://htilquin.github.io/" target="_blank">Hélène T.</a></p>
</div>
"""
