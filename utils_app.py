import dlib
import joblib
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from time import time

def start_timer():
    return time()

def stop_timer(start_time):
    return round(time()-start_time, 2)

@st.cache(suppress_st_warning=True)
def get_face_detector():
    # HOG based detector from dlib
    return dlib.get_frontal_face_detector()

def detect_faces(image, detector) :
    face_rectangles = detector(image, 1)
    return face_rectangles

def get_wider_rectangle(rect):
    width = rect.right() - rect.left()
    height = rect.bottom() - rect.top()
    new_rect = dlib.rectangle(
        max(0, int(rect.left() - .05*width)), 
        max(0, int(rect.top() - 0.2*height)), 
        int(rect.right() + .05*width), 
        int(rect.bottom() + 0.2*height))
    return new_rect

def read_image_from_streamlit_cv2(streamlit_image):
    image = cv2.imdecode(np.frombuffer(streamlit_image.read(), np.uint8), 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def choose_faces(images, columns):
    image_number = 0
    chosen_ones = []
    for _ in range(int(len(images)/columns)+1) :
        cols = st.columns(columns)
        for column in range(0,columns):
            if image_number >= len(images) :
                break
            else :
                with cols[column]:
                    st.image(images[image_number], use_column_width=True)
                    chosen = st.checkbox(f"{image_number}", value=True)
                    if chosen :
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
                cv2.flip(cv2.resize(images[i], dimensions, interpolation=cv2.INTER_LINEAR), 1)
                for i in chosen_ones#[::-1]
            )
        )

        images_frieze = images_frieze + images_frieze_flipped

    return images_frieze
    
def add_frame(image, images_frieze, repetition):
    motif_top = np.concatenate(images_frieze*repetition, axis=1)
    motif_b = np.concatenate(images_frieze[::-1]*repetition, axis=1)

    height_original, width_original, _ = image.shape
    height_motif_top, width_motif_top, _ = motif_top.shape
    ratio_motif_top = width_motif_top / height_motif_top
    
    dimensions = (width_original, round(width_original/ratio_motif_top) )
    frise_top = cv2.resize(motif_top, dimensions, interpolation=cv2.INTER_LINEAR)
    frise_b = cv2.resize(motif_b, dimensions, interpolation=cv2.INTER_LINEAR)

    fresque_top_bottom = np.concatenate((frise_top, image, frise_b), axis=0)

    height_fresque_tb, width_fresque_tb, _ = fresque_top_bottom.shape
    height_frise_top, width_frise_top, _ = frise_top.shape

    nb_motif_v = round((height_fresque_tb / height_frise_top)/len(images_frieze))

    motif_left = np.concatenate(images_frieze[::-1]*nb_motif_v, axis=0)
    motif_right = np.concatenate(images_frieze*nb_motif_v, axis=0)

    height_motif_v, width_motif_v, _ = motif_left.shape
    ratio_motif_v = width_motif_v / height_motif_v

    dimensions = (round(height_fresque_tb*ratio_motif_v), height_fresque_tb )
    frise_left = cv2.resize(motif_left, dimensions, interpolation=cv2.INTER_LINEAR)
    frise_right = cv2.resize(motif_right, dimensions, interpolation=cv2.INTER_LINEAR)

    fresque_totale = np.concatenate((frise_left, fresque_top_bottom, frise_right), axis=1)
    return fresque_totale


def add_frame_round(image, repetition):
    height, width, _ = image.shape
    mini_dimension = min(height, width)
    image_square = image[0:mini_dimension, 0:mini_dimension]
    lum_img = Image.new('L', [mini_dimension, mini_dimension], 0)

    draw = ImageDraw.Draw(lum_img)
    draw.pieslice([(0,0), (mini_dimension, mini_dimension)], 0, 360,
                    fill = 255, outline = "white")
    
    img_arr = np.array(image_square)
    lum_img_arr = np.array(lum_img)
    final_img_arr = np.dstack((img_arr, lum_img_arr))

    return final_img_arr





footer="""<style>
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