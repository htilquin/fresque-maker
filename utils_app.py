import dlib
import joblib
import numpy as np
import cv2
import streamlit as st
import matplotlib.pyplot as plt

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

def grid_display_and_motif(images, columns):
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
                    chosen = st.checkbox(f"{image_number}")
                    if chosen :
                        chosen_ones.append(image_number)
                    image_number += 1
    
    if len(chosen_ones) == 0:
        st.write('Pick at least one picture !')

    else:
        height, width, _ = images[0].shape
        dimensions = (width, height)

        images_tuple = tuple(
            (
                cv2.resize(images[i], dimensions, interpolation=cv2.INTER_LINEAR)
                for i in chosen_ones
            )
        )

        return images_tuple
    
def make_fresque(image, images_tuple, a):
    motif_h = np.concatenate(images_tuple*a, axis=1)
    motif_b = np.concatenate(images_tuple[::-1]*a, axis=1)

    height_original, width_original, _ = image.shape
    height_motif_h, width_motif_h, _ = motif_h.shape
    ratio_motif_h = width_motif_h / height_motif_h
    dimensions = (width_original, round(width_original/ratio_motif_h) )
    frise_h = cv2.resize(motif_h, dimensions, interpolation=cv2.INTER_LINEAR)
    frise_b = cv2.resize(motif_b, dimensions, interpolation=cv2.INTER_LINEAR)

    fresque_haut_bas = np.concatenate((frise_h, image, frise_b), axis=0)

    height_fresque_hb, width_fresque_hb, _ = fresque_haut_bas.shape
    height_frise_h, width_frise_h, _ = frise_h.shape

    nb_motif_v = round((height_fresque_hb / height_frise_h)/len(images_tuple))

    motif_g = np.concatenate(images_tuple[::-1]*nb_motif_v, axis=0)
    motif_d = np.concatenate(images_tuple*nb_motif_v, axis=0)
    height_motif_v, width_motif_v, _ = motif_g.shape
    ratio_motif_v = width_motif_v / height_motif_v

    dimensions = (round(height_fresque_hb*ratio_motif_v), height_fresque_hb )
    frise_g = cv2.resize(motif_g, dimensions, interpolation=cv2.INTER_LINEAR)
    frise_d = cv2.resize(motif_d, dimensions, interpolation=cv2.INTER_LINEAR)

    fresque_totale = np.concatenate((frise_g, fresque_haut_bas, frise_d), axis=1)
    return fresque_totale






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