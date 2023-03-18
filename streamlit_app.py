import streamlit as st
import numpy as np
from utils_app import *


st.set_page_config(
    page_title="Fresco Maker", 
    page_icon="🖼️", 
    layout='centered', 
    initial_sidebar_state='auto', 
    menu_items={'About': """### Fresco Maker 
    \nThis app has been created to help you make a wonderful fresco out of any picture.
    \n ---
    \n Face detection : HOG based detector from Dlib."""}
)

# TO DO : si un seul visage détecté : pas besoin de sélectionner
# forme du cadre (rond)
# recadrage de la photo avant
# sélection d'un autre visage non détecté

start_timer = start_timer()

### SIDEBAR
st.sidebar.markdown("# File")
uploaded_picture = st.sidebar.file_uploader("Choose a picture", type=['png', 'jpg', 'jpeg'], )

st.markdown("## Fresco Maker !")
if uploaded_picture is None :
    st.write("⟵ Upload a picture using the sidebar :)")

else :
    # get all the faces from the picture
    detector = dlib.get_frontal_face_detector()
    image = read_image_from_streamlit_cv2(uploaded_picture)
    image_clean = image.copy()
    faces = detect_faces(image, detector)

    full_pic = st.checkbox("Use full picture ?")

    if not full_pic and len(faces) == 0 :
        caption = "No face detected..."
        st.image(image, caption)

    else :
        if full_pic:
            face_images = [image_clean]
            chosen_ones = [0]
        else:
            face_images = []
            for face_rect in faces :
                # make rectangle wider
                wider_face_rect = get_wider_rectangle(face_rect)
                face_image = image_clean[wider_face_rect.top(): wider_face_rect.bottom(), wider_face_rect.left(): wider_face_rect.right()].copy()
                face_images.append(face_image)

            st.write("Pick or unpick the faces you want to see in the frame:")
            chosen_ones = choose_faces(face_images, 6)

            if len(chosen_ones) == 0:
                st.write('Pick at least one picture !')

        st.sidebar.markdown("# Options")
        symetry = st.sidebar.checkbox('Use the power of symetry')

        images_frieze = motif_maker(face_images, chosen_ones, symetry)

        st.write("⟵ Twick the options using the sidebar :)")

        repetition = st.sidebar.slider("Times the frieze is repeted in the 'frame':", min_value=1, max_value=20, value=5)
        make_round = st.sidebar.checkbox('Make it round !')

        # let's make a fresque !!!
        if images_frieze :
            if make_round:
                st.write('Soooooooooooonnnnnnnnnnnnnnnnnnn')
                fresque_totale = add_frame_round(image, repetition)

            if not make_round :
                fresque_totale = add_frame(image, images_frieze, repetition)
            st.image(fresque_totale)


st.markdown(footer,unsafe_allow_html=True)

st.write(f"Total loading time : {stop_timer(start_timer)} sec.")