import streamlit as st
from utils_app import *

st.set_page_config(
    page_title="Fresque Maker", 
    page_icon="😃", 
    layout='centered', 
    initial_sidebar_state='auto', 
    menu_items={'About': """### Fresque Maker 
    \nThis app has been created to help you make a wonderful fresque out of any picture.
    \n ---
    \n Face detection : HOG based detector from Dlib."""}
)

# TO DO : si un seul visage détecté : pas besoin de sélectionner
# symétrie à choisir
# forme du 

start_timer = start_timer()

### SIDEBAR
st.sidebar.markdown("### Options")
a = st.sidebar.slider("Fresque grid-size", min_value=1, max_value=20, value=5)

st.sidebar.markdown("### File")
uploaded_picture = st.sidebar.file_uploader("Choose a picture", type=['png', 'jpg', 'jpeg'], )

st.markdown("## Fresque Maker !")
if uploaded_picture is None :
    st.write("⟵ You can upload a picture using the sidebar :)")


else :
    # get all the faces from the picture
    detector = dlib.get_frontal_face_detector()
    image = read_image_from_streamlit_cv2(uploaded_picture)
    image_clean = image.copy()
    faces = detect_faces(image, detector)

    if len(faces) == 0 :
        caption = "No face detected..."
        st.image(image, caption)

    else :
        face_images = []
        for face_rect in faces :
            # make rectangle wider
            wider_face_rect = get_wider_rectangle(face_rect)
            face_image = image_clean[wider_face_rect.top(): wider_face_rect.bottom(), wider_face_rect.left(): wider_face_rect.right()].copy()
            face_images.append(face_image)

        st.write("Choisissez les visages à prendre en compte pour les frises :")
        images_tuple = grid_display_and_motif(face_images, 6)

        # let's make a fresque !!!
        if images_tuple :

            fresque_totale = make_fresque(image, images_tuple, a)
            st.image(fresque_totale)


st.markdown(footer,unsafe_allow_html=True)

st.write(f"Total loading time : {stop_timer(start_timer)} sec.")
