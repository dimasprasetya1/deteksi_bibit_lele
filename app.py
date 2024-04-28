import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from tempfile import NamedTemporaryFile
from scipy import ndimage

# Fungsi untuk memuat template
def load_templates(template_dir='edited/'):
    templates = []
    for template_file in os.listdir(template_dir):
        if template_file.endswith(".png"):
            template_bgr = cv2.imread(os.path.join(template_dir, template_file), cv2.IMREAD_COLOR)
            template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
            template_gray_resized = cv2.resize(template_gray, (64, 128), interpolation=cv2.INTER_AREA)
            templates.append(template_gray_resized)
    return templates

# Fungsi utama untuk memproses gambar
def process_image(image, templates, threshold=0.5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 6000
    max_area = 50000
    counted_lele = 0
    contoured_image = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x,y,w,h = cv2.boundingRect(contour)
            crop = cleaned_image[y:y+h, x:x+w]
            contour_image_resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
            found = False
            for template in templates:
                for deg in range(36):
                    tmpl = ndimage.rotate(template, deg*10)
                    tmpl = cv2.resize(tmpl, (64, 128), interpolation=cv2.INTER_AREA)
                    bit = cv2.bitwise_not(cv2.bitwise_xor(tmpl, contour_image_resized)) / 255
                    jumlah = np.sum(bit)
                    if jumlah > threshold * (64*128):
                        counted_lele += 1
                        cv2.putText(contoured_image, str(counted_lele), (x, y + h//2), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)
                        found = True
                        break
                if found:
                    break

    return contoured_image, counted_lele

# Memuat template
templates = load_templates()

# Judul aplikasi
st.title("Deteksi Jumlah Bibit Lele")

# Unggah gambar atau ambil gambar menggunakan kamera
uploaded_file = st.file_uploader("Unggah gambar lele atau ambil gambar menggunakan kamera di bawah ini:", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    result_image, count_lele = process_image(image, templates)
    st.image(result_image, caption=f'Total bibit lele terdeteksi: {count_lele}', use_column_width=True)
