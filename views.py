from flask import Flask, request, Response, Blueprint, render_template
import numpy as np
import cv2
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


views = Blueprint('views', __name__)

@views.route("/")
def home():
    return render_template("index.html")

@views.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        in_memory_file = io.BytesIO()
        uploaded_file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 1
        image = cv2.imdecode(data, color_image_flag)

        figures = process_image(image)

        figure_images = []
        for fig in figures:
            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img.reshape(canvas.get_width_height()[::-1] + (3,))
            figure_images.append(img)

        def pad_image(img, target_size, bg_color=(255, 255, 255)):
            padded_img = np.full(target_size + (3,), bg_color, dtype=np.uint8)
            padded_img[:img.shape[0], :img.shape[1], :] = img
            return padded_img

        max_height = max(img.shape[0] for img in figure_images)
        max_width = max(img.shape[1] for img in figure_images)

        padded_images = [pad_image(img, (max_height, max_width)) for img in figure_images]

        n_cols = 4
        row_images = [padded_images[i:i + n_cols] for i in range(0, len(padded_images), n_cols)]

        combined_rows = []
        for row in row_images:
            while len(row) < n_cols:
                row.append(np.full((max_height, max_width, 3), 255, dtype=np.uint8))
            
            combined_row = np.concatenate(row, axis=1) 
            combined_rows.append(combined_row)

        combined_image = np.concatenate(combined_rows, axis=0)

        output = io.BytesIO()
        plt.imsave(output, combined_image, format='png')
        output.seek(0)
        return Response(output.getvalue(), mimetype='image/png')

    return "No file uploaded"

def process_image(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sigmas = [5, 10, 15]
    figures = []

    ft2d = np.fft.fft2(grey_image)
    ft_shift2d = np.fft.fftshift(ft2d)
    mag_spec = 20* np.log(np.abs(ft_shift2d))

    # Function to create Gaussian filter
    def create_gaussian_filter(size, sigma):
        m, n = [(ss-1.)/2. for ss in size]
        y, x = np.ogrid[-m:m+1,-n:n+1]
        filter = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        return filter / filter.sum()
    
    # Function to apply Gaussian low-pass filter
    def apply_gaussian_filter(image, sigma):
        gaussian_filter = create_gaussian_filter((30, 30), sigma)
        gaussian_filter_ft = np.fft.fft2(gaussian_filter, s=image.shape)
        gaussian_filter_ft_shift = np.fft.fftshift(gaussian_filter_ft)

        ft = np.fft.fft2(image)
        ft_shift = np.fft.fftshift(ft)

        filtered_ft_shift = ft_shift * gaussian_filter_ft_shift
        filtered_ft = np.fft.ifftshift(filtered_ft_shift)
        filtered_image = np.fft.ifft2(filtered_ft)

        return np.abs(filtered_image)

    # Function to create Gaussian high-pass filter
    def create_gaussian_highpass_filter(size, sigma):
        low_pass = create_gaussian_filter(size, sigma)
        high_pass = np.ones(size) - low_pass
        return np.fft.ifftshift(high_pass)

    # Function to apply Gaussian high-pass filter
    def apply_highpass_filter(image, sigma):
        highpass_filter = create_gaussian_highpass_filter(image.shape, sigma)
        highpass_filter_ft = np.fft.fft2(highpass_filter, s=image.shape)
        highpass_filter_ft_shift = np.fft.fftshift(highpass_filter_ft)

        ft_shift = np.fft.fftshift(np.fft.fft2(image))
        filtered_ft_shift = ft_shift * highpass_filter_ft_shift
        filtered_ft = np.fft.ifftshift(filtered_ft_shift)
        filtered_image = np.fft.ifft2(filtered_ft)

        return np.abs(filtered_image)

#Plotting
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.imshow(grey_image, cmap='gray')
    axis.set_title('Original Image')
    figures.append(fig)
    
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.imshow(mag_spec, cmap='gray')
    axis.set_title('Magnitude Spectrum')
    figures.append(fig)
    
    for sigma in sigmas:
        gaussian_filter = create_gaussian_filter(grey_image.shape, sigma)
        gaussian_filter_ft = np.fft.fft2(gaussian_filter, s=grey_image.shape)
        gaussian_filter_ft_shift = np.fft.fftshift(gaussian_filter_ft)
        gaussian_filter_mag_spec = 20 * np.log(np.abs(gaussian_filter_ft_shift) + 1)

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.imshow(gaussian_filter_mag_spec, cmap='gray')
        axis.set_title(f'Gaussian Low-pass Filter Spectrum (sigma = {sigma})')
        figures.append(fig)

    for sigma in sigmas:
        highpass_filter = create_gaussian_highpass_filter(grey_image.shape, sigma)
        highpass_filter_ft = np.fft.fft2(highpass_filter, s=grey_image.shape)
        highpass_filter_ft_shift = np.fft.fftshift(highpass_filter_ft)
        highpass_filter_mag_spec = 20 * np.log(np.abs(highpass_filter_ft_shift))

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.imshow(highpass_filter_mag_spec, cmap='gray_r')
        axis.set_title(f'Gaussian High-pass Filter Spectrum (sigma = {sigma})')
        figures.append(fig)

    for sigma in sigmas:
        filtered_image = apply_gaussian_filter(grey_image, sigma)
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.imshow(filtered_image, cmap='gray')
        axis.set_title(f'Low-pass Filter with sigma = {sigma}')
        figures.append(fig)

    for sigma in sigmas:
        highpass_filtered_image = apply_highpass_filter(grey_image, sigma)
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.imshow(highpass_filtered_image, cmap='gray')
        axis.set_title(f'High-pass Filter with sigma = {sigma}')
        figures.append(fig)

    return figures