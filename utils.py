from dash import dcc, html
import json
import plotly.graph_objs as go
import dash_reusable_components as drc
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import numpy as np
from scipy import ndimage
from skimage import exposure, restoration


# [filename, image_signature, action_stack]
STORAGE_PLACEHOLDER = json.dumps({
    'filename': None,
    'image_signature': None, 
    'action_stack': []
})

IMAGE_STRING_PLACEHOLDER = drc.pil_to_b64(Image.open('images/default.jpg').copy(), enc_format='jpeg')

GRAPH_PLACEHOLDER = dcc.Graph(id='interactive-image', style={'height': '80vh'})

# Maps process name to the Image filter corresponding to that process
FILTERS_DICT = {
    'blur': ImageFilter.BLUR,
    'contour': ImageFilter.CONTOUR,
    'detail': ImageFilter.DETAIL,
    'edge_enhance': ImageFilter.EDGE_ENHANCE,
    'edge_enhance_more': ImageFilter.EDGE_ENHANCE_MORE,
    'emboss': ImageFilter.EMBOSS,
    'find_edges': ImageFilter.FIND_EDGES,
    'sharpen': ImageFilter.SHARPEN,
    'smooth': ImageFilter.SMOOTH,
    'smooth_more': ImageFilter.SMOOTH_MORE,
    'negative': 'negative'  # Custom implementation
}

ENHANCEMENT_DICT = {
    'color': ImageEnhance.Color,
    'contrast': ImageEnhance.Contrast,
    'brightness': ImageEnhance.Brightness,
    'sharpness': ImageEnhance.Sharpness
}


def generate_lasso_mask(image, selectedData):
    """
    Generates a polygon mask using the given lasso coordinates
    :param selectedData: The raw coordinates selected from the data
    :return: The polygon mask generated from the given coordinate
    """

    height = image.size[1]
    y_coords = selectedData['lassoPoints']['y']
    y_coords_corrected = [height - coord for coord in y_coords]

    coordinates_tuple = list(zip(selectedData['lassoPoints']['x'], y_coords_corrected))
    mask = Image.new('L', image.size)
    draw = ImageDraw.Draw(mask)
    draw.polygon(coordinates_tuple, fill=255)

    return mask


def apply_filters(image, zone, filter, mode):
    filter_selected = FILTERS_DICT[filter]

    if filter == 'negative':
        # Custom negative filter implementation
        if mode == 'select':
            crop = image.crop(zone)
            crop_mod = apply_negative_filter(crop)
            image.paste(crop_mod, zone)
        elif mode == 'lasso':
            im_filtered = apply_negative_filter(image)
            image.paste(im_filtered, mask=zone)
    else:
        # Standard PIL filters
        if mode == 'select':
            crop = image.crop(zone)
            crop_mod = crop.filter(filter_selected)
            image.paste(crop_mod, zone)
        elif mode == 'lasso':
            im_filtered = image.filter(filter_selected)
            image.paste(im_filtered, mask=zone)


def apply_enhancements(image, zone, enhancement, enhancement_factor, mode):
    enhancement_selected = ENHANCEMENT_DICT[enhancement]
    enhancer = enhancement_selected(image)

    im_enhanced = enhancer.enhance(enhancement_factor)

    if mode == 'select':
        crop = im_enhanced.crop(zone)
        image.paste(crop, box=zone)

    elif mode == 'lasso':
        image.paste(im_enhanced, mask=zone)


def apply_resolution_scaling(image, resolution_factor):
    """
    Apply spatial resolution scaling to the image.
    :param image: PIL Image object
    :param resolution_factor: Float between 0.1 and 1.0, where 1.0 is original resolution
    :return: PIL Image object with adjusted resolution
    """
    if resolution_factor == 1.0:
        return image
    
    # Calculate new dimensions
    original_width, original_height = image.size
    new_width = max(1, int(original_width * resolution_factor))
    new_height = max(1, int(original_height * resolution_factor))
    
    # Resize the image using LANCZOS for better quality (compatible with older PIL versions)
    try:
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except AttributeError:
        # Fallback for older PIL versions
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image


def apply_log_transformation(image, c_constant=1.0):
    """
    Apply logarithmic transformation to enhance dark regions of the image.
    Formula: s = c * log(1 + r), where r is the input pixel value
    :param image: PIL Image object
    :param c_constant: Constant to control the transformation intensity
    :return: PIL Image object with log transformation applied
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Handle different image modes
    if image.mode == 'RGBA':
        # Apply transformation only to RGB channels, keep alpha unchanged
        rgb_array = img_array[:, :, :3].astype(np.float32)
        alpha_array = img_array[:, :, 3:4]
        
        # Apply log transformation
        log_array = c_constant * np.log1p(rgb_array)
        
        # Normalize to 0-255 range
        log_normalized = (255 * (log_array / (c_constant * np.log1p(255)))).astype(np.uint8)
        
        # Combine with alpha channel
        result_array = np.concatenate([log_normalized, alpha_array], axis=2)
        
    elif image.mode == 'RGB':
        # Apply transformation to all RGB channels
        img_array = img_array.astype(np.float32)
        
        # Apply log transformation
        log_array = c_constant * np.log1p(img_array)
        
        # Normalize to 0-255 range
        result_array = (255 * (log_array / (c_constant * np.log1p(255)))).astype(np.uint8)
        
    elif image.mode == 'L':
        # Grayscale image
        img_array = img_array.astype(np.float32)
        
        # Apply log transformation
        log_array = c_constant * np.log1p(img_array)
        
        # Normalize to 0-255 range
        result_array = (255 * (log_array / (c_constant * np.log1p(255)))).astype(np.uint8)
    
    else:
        # For other modes, return original image
        return image
    
    # Convert back to PIL image
    return Image.fromarray(result_array, mode=image.mode)


def show_histogram(image):
    def hg_trace(name, color, hg):
        line = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            name=name,
            line=dict(color=(color)),
            mode='lines',
            showlegend=False
        )
        fill = go.Scatter(
            x=list(range(0, 256)),
            y=hg,
            mode='lines',
            name=name,
            line=dict(color=(color)),
            fill='tozeroy',
            showlegend=False,
            hoverinfo='skip'
        )

        return line, fill

    hg = image.histogram()

    if image.mode == 'RGBA':
        rhg = hg[0:256]
        ghg = hg[256:512]
        bhg = hg[512:768]
        ahg = hg[768:]

        data = [
            *hg_trace('Red', '#FF4136', rhg),
            *hg_trace('Green', '#2ECC40', ghg),
            *hg_trace('Blue', '#0074D9', bhg),
            *hg_trace('Alpha', 'gray', ahg)
        ]

        title = 'RGBA Histogram'

    elif image.mode == 'RGB':
        # Returns a 768 member array with counts of R, G, B values
        rhg = hg[0:256]
        ghg = hg[256:512]
        bhg = hg[512:768]

        data = [
            *hg_trace('Red', '#FF4136', rhg),
            *hg_trace('Green', '#2ECC40', ghg),
            *hg_trace('Blue', '#0074D9', bhg),
        ]

        title = 'RGB Histogram'

    else:
        data = [*hg_trace('Gray', 'gray', hg)]

        title = 'Grayscale Histogram'

    layout = go.Layout(
        title=title,
        margin=dict(l=35, r=35),
        legend=dict(x=0, y=1.15, orientation="h")
    )

    return go.Figure(data=data, layout=layout)


def apply_negative_filter(image):
    """
    Apply negative filter to invert image colors.
    :param image: PIL Image object
    :return: PIL Image object with negative filter applied
    """
    img_array = np.array(image)
    
    if image.mode == 'RGBA':
        # Invert RGB channels, keep alpha unchanged
        img_array[:, :, :3] = 255 - img_array[:, :, :3]
    else:
        # Invert all channels for RGB and grayscale
        img_array = 255 - img_array
    
    return Image.fromarray(img_array, mode=image.mode)


def apply_histogram_equalization(image):
    """
    Apply histogram equalization to enhance image contrast.
    :param image: PIL Image object
    :return: PIL Image object with histogram equalization applied
    """
    img_array = np.array(image)
    
    if image.mode == 'RGB':
        # Apply equalization to each channel separately
        img_eq = np.zeros_like(img_array)
        for i in range(3):
            img_eq[:, :, i] = exposure.equalize_hist(img_array[:, :, i]) * 255
        result_array = img_eq.astype(np.uint8)
        
    elif image.mode == 'RGBA':
        # Apply equalization to RGB channels, keep alpha unchanged
        img_eq = np.zeros_like(img_array)
        for i in range(3):
            img_eq[:, :, i] = exposure.equalize_hist(img_array[:, :, i]) * 255
        img_eq[:, :, 3] = img_array[:, :, 3]  # Keep alpha channel
        result_array = img_eq.astype(np.uint8)
        
    elif image.mode == 'L':
        # Grayscale histogram equalization
        img_eq = exposure.equalize_hist(img_array) * 255
        result_array = img_eq.astype(np.uint8)
    else:
        return image
    
    return Image.fromarray(result_array, mode=image.mode)


def apply_histogram_matching(image, reference_image):
    """
    Apply histogram matching to match the histogram of a reference image.
    :param image: PIL Image object to be processed
    :param reference_image: PIL Image object as reference
    :return: PIL Image object with histogram matching applied
    """
    if reference_image is None:
        # If no reference image is provided, return the original image
        return image

    img_array = np.array(image)
    ref_array = np.array(reference_image.convert(image.mode))

    # Check if the modes are compatible (e.g., both RGB or both grayscale)
    if img_array.shape[-1] != ref_array.shape[-1] and not (len(img_array.shape) == 2 and len(ref_array.shape) == 2):
        # If modes are incompatible, return the original image
        print("Warning: Image and reference image modes are incompatible for histogram matching.")
        return image
    
    # Proceed with histogram matching
    if image.mode in ['RGB', 'RGBA']:
        # Apply matching to each channel for RGB/RGBA
        img_matched = np.zeros_like(img_array)
        channels_to_match = 3 if image.mode == 'RGB' else 3 # Match only RGB
        
        for i in range(channels_to_match):
            img_matched[:, :, i] = exposure.match_histograms(
                img_array[:, :, i], ref_array[:, :, i]
            )
        if image.mode == 'RGBA':
            img_matched[:, :, 3] = img_array[:, :, 3] # Preserve alpha channel
            
        result_array = img_matched.astype(np.uint8)
        
    elif image.mode == 'L':
        # Grayscale matching
        img_matched = exposure.match_histograms(img_array, ref_array)
        result_array = img_matched.astype(np.uint8)
        
    else:
        # For other modes, return the original image
        return image
    
    return Image.fromarray(result_array, mode=image.mode)





def add_noise(image, noise_type='gaussian', intensity=25):
    """
    Add different types of noise to the image.
    :param image: PIL Image object
    :param noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson', 'uniform')
    :param intensity: Intensity of noise (0-100)
    :return: PIL Image object with noise added
    """
    img_array = np.array(image).astype(np.float32)
    intensity = max(0, min(100, intensity))  # Clamp between 0-100
    
    if noise_type == 'gaussian':
        # Gaussian noise
        noise = np.random.normal(0, intensity * 2.55, img_array.shape)
        noisy_array = img_array + noise
        
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        noisy_array = img_array.copy()
        prob = intensity / 200  # Convert to probability (0-0.5)
        
        # Salt noise (white pixels)
        salt_coords = np.random.random(img_array.shape[:2]) < prob/2
        noisy_array[salt_coords] = 255
        
        # Pepper noise (black pixels)
        pepper_coords = np.random.random(img_array.shape[:2]) < prob/2
        noisy_array[pepper_coords] = 0
        
    elif noise_type == 'poisson':
        # Poisson noise
        # Scale image to appropriate range for Poisson
        scaled = img_array * (intensity / 25)
        noisy_array = np.random.poisson(scaled).astype(np.float32)
        noisy_array = noisy_array * (25 / intensity) if intensity > 0 else img_array
        
    elif noise_type == 'uniform':
        # Uniform noise
        noise = np.random.uniform(-intensity * 2.55, intensity * 2.55, img_array.shape)
        noisy_array = img_array + noise
    else:
        return image
    
    # Clip values to valid range and convert back to uint8
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy_array, mode=image.mode)


def reduce_noise(image, method='gaussian', strength=1.0):
    """
    Apply noise reduction techniques to the image.
    :param image: PIL Image object
    :param method: Noise reduction method ('gaussian', 'median', 'bilateral', 'wiener')
    :param strength: Strength of noise reduction (0.1 - 3.0)
    :return: PIL Image object with noise reduction applied
    """
    img_array = np.array(image)
    strength = max(0.1, min(3.0, strength))
    
    if method == 'gaussian':
        # Gaussian blur for noise reduction
        sigma = strength
        if image.mode in ['RGB', 'RGBA']:
            result_array = np.zeros_like(img_array)
            channels = 3 if image.mode == 'RGB' else 4
            for i in range(min(channels, 3)):  # Don't blur alpha channel
                result_array[:, :, i] = ndimage.gaussian_filter(img_array[:, :, i], sigma=sigma)
            if image.mode == 'RGBA':
                result_array[:, :, 3] = img_array[:, :, 3]  # Keep alpha
        else:
            result_array = ndimage.gaussian_filter(img_array, sigma=sigma)
            
    elif method == 'median':
        # Median filter for noise reduction
        size = int(strength * 2) + 1  # Convert strength to odd kernel size
        if image.mode in ['RGB', 'RGBA']:
            result_array = np.zeros_like(img_array)
            channels = 3 if image.mode == 'RGB' else 4
            for i in range(min(channels, 3)):
                result_array[:, :, i] = ndimage.median_filter(img_array[:, :, i], size=size)
            if image.mode == 'RGBA':
                result_array[:, :, 3] = img_array[:, :, 3]
        else:
            result_array = ndimage.median_filter(img_array, size=size)
            
    elif method == 'bilateral':
        # Simplified bilateral-like filter using Gaussian
        # (True bilateral filtering requires more complex implementation)
        sigma = strength
        if image.mode in ['RGB', 'RGBA']:
            result_array = np.zeros_like(img_array)
            for i in range(min(3, img_array.shape[2])):
                result_array[:, :, i] = ndimage.gaussian_filter(img_array[:, :, i], sigma=sigma)
            if image.mode == 'RGBA':
                result_array[:, :, 3] = img_array[:, :, 3]
        else:
            result_array = ndimage.gaussian_filter(img_array, sigma=sigma)
            
    elif method == 'wiener':
        # Simplified Wiener-like filter using Gaussian
        # (True Wiener filtering requires frequency domain operations)
        sigma = strength * 0.5
        if image.mode in ['RGB', 'RGBA']:
            result_array = np.zeros_like(img_array)
            for i in range(min(3, img_array.shape[2])):
                result_array[:, :, i] = ndimage.gaussian_filter(img_array[:, :, i], sigma=sigma)
            if image.mode == 'RGBA':
                result_array[:, :, 3] = img_array[:, :, 3]
        else:
            result_array = ndimage.gaussian_filter(img_array, sigma=sigma)
    else:
        return image
    
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    return Image.fromarray(result_array, mode=image.mode)
