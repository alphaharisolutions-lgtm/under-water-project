import cv2
import numpy as np

class VariationalEnhancer:
    """
    Implements a physics-based variational model for underwater image restoration.
    Consistently handles BGR (input) to RGB (internal/output) conversion.
    """
    def __init__(self, omega=0.95, win_size=15):
        self.omega = omega
        self.win_size = win_size

    def get_dark_channel(self, image):
        # image is RGB float
        # Underwater Dark Channel Prior often benefits from using Blue and Green channels
        # but standard DCP uses min of all channels.
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.win_size, self.win_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def estimate_backlight(self, image, dark_channel):
        """
        Estimates the background light (B) in RGB.
        Uses a robust median of the top 0.1% brightest pixels in the dark channel.
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        top_k = int(max(num_pixels * 0.001, 1))

        flat_img = image.reshape(num_pixels, 3)
        flat_dark = dark_channel.ravel()
        
        indices = np.argpartition(flat_dark, -top_k)[-top_k:]
        # backlight is [R, G, B]
        backlight = np.median(flat_img[indices], axis=0)
        
        # Stability check: ensure backlight isn't pure black
        backlight = np.maximum(backlight, 0.01)
        
        # Purely underwater heuristic: Red channel is absorbed most.
        # Background light should have B >= G > R.
        # If R is too high, it's likely a foreground object (like yellow coral).
        if backlight[0] > backlight[2] * 0.8: # Red vs Blue
            backlight[0] = backlight[2] * 0.4 # Cap Red
            
        return backlight

    def get_transmission(self, image, backlight):
        # image/backlight are RGB
        norm_img = image / backlight
        # DCP on normalized image
        # min_c(min_y(I^c(y)/A^c))
        dc_norm = self.get_dark_channel(norm_img)
        transmission = 1 - self.omega * dc_norm
        return np.clip(transmission, 0.1, 0.9)

    def guided_filter(self, I, p, r, eps):
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
        return mean_a * I + mean_b

    def color_balance(self, image):
        """
        Conservative Global Contrast Stretching.
        Preserving ratios ensures "No Change Colour" while maximizing dynamic range.
        """
        p_low = np.percentile(image, 0.2)
        p_high = np.percentile(image, 99.8)
        out = (image - p_low) / (p_high - p_low + 1e-6)
        return np.clip(out, 0, 1)

    def apply_clahe(self, image):
        """
        Applies Contrast Limited Adaptive Histogram Equalization for HD clarity.
        Operates on LAB color space to preserve color while boosting local contrast.
        """
        img_uint8 = (image * 255).astype(np.uint8)
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Grid size 8x8 is standard for high-res images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final.astype(np.float32) / 255.0

    def process(self, image_numpy):
        """
        Input: image_numpy (BGR, uint8)
        Output: (RGB_float_0_1, transmission_map)
        """
        # Convert to RGB Float
        img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 1. Dark Channel on raw image
        dc = self.get_dark_channel(img_rgb)
        
        # 2. Backlight Estimation
        backlight = self.estimate_backlight(img_rgb, dc)
        
        # 3. Transmission Estimation
        t_raw = self.get_transmission(img_rgb, backlight)
        
        # 4. Refine Transmission with Guided Filter
        # Radius 20 is better for preserving high-frequency clarity than 60
        gray = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_refined = self.guided_filter(gray, t_raw, r=20, eps=1e-3)
        # Floor 0.25 allows for much clearer results than 0.4
        t_refined = np.clip(t_refined, 0.25, 0.95).astype(np.float32)
        
        # 5. Scene Recovery: J = (I - B)/t + B
        t_3d = np.expand_dims(t_refined, axis=2)
        recovered = (img_rgb - backlight) / t_3d + backlight
        recovered = np.clip(recovered, 0, 1)
        
        # 6. Global Enhancement (Hue preservation)
        balanced = self.color_balance(recovered)
        
        # 7. Local Contrast Boost for "Full HD Clarity"
        final = self.apply_clahe(balanced)
        
        return final, t_refined
