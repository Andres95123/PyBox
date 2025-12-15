from typing import Literal
import numpy as np

METHODS = Literal["MAE", "MSE", "RMSE", "COSINE", "GMD"]

class ImageComparator:
    def __init__(self):
        """Initialize the ImageComparator."""
        pass

    def compare(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        method: METHODS = "MAE",
    ) -> np.ndarray:
        """
        Compares images using the specified method and returns the difference.

        Args:
            method (str): The method to use for comparison (MAE, MSE, RMSE, COSINE, GMD).
            image1 (np.ndarray): The first image to compare (H, W, C).
            image2 (np.ndarray): The second image to compare (H, W, C).

        Returns:
            np.ndarray: The result of the image comparison in RGB format (H, W, C or H, W).
                Always returns 3 channels for consistency.

        Raises:
            ValueError: If the method is not a valid option.
        """

        dif_img = None

        if method == "MAE":
            dif_img = np.abs(image1 - image2)
            # Normalizar a [0, 255]
            maxv = np.max(dif_img)
            if maxv > 0:
                dif_img = (dif_img / maxv) * 255
            else:
                dif_img = np.zeros_like(dif_img)
            dif_img = dif_img.astype(np.uint8)
        elif method == "MSE":
            dif_img = np.square(image1 - image2)
            # Normalizar a [0, 255]
            maxv = np.max(dif_img)
            if maxv > 0:
                dif_img = (dif_img / maxv) * 255
            else:
                dif_img = np.zeros_like(dif_img)
            dif_img = dif_img.astype(np.uint8)
        elif method == "RMSE":
            dif_img = np.sqrt(np.square(image1 - image2))
            # Normalize the RMSE result to the range [0, 255], avoid division by zero
            maxv = np.max(dif_img)
            if maxv > 0:
                dif_img = (dif_img / maxv) * 255
            else:
                dif_img = np.zeros_like(dif_img)
            dif_img = dif_img.astype(np.uint8)
        elif method == "COSINE":

            def cosine_sim(a, b):
                dot = np.sum(a * b, axis=-1)
                norm_a = np.linalg.norm(a, axis=-1)
                norm_b = np.linalg.norm(b, axis=-1)
                return dot / (norm_a * norm_b + 1e-10)

            sim = cosine_sim(image1.astype(float), image2.astype(float))
            diff = 1 - sim
            # Expand to 3 channels for consistency
            dif_img = np.stack([diff] * 3, axis=-1)
            # Normalize to [0, 255], avoid division by zero
            maxv = np.max(dif_img)
            if maxv > 0:
                dif_img = (dif_img / maxv) * 255
            else:
                dif_img = np.zeros_like(dif_img)
            dif_img = dif_img.astype(np.uint8)
        elif method == "GMD":

            def compute_grad(img):
                grad = np.zeros_like(img, dtype=float)
                for c in range(img.shape[2]):
                    ch = img[:, :, c].astype(float)
                    grad_y, grad_x = np.gradient(ch)
                    grad[:, :, c] = np.sqrt(grad_x**2 + grad_y**2)
                return grad

            g1 = compute_grad(image1)
            g2 = compute_grad(image2)
            dif_img = np.abs(g1 - g2)
            # Normalize to [0, 255], avoid division by zero
            maxv = np.max(dif_img)
            if maxv > 0:
                dif_img = (dif_img / maxv) * 255
            else:
                dif_img = np.zeros_like(dif_img)
            dif_img = dif_img.astype(np.uint8)
        else:
            raise ValueError(
                f"Method must be a valid option : {', '.join(METHODS.__args__)}"
            )

        # Always ensure 3 channels for consistency
        if len(dif_img.shape) == 2:
            dif_img = np.stack([dif_img] * 3, axis=-1)
        elif dif_img.shape[-1] == 1:
            dif_img = np.repeat(dif_img, 3, axis=-1)

        return dif_img
