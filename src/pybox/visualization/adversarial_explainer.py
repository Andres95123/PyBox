from matplotlib import pyplot as plt
import numpy as np
from art.attacks.evasion import (
    CarliniL0Method,
    CarliniLInfMethod,
    CarliniL2Method,
    DeepFool,
    FastGradientMethod,
    ProjectedGradientDescent,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Any, Literal, Callable
from tqdm import tqdm

from ..compare.image_comparator import ImageComparator, METHODS, CMAP


METHOD_MAP = {
    "cwL0": CarliniL0Method,
    "cwL2": CarliniL2Method,
    "cwLinf": CarliniLInfMethod,
    "deepfool": DeepFool,
    "fgsm": FastGradientMethod,
    "pgd": ProjectedGradientDescent,
}


class AdversarialExplainer:
    """
    Class for visualizing adversarial examples using various attack methods.

    This class provides comprehensive visualization of adversarial attacks by generating
    adversarial examples using multiple attack methods and displaying them alongside
    their original images, difference maps, and aggregated statistics.

    Supported attack methods: 'cwL0', 'cwL2', 'cwLinf', 'deepfool', 'fgsm', 'pgd'

    Supported difference calculations:
    - 'MAE': Mean Absolute Error (MAE) - L1 distance
    - 'MSE': Mean Squared Error (MSE) - L2 distance squared
    - 'RMSE': Root Mean Square Error - square root of MSE
    - 'COSINE': Cosine distance - 1 minus cosine similarity
    - 'GMD': Gradient Magnitude Difference - perceptual quality metric
    """

    def __init__(
        self,
        # Base configuration
        adversarial_generator: Any,
        methods: List[str],
        # Preferences
        plot_difference: bool = True,
        difference_calculation: METHODS = "MAE",
        cmap : CMAP = "RGB",
        # Normalization
        scaler: MinMaxScaler | StandardScaler | RobustScaler | None = MinMaxScaler(),
        clipping_range: tuple[float, float] | None = (0.0, 1.0),
        # ART Configuration
        max_iter: int = 10,
    ) -> None:
        """
        Initialize the AdversarialExplainer with configuration parameters.

        Args:
            adversarial_generator: A trained classifier with predict() and loss_gradient() methods.
                                 Must be compatible with ART's attack implementations.

            methods: List of adversarial attack methods to use. Supported methods:
                    'cwL0', 'cwL2', 'cwLinf', 'deepfool', 'fgsm', 'pgd'

            plot_difference: If True, displays difference maps between original and adversarial images.
                           Defaults to True.

            difference_calculation: Method for calculating pixel-wise differences. Options:
                - 'MAE': |adv - orig| (Mean Absolute Error)
                - 'MSE': (adv - orig)² (Mean Squared Error)
                - 'RMSE': √((adv - orig)²) (Root Mean Square Error)
                - 'COSINE': 1 - cosine_similarity (cosine distance)
                - 'GMD': Gradient Magnitude Difference (perceptual quality metric)
                Defaults to 'MAE'.

            scaler: Scaler for normalizing difference maps. Options: MinMaxScaler(),
                   StandardScaler(), RobustScaler(), or None. Defaults to MinMaxScaler().

            clipping_range: Tuple (min, max) for clipping pixel values. None disables clipping.
                          Defaults to (0.0, 1.0).

            max_iter: Maximum iterations for iterative attack methods. Defaults to 10.

        Raises:
            ValueError: If methods contains unsupported attack types.
            TypeError: If adversarial_generator lacks required methods.
        """
        if not all(method in METHOD_MAP.keys() for method in methods):
            raise ValueError(
                "methods must be a list containing any of the following strings: "
                f"{', '.join(METHOD_MAP.keys())}"
            )

        if not hasattr(adversarial_generator, "predict") or not hasattr(
            adversarial_generator, "loss_gradient"
        ):
            raise TypeError(
                "adversarial_generator must be a valid classifier with predict and loss_gradient methods"
            )

        self.adversarial_generator: Any = adversarial_generator
        self.methods: List[str] = methods
        self.plot_difference: bool = plot_difference
        self.difference_calculation: METHODS = difference_calculation
        self.scaler: MinMaxScaler | StandardScaler | RobustScaler | None = scaler
        self.clipping_range: tuple[float, float] | None = clipping_range
        self.max_iter: int = max_iter
        self.cmap: CMAP = cmap

    def __get_method_instance(self, method_name: str) -> Any:
        """Instantiate the attack method based on its name."""
        attack_class = METHOD_MAP[method_name]
        if method_name in ["cwL0", "cwL2", "cwLinf", "deepfool"]:
            return attack_class(
                classifier=self.adversarial_generator,
                max_iter=self.max_iter,
                verbose=False,
            )
        elif method_name == "fgsm":
            return attack_class(estimator=self.adversarial_generator, eps=0.2)
        elif method_name == "pgd":
            return attack_class(
                estimator=self.adversarial_generator, eps=0.3, max_iter=self.max_iter
            )
        else:
            raise ValueError(f"Unsupported attack method: {method_name}")

    def explain(
        self,
        input_imgs: np.ndarray,
        ground_truth: np.ndarray | None = None,
        predict_labels: bool = False,
    ) -> None:
        """
        Visualize adversarial examples generated by multiple attack methods on input images.

        This method generates adversarial examples using the attack methods specified during
        initialization and creates a comprehensive visualization showing the original images,
        adversarial variants, their differences, and a summary of aggregated differences across
        all attack methods.

        **Plot Layout:**
        Each row displays one input image with the following columns:
        - Original Image
        - [Adversarial Image 1] [Difference 1] [Adversarial Image 2] [Difference 2] ...
        - Sum of Differences (if plot_difference=True)

        **Labels and Annotations:**
        - Ground truth labels (if provided) are shown below the original image
        - Predicted labels (if predict_labels=True) are shown below each adversarial image
        - Normalized differences are visualized using the configured scaler

        **Normalization:**
        - Pixel differences are computed using the specified method:
          * 'MAE': |adv - orig| (Mean Absolute Error/L1 distance)
          * 'MSE': (adv - orig)² (Mean Squared Error/L2 distance squared)
          * 'RMSE': √((adv - orig)²) (Root Mean Square Error)
          * 'COSINE': 1 - cosine_similarity (cosine distance)
          * 'GMD': Gradient Magnitude Difference (perceptual quality metric)
        - Differences are scaled using the configured scaler (MinMaxScaler, StandardScaler, etc.)
        - All values are clipped to the specified range if clipping_range is configured

        :param input_imgs: Input images to generate adversarial examples from.
                          Must be a 4D numpy array with shape (num_images, height, width, channels).
                          Pixel values should typically be in range [0, 1] or [0, 255].
        :type input_imgs: np.ndarray

        :param ground_truth: Ground truth labels for the input images. When provided, these labels
                            are displayed below the original image column for reference. Must have
                            shape (num_images,) and contain class indices or labels.
                            Defaults to None (no ground truth labels displayed).
        :type ground_truth: np.ndarray | None

        :param predict_labels: If True, the model's predicted class labels are displayed below
                              each adversarial image. This helps visualize how the adversarial
                              perturbations affect the model's predictions.
                              Defaults to False (no predicted labels displayed).
        :type predict_labels: bool

        :return: None. The visualization is displayed using matplotlib.pyplot.show().
        :rtype: None

        :raises TypeError: If input_imgs is not a numpy array or adversarial_generator lacks
                          required methods (predict and loss_gradient).
        :raises ValueError: If input_imgs is empty (shape[0] == 0) or contains invalid data.

        **Example:**

            >>> from pybox.visualization import AdversarialExplainer
            >>> import numpy as np
            >>> # Assuming you have a trained model and test images
            >>>
            >>> # Example with squared error (MSE) differences
            >>> explainer = AdversarialExplainer(
            ...     adversarial_generator=model,
            ...     methods=['fgsm', 'pgd'],
            ...     difference_calculation='MSE'  # MSE differences
            ... )
            >>>
            >>> # Example with GMD-based perceptual differences
            >>> explainer_ssim = AdversarialExplainer(
            ...     adversarial_generator=model,
            ...     methods=['fgsm', 'deepfool'],
            ...     difference_calculation='GMD'  # Perceptual quality differences
            ... )
            >>>
            >>> # Example with cosine distance (useful for feature-level differences)
            >>> explainer_cosine = AdversarialExplainer(
            ...     adversarial_generator=model,
            ...     methods=['cwL2', 'pgd'],
            ...     difference_calculation='COSINE'  # Cosine distance
            ... )
            >>>
            >>> test_images = np.random.rand(3, 32, 32, 3)  # CIFAR-like images
            >>> ground_truth = np.array([5, 2, 8])
            >>> explainer.explain(test_images, ground_truth=ground_truth, predict_labels=True)
        """

        # Input validation: Type checking
        if not isinstance(input_imgs, np.ndarray):
            raise TypeError(
                f"input_imgs must be a numpy ndarray, got {type(input_imgs).__name__}"
            )

        # Input validation: Check if there are images to process
        if input_imgs.shape[0] == 0:
            raise ValueError(
                "input_imgs must contain at least one image to visualize. "
                f"Got shape with 0 images: {input_imgs.shape}"
            )

        # Calculate plot dimensions
        cols_per_method = 1 + (1 if self.plot_difference else 0)
        num_figures = (
            1 + len(self.methods) * cols_per_method + (1 if self.plot_difference else 0)
        )

        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=input_imgs.shape[0],
            ncols=num_figures,
            figsize=(3 * num_figures, 3 * input_imgs.shape[0]),
        )

        # Process each image
        for i in tqdm(range(input_imgs.shape[0]), desc="Generating adversarial images"):
            # Clip input image if necessary
            if self.clipping_range is not None:
                input_imgs[i] = np.clip(
                    input_imgs[i], self.clipping_range[0], self.clipping_range[1]
                )

            # Display original image
            axes[i, 0].imshow(input_imgs[i])
            axes[i, 0].set_title("Original Image", fontweight="bold")

            # Add ground truth label if provided
            if ground_truth is not None:
                axes[i, 0].set_xlabel(f"GT: {ground_truth[i]}", fontsize=9)

            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])

            # Store differences for aggregation at the end
            all_differences = []

            # Generate adversarial examples for each method
            for j, method in enumerate(self.methods):
                # Get attack instance for this method
                attack = self.__get_method_instance(method)

                # Generate adversarial example
                selected_img = input_imgs[i : i + 1]
                adversarial_img = attack.generate(x=selected_img)

                # Clip adversarial image if necessary
                if self.clipping_range is not None:
                    adversarial_img = np.clip(
                        adversarial_img,
                        self.clipping_range[0],
                        self.clipping_range[1],
                    )

                # Display adversarial image
                adv_col = 1 + j * cols_per_method
                axes[i, adv_col].imshow(adversarial_img[0])
                axes[i, adv_col].set_title(f"Adversarial ({method})", fontweight="bold")

                # Add predicted label if requested
                if predict_labels:
                    pred_label = np.argmax(
                        self.adversarial_generator.predict(adversarial_img), axis=1
                    )[0]
                    axes[i, adv_col].set_xlabel(f"Pred: {pred_label}", fontsize=9)

                axes[i, adv_col].set_xticks([])
                axes[i, adv_col].set_yticks([])

                # Calculate and display difference map
                if self.plot_difference:
                    # Compute difference using the configured calculation method

                    difference = ImageComparator().compare(
                        adversarial_img[0],
                        selected_img[0],
                        method=self.difference_calculation,
                        cmap=self.cmap,
                    )

                    # Normalize difference using configured scaler
                    if self.scaler is not None:
                        difference = self.scaler.fit_transform(
                            difference.reshape(-1, 1)
                        ).reshape(difference.shape)

                    # Clip difference if necessary
                    if self.clipping_range is not None:
                        difference = np.clip(
                            difference, self.clipping_range[0], self.clipping_range[1]
                        )

                    all_differences.append(difference)

                    # Display difference map
                    diff_col = adv_col + 1
                    axes[i, diff_col].imshow(difference, cmap="hot")
                    axes[i, diff_col].set_title(
                        f"Difference ({method})", fontweight="bold"
                    )
                    axes[i, diff_col].set_xticks([])
                    axes[i, diff_col].set_yticks([])

            # Display aggregated differences across all methods
            if self.plot_difference and all_differences:
                summed_differences = np.sum(all_differences, axis=0)

                # Normalize aggregated differences
                if self.scaler is not None:
                    summed_differences = self.scaler.fit_transform(
                        summed_differences.reshape(-1, 1)
                    ).reshape(summed_differences.shape)

                # Clip aggregated differences
                if self.clipping_range is not None:
                    summed_differences = np.clip(
                        summed_differences,
                        self.clipping_range[0],
                        self.clipping_range[1],
                    )

                # Display sum of differences
                sum_col = num_figures - 1
                axes[i, sum_col].imshow(summed_differences, cmap="hot")
                axes[i, sum_col].set_title("Sum of Differences", fontweight="bold")
                axes[i, sum_col].set_xticks([])
                axes[i, sum_col].set_yticks([])

        # Apply tight layout and display
        plt.tight_layout()
        plt.show()
