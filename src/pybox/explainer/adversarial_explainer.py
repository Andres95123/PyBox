import numpy as np
from art.attacks import EvasionAttack
import inspect
import warnings

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import List, Any, Callable
from tqdm import tqdm

from ..compare.image_comparator import ImageComparator, METHODS


class AdversarialExplainer:
    """
    AdversarialExplainer: Generate adversarial examples and compute differences across multiple attacks.

    This class generates adversarial examples using multiple attack methods and computes per-pixel
    differences between original and adversarial images. It returns structured numpy arrays that can
    be used for analysis, visualization, or further processing.

    Main parameters:
        adversarial_generator: ART-compatible classifier implementing predict()
                               (and optionally loss_gradient()). Must be compatible with ART attacks.
        methods: List (or tuple) of attacks. Elements can be EvasionAttack instances or classes
                 that will be instantiated. ART attacks (art.attacks.evasion.*) are recommended.
        compute_differences (bool): Whether to compute difference maps. Default: True.
        difference_calculation (str|callable): Method to compute per-pixel differences:
            - 'MAE'   : Mean Absolute Error (|adv - orig|)
            - 'MSE'   : Mean Squared Error ((adv - orig)**2)
            - 'RMSE'  : Root Mean Square Error (sqrt(MSE))
            - 'COSINE': 1 - cosine_similarity (cosine distance)
            - 'GMD'   : Gradient Magnitude Difference (perceptual metric)
          A custom callable f(adv, orig) -> image is also accepted.
        scaler: Scaler to normalize difference maps: MinMaxScaler(), StandardScaler(),
                RobustScaler() or None. Default: MinMaxScaler().
        clipping_range: Tuple (min, max) to clip pixel values. None disables clipping.
                        Default: (0.0, 1.0).
        max_iter: Maximum iterations for iterative attacks. Default: 10.

    Behavior and validations:
        - If methods is not a list/tuple, a TypeError is raised.
        - Classes provided in methods are attempted to be instantiated; if instantiation fails,
          a TypeError is raised.
        - If adversarial_generator does not implement predict(), a TypeError is raised.
        - A custom difference function must accept two images and return an image with the same
          shape as the expected difference map.
        - For attack-specific issues, consult ART implementations or open an issue.

    Notes:
        - All returned difference maps are in RGB format (H, W, C)
        - Color transformations (e.g., to grayscale) should be applied during visualization in Plotter
    """

    def __init__(
        self,
        # Base configuration
        adversarial_generator,
        methods: List[EvasionAttack],
        # Preferences
        compute_differences: bool = True,
        difference_calculation: METHODS | Callable = "MAE",
        # Normalization
        scaler: MinMaxScaler | StandardScaler | RobustScaler | None = MinMaxScaler(),
        clipping_range: tuple[float, float] | None = (0.0, 1.0),
        # ART Configuration
        max_iter: int = 10,
    ) -> None:
        if not isinstance(methods, (list, tuple)):
            raise TypeError(
                "methods must be a list or tuple of EvasionAttack instances or EvasionAttack classes"
            )

        # Normalize methods: accept instances or classes (try to instantiate classes)
        # Store both instances and their metadata for re-instantiation
        normalized_methods = []
        method_metadata = []  # Store original classes and init params

        for m in methods:
            if inspect.isclass(m):
                # Store the class for later re-instantiation with different params
                method_class = m
                try:
                    m = m(adversarial_generator, verbose=False)
                except Exception as e:
                    raise TypeError(
                        f"Failed to instantiate attack class {getattr(m, '__name__', str(m))}: {e}"
                    )
            else:
                # Instance provided - try to get its class
                method_class = type(m)

            if not hasattr(m, "generate"):
                raise TypeError(
                    f"Each element in 'methods' must be an instance of ART EvasionAttack (or a class that can be instantiated), got {type(m).__name__}"
                )
            normalized_methods.append(m)
            method_metadata.append(method_class)

        methods = normalized_methods

        if not hasattr(adversarial_generator, "predict"):
            raise TypeError(
                "Adversarial generator must implement a 'predict' method. You may use one of the ART classifiers or wrappers: "
            )

        self.adversarial_generator: Any = adversarial_generator
        self.methods: List[EvasionAttack] = methods
        self.method_metadata: List[type] = method_metadata
        self.compute_differences: bool = compute_differences
        self.difference_calculation: METHODS | Callable = difference_calculation
        self.scaler: MinMaxScaler | StandardScaler | RobustScaler | None = scaler
        self.clipping_range: tuple[float, float] | None = clipping_range
        self.max_iter: int = max_iter

    def explain(
        self,
        input_imgs: np.ndarray,
        ground_truth: np.ndarray | None = None,
        target_y: int | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate adversarial examples and compute differences across multiple attack methods.

        This method generates adversarial examples using the attack methods specified during
        initialization and computes per-pixel differences between original and adversarial images.

        Args:
            input_imgs (np.ndarray): Input images with shape (n_images, height, width, channels).
            ground_truth (np.ndarray, optional): Ground truth labels for each image. Shape: (n_images,).
            target_y (int | np.ndarray, optional): Target label(s) for targeted attacks.
                If int, attempts to misclassify all images to this class.
                If np.ndarray, must match shape (n_images,), providing a target for each image.
                If None, performs untargeted attacks (default behavior).

        Returns:
            dict[str, np.ndarray]: Dictionary containing:
                - 'originals': Original input images. Shape: (n_images, height, width, channels)
                - 'adversarials': Adversarial examples. Shape: (n_images, n_methods, height, width, channels)
                - 'differences': Difference maps. Shape: (n_images, n_methods, height, width, channels)
                  Only present if compute_differences=True.
                - 'aggregated': Sum of differences across all methods. Shape: (n_images, height, width, channels)
                  Only present if compute_differences=True.
                - 'predictions': Predicted labels for adversarial examples. Shape: (n_images, n_methods)
                - 'ground_truth': Ground truth labels (if provided). Shape: (n_images,)
                - 'method_names': List of attack method names. Length: n_methods

        Raises:
            TypeError: If input_imgs is not a numpy ndarray or adversarial_generator cannot predict.
            ValueError: If input_imgs contains no images.
        """

        # Input validation
        self._validate_input(input_imgs)

        n_images = input_imgs.shape[0]
        n_methods = len(self.methods)
        img_shape = input_imgs.shape[1:]

        # Validate target_y if provided
        targets = None
        active_methods = self.methods

        if target_y is not None:
            if isinstance(target_y, (int, np.integer)):
                targets = np.full(n_images, target_y, dtype=np.int64)
            elif isinstance(target_y, np.ndarray):
                if target_y.shape[0] != n_images:
                    raise ValueError(
                        f"target_y array shape {target_y.shape} does not match number of images ({n_images})"
                    )
                targets = target_y.astype(np.int64)
            else:
                raise TypeError("target_y must be an int or a numpy array.")

            # Re-instantiate methods with targeted=True
            active_methods = self._create_targeted_methods()

        # Initialize output dictionary
        results = self._initialize_results(n_images, n_methods, img_shape, ground_truth)

        # Process each image
        for i in tqdm(range(n_images), desc="Generating adversarial images"):
            target_class = targets[i] if targets is not None else None
            self._process_image(i, input_imgs[i], results, target_class, active_methods)

        return results

    def _validate_input(self, input_imgs: np.ndarray) -> None:
        """Validate input images."""
        if not isinstance(input_imgs, np.ndarray):
            raise TypeError(
                f"input_imgs must be a numpy ndarray, got {type(input_imgs).__name__}"
            )

        if input_imgs.shape[0] == 0:
            raise ValueError(
                "input_imgs must contain at least one image. "
                f"Got shape with 0 images: {input_imgs.shape}"
            )

    def _create_targeted_methods(self) -> List[EvasionAttack]:
        """Re-instantiate methods with targeted=True for targeted attacks."""
        targeted_methods = []
        for method_class in self.method_metadata:
            try:
                # Try to create with targeted=True
                # Check if the constructor accepts 'targeted' parameter
                sig = inspect.signature(method_class.__init__)
                if "targeted" in sig.parameters:
                    targeted_method = method_class(
                        self.adversarial_generator, targeted=True, verbose=False
                    )
                else:
                    # Method doesn't support targeted attacks, use original
                    warnings.warn(
                        f"{method_class.__name__} does not support targeted attacks. "
                        f"Results may not match the target class.",
                        UserWarning,
                    )
                    # Use the original instance
                    targeted_method = self.methods[len(targeted_methods)]
                targeted_methods.append(targeted_method)
            except Exception as e:
                warnings.warn(
                    f"Failed to create targeted version of {method_class.__name__}: {e}. "
                    f"Using untargeted version.",
                    UserWarning,
                )
                targeted_methods.append(self.methods[len(targeted_methods)])
        return targeted_methods

    def _initialize_results(
        self,
        n_images: int,
        n_methods: int,
        img_shape: tuple,
        ground_truth: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        """Initialize results dictionary with correct shapes."""
        results = {
            "originals": np.zeros((n_images, *img_shape)),
            "adversarials": np.zeros((n_images, n_methods, *img_shape)),
            "predictions": np.zeros((n_images, n_methods), dtype=np.int64),
            "method_names": [method.__class__.__name__ for method in self.methods],
        }

        if self.compute_differences:
            results["differences"] = np.zeros((n_images, n_methods, *img_shape))
            results["aggregated"] = np.zeros((n_images, *img_shape))

        if ground_truth is not None:
            results["ground_truth"] = ground_truth

        return results

    def _process_image(
        self,
        img_idx: int,
        input_img: np.ndarray,
        results: dict[str, np.ndarray],
        target_class: int | None = None,
        methods: List[EvasionAttack] | None = None,
    ) -> None:
        """Process a single image: clip, generate adversarials, and compute differences."""
        if methods is None:
            methods = self.methods

        clipped_input = self._clip_image(input_img)
        results["originals"][img_idx] = clipped_input

        # Get original prediction for success check
        if target_class is None:
            orig_pred_prob = self.adversarial_generator.predict(
                clipped_input[np.newaxis, ...]
            )
            original_pred = np.argmax(orig_pred_prob, axis=1)[0]
        else:
            original_pred = None  # Not needed for targeted check

        all_differences = []

        for method_idx, method in enumerate(methods):
            adversarial_img = self._generate_adversarial(
                method, clipped_input, target_class
            )
            results["adversarials"][img_idx, method_idx] = adversarial_img

            # Get prediction
            pred_prob = self.adversarial_generator.predict(
                adversarial_img[np.newaxis, ...]
            )
            pred_label = np.argmax(pred_prob, axis=1)[0]
            results["predictions"][img_idx, method_idx] = pred_label

            # Determine success
            if target_class is not None:
                # Targeted attack: success if prediction matches target
                success = pred_label == target_class
            else:
                # Untargeted attack: success if prediction differs from original
                success = pred_label != original_pred

            # Compute differences
            if self.compute_differences:
                if success:
                    difference = self._compute_difference(
                        adversarial_img, clipped_input
                    )
                    results["differences"][img_idx, method_idx] = difference
                    all_differences.append(difference)
                else:
                    # If failed, leave as zeros (default) and do not add to aggregation
                    pass

        # Aggregate differences
        if self.compute_differences and all_differences:
            aggregated = self._aggregate_differences(all_differences)
            results["aggregated"][img_idx] = aggregated

    def _clip_image(self, image: np.ndarray) -> np.ndarray:
        """Clip image to specified range."""
        if self.clipping_range is None:
            return image.copy()
        return np.clip(image, self.clipping_range[0], self.clipping_range[1])

    def _generate_adversarial(
        self, method, clipped_input: np.ndarray, target_y: int | None = None
    ) -> np.ndarray:
        """Generate adversarial example and return clipped version."""
        selected_img = clipped_input[np.newaxis, ...]

        if target_y is not None:
            # For targeted attacks, pass the target class to generate()
            # ART expects y as indices array
            y = np.array([target_y], dtype=np.int64)
            adversarial_img = method.generate(x=selected_img, y=y)[0]
        else:
            # Untargeted attack
            adversarial_img = method.generate(x=selected_img)[0]

        if self.clipping_range is not None:
            adversarial_img = np.clip(
                adversarial_img,
                self.clipping_range[0],
                self.clipping_range[1],
            )

        return adversarial_img

    def _compute_difference(
        self, adversarial_img: np.ndarray, original_img: np.ndarray
    ) -> np.ndarray:
        """Compute and normalize difference between adversarial and original image."""
        # Compute difference using configured method
        if callable(self.difference_calculation):
            difference = self.difference_calculation(adversarial_img, original_img)
        else:
            difference = ImageComparator().compare(
                adversarial_img,
                original_img,
                method=self.difference_calculation,
            )

        # Normalize using scaler
        if self.scaler is not None:
            difference = self.scaler.fit_transform(difference.reshape(-1, 1)).reshape(
                difference.shape
            )

        # Clip difference
        if self.clipping_range is not None:
            difference = np.clip(
                difference, self.clipping_range[0], self.clipping_range[1]
            )

        return difference

    def _aggregate_differences(self, differences: list) -> np.ndarray:
        """Aggregate and normalize differences across methods."""
        aggregated = np.sum(differences, axis=0)

        if self.scaler is not None:
            aggregated = self.scaler.fit_transform(aggregated.reshape(-1, 1)).reshape(
                aggregated.shape
            )

        if self.clipping_range is not None:
            aggregated = np.clip(
                aggregated,
                self.clipping_range[0],
                self.clipping_range[1],
            )

        return aggregated
