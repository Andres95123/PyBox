from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import warnings


class Plotter:
    """
    Plotter: Visualize adversarial example analysis results using matplotlib.

    This class takes the output from AdversarialExplainer.explain() and creates
    comprehensive visualizations showing original images, adversarial variants,
    difference maps, and aggregated differences across attack methods.

    The output is organized in a grid where each row represents an image and
    each column represents different components (original, adversarials, differences, aggregated).
    """

    def __init__(self) -> None:
        """Initialize the Plotter."""

    def plot(
        self,
        explain_results: dict,
        figsize: tuple[int, int] | None = None,
        cmap: str = "hot",
        show: bool = True,
    ) -> Figure:
        """
        Create a comprehensive visualization of adversarial example analysis.

        Args:
            explain_results (dict): Output dictionary from AdversarialExplainer.explain()
                containing 'originals', 'adversarials', 'differences', 'aggregated', etc.
            figsize (tuple, optional): Figure size as (width, height). If None, automatically
                calculated as (3 * num_cols, 3 * num_rows).
            cmap (str): Colormap for difference visualizations. Default: 'hot'.
                Can be any valid matplotlib colormap (e.g., 'hot', 'viridis', 'gray', etc.).
                Applied only to difference maps during visualization. Original and adversarial
                images are always displayed in RGB.
            show (bool): Whether to display the plot. Default: True.

        Returns:
            Figure: The matplotlib figure object.

        Raises:
            ValueError: If explain_results is missing required keys or has incompatible shapes.
        """

        # Validate input
        if not isinstance(explain_results, dict):
            raise ValueError(
                "explain_results must be a dictionary from AdversarialExplainer.explain()"
            )

        required_keys = {"originals", "adversarials"}
        if not required_keys.issubset(explain_results.keys()):
            raise ValueError(f"explain_results must contain keys: {required_keys}")

        originals = explain_results["originals"]
        adversarials = explain_results["adversarials"]
        differences = explain_results.get("differences", None)
        aggregated = explain_results.get("aggregated", None)
        method_names = explain_results.get("method_names", [])
        ground_truth = explain_results.get("ground_truth", None)
        predictions = explain_results.get("predictions", None)

        n_images, n_methods = adversarials.shape[0], adversarials.shape[1]

        # Calculate number of columns
        # Layout: Original | [Adversarial_1, Diff_1] | [Adversarial_2, Diff_2] | ... | Aggregated
        cols_per_method = 1 + (1 if differences is not None else 0)
        num_cols = (
            1 + n_methods * cols_per_method + (1 if aggregated is not None else 0)
        )

        # Calculate figure size
        if figsize is None:
            figsize = (3 * num_cols, 3 * n_images)

        # Create figure and axes
        fig, axes = plt.subplots(
            nrows=n_images,
            ncols=num_cols,
            figsize=figsize,
        )

        # Handle axes array conversion
        axes = self._normalize_axes(axes, n_images, num_cols)

        # Process each image row
        for i in range(n_images):
            self._plot_row(
                axes,
                i,
                num_cols,
                n_methods,
                originals,
                adversarials,
                differences,
                aggregated,
                method_names,
                ground_truth,
                predictions,
                cmap,
            )

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def _normalize_axes(self, axes, n_images: int, num_cols: int) -> np.ndarray:
        """Convert matplotlib axes to 2D numpy array."""
        if n_images == 1 and num_cols == 1:
            return np.array([[axes]])
        elif n_images == 1:
            return np.array([axes])
        elif num_cols == 1:
            return np.array([[ax] for ax in axes])
        else:
            return np.array(axes)

    def _plot_row(
        self,
        axes: np.ndarray,
        row_idx: int,
        num_cols: int,
        n_methods: int,
        originals: np.ndarray,
        adversarials: np.ndarray,
        differences: np.ndarray | None,
        aggregated: np.ndarray | None,
        method_names: list,
        ground_truth: np.ndarray | None,
        predictions: np.ndarray | None,
        cmap: str,
    ) -> None:
        """Plot a single image row."""
        # Original image
        ax = axes[row_idx, 0]
        self._plot_image(ax, originals[row_idx], title="Original")

        if ground_truth is not None:
            ax.set_xlabel(f"GT: {ground_truth[row_idx]}", fontsize=9)

        # Adversarial and difference columns
        cols_per_method = 1 + (1 if differences is not None else 0)

        for j in range(n_methods):
            self._plot_method_columns(
                axes,
                row_idx,
                j,
                cols_per_method,
                adversarials,
                differences,
                method_names,
                predictions,
                cmap,
            )

        # Aggregated differences at the end
        if aggregated is not None:
            agg_col = num_cols - 1
            ax = axes[row_idx, agg_col]
            self._plot_image(
                ax,
                aggregated[row_idx],
                title="Sum of Differences",
                cmap=cmap,
            )

    def _plot_method_columns(
        self,
        axes: np.ndarray,
        row_idx: int,
        method_idx: int,
        cols_per_method: int,
        adversarials: np.ndarray,
        differences: np.ndarray | None,
        method_names: list,
        predictions: np.ndarray | None,
        cmap: str,
    ) -> None:
        """Plot adversarial and difference columns for a method."""
        adv_col = 1 + method_idx * cols_per_method
        method_name = (
            method_names[method_idx]
            if method_idx < len(method_names)
            else f"Method {method_idx}"
        )

        # Adversarial image
        ax = axes[row_idx, adv_col]
        self._plot_image(
            ax, adversarials[row_idx, method_idx], title=f"Adversarial ({method_name})"
        )

        if predictions is not None:
            ax.set_xlabel(f"Pred: {predictions[row_idx, method_idx]}", fontsize=9)

        # Difference map
        if differences is not None:
            diff_col = adv_col + 1
            ax = axes[row_idx, diff_col]
            self._plot_image(
                ax,
                differences[row_idx, method_idx],
                title=f"Difference ({method_name})",
                cmap=cmap,
            )

    def plot_single(
        self,
        image_idx: int,
        explain_results: dict,
        figsize: tuple[int, int] | None = None,
        cmap: str = "hot",
        show: bool = True,
    ) -> Figure:
        """
        Create a visualization for a single image.

        Args:
            image_idx (int): Index of the image to plot.
            explain_results (dict): Output dictionary from AdversarialExplainer.explain().
            figsize (tuple, optional): Figure size as (width, height).
            cmap (str): Colormap for difference visualizations. Default: 'hot'.
            show (bool): Whether to display the plot. Default: True.

        Returns:
            Figure: The matplotlib figure object.
        """

        n_images = explain_results["originals"].shape[0]
        if not 0 <= image_idx < n_images:
            raise ValueError(f"image_idx must be between 0 and {n_images - 1}")

        # Create a single-image subset
        single_result = {
            "originals": explain_results["originals"][image_idx : image_idx + 1],
            "adversarials": explain_results["adversarials"][image_idx : image_idx + 1],
            "method_names": explain_results.get("method_names", []),
        }

        if "differences" in explain_results:
            single_result["differences"] = explain_results["differences"][
                image_idx : image_idx + 1
            ]

        if "aggregated" in explain_results:
            single_result["aggregated"] = explain_results["aggregated"][
                image_idx : image_idx + 1
            ]

        if "ground_truth" in explain_results:
            single_result["ground_truth"] = explain_results["ground_truth"][
                image_idx : image_idx + 1
            ]

        if "predictions" in explain_results:
            single_result["predictions"] = explain_results["predictions"][
                image_idx : image_idx + 1
            ]

        return self.plot(single_result, figsize=figsize, cmap=cmap, show=show)

    def plot_comparison(
        self,
        explain_results: dict,
        method_indices: list[int] | None = None,
        figsize: tuple[int, int] | None = None,
        show: bool = True,
    ) -> Figure:
        """
        Create a comparison plot for specific attack methods.

        Args:
            explain_results (dict): Output dictionary from AdversarialExplainer.explain().
            method_indices (list[int], optional): Indices of methods to compare. If None, plot all.
            figsize (tuple, optional): Figure size as (width, height).
            show (bool): Whether to display the plot. Default: True.

        Returns:
            Figure: The matplotlib figure object.
        """

        n_methods = explain_results["adversarials"].shape[1]
        if method_indices is None:
            method_indices = list(range(n_methods))

        for idx in method_indices:
            if not 0 <= idx < n_methods:
                raise ValueError(
                    f"method_indices must be between 0 and {n_methods - 1}"
                )

        n_images = explain_results["originals"].shape[0]

        if figsize is None:
            figsize = (5 * len(method_indices), 4 * n_images)

        fig, axes = plt.subplots(
            nrows=n_images,
            ncols=len(method_indices),
            figsize=figsize,
        )

        if n_images == 1:
            axes = axes.reshape(1, -1)
        elif len(method_indices) == 1:
            axes = axes.reshape(-1, 1)

        method_names = explain_results.get("method_names", [])
        predictions = explain_results.get("predictions", None)
        adversarials = explain_results["adversarials"]

        for i in range(n_images):
            for col_idx, method_idx in enumerate(method_indices):
                self._plot_comparison_cell(
                    axes,
                    i,
                    col_idx,
                    n_images,
                    len(method_indices),
                    method_idx,
                    method_names,
                    predictions,
                    adversarials,
                )

        plt.tight_layout()

        if show:
            plt.show()

        return fig

    def _plot_comparison_cell(
        self,
        axes: np.ndarray,
        row_idx: int,
        col_idx: int,
        n_images: int,
        n_cols: int,
        method_idx: int,
        method_names: list,
        predictions: np.ndarray | None,
        adversarials: np.ndarray,
    ) -> None:
        """Helper method to plot a single cell in comparison plot."""

        if n_images == 1:
            ax = axes[col_idx]
        elif n_cols == 1:
            ax = axes[row_idx]
        else:
            ax = axes[row_idx, col_idx]

        method_name = (
            method_names[method_idx]
            if method_idx < len(method_names)
            else f"Method {method_idx}"
        )

        self._plot_image(ax, adversarials[row_idx, method_idx], title=f"{method_name}")

        if predictions is not None:
            ax.set_xlabel(f"Pred: {predictions[row_idx, method_idx]}", fontsize=9)

    @staticmethod
    def _plot_image(
        ax,
        image: np.ndarray,
        title: str = "",
        cmap: str | None = None,
    ) -> None:
        """
        Helper method to plot an image on a matplotlib axis with colormap support.

        Handles images with different channel counts:
        - 2D (H, W): Grayscale image
        - 3D with 1 channel (H, W, 1): Grayscale image
        - 3D with 3 channels (H, W, 3): RGB image
        - 3D with 4 channels (H, W, 4): ARGB image (alpha channel removed)

        Args:
            ax: Matplotlib axis object.
            image (np.ndarray): Image array to plot. Expects shape (H, W) or (H, W, C).
            title (str): Title for the subplot.
            cmap (str, optional): Colormap to use. If None:
                - For RGB images: No colormap (natural RGB display)
                - For grayscale: 'gray' colormap
                Any valid matplotlib colormap can be specified (e.g., 'hot', 'viridis', 'gray').
                When a cmap is applied to RGB images, they are converted to grayscale first.

        Raises:
            ValueError: If image has invalid dimensions or channel count.
            TypeError: If cmap is not a valid matplotlib colormap name.
        """
        # Validate and normalize image
        img_to_plot, is_grayscale, n_channels = Plotter._validate_image(image)

        # Apply visualization
        Plotter._apply_visualization(ax, img_to_plot, is_grayscale, n_channels, cmap)

        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def _validate_image(image: np.ndarray) -> tuple[np.ndarray, bool, int]:
        """
        Validate and normalize image array.

        Returns:
            Tuple of (normalized_image, is_grayscale, n_channels)
        """
        # Validate image dimensions
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Image must be 2D (H, W) or 3D (H, W, C), got shape {image.shape}"
            )

        # Handle 2D images (already grayscale)
        if image.ndim == 2:
            return image, True, 1

        # 3D image - analyze channels
        n_channels = image.shape[2]

        if n_channels == 1:
            # Grayscale with explicit channel dimension
            return image.squeeze(), True, 1
        elif n_channels == 3:
            # RGB image
            return image, False, 3
        elif n_channels == 4:
            # ARGB image - remove alpha channel
            warnings.warn(
                "Image has 4 channels (ARGB), displaying RGB only (alpha channel removed)",
                UserWarning,
            )
            return image[:, :, :3], False, 3
        else:
            raise ValueError(
                f"Image has unsupported number of channels: {n_channels}. "
                "Expected 1, 2, 3 (RGB), or 4 (ARGB) channels."
            )

    @staticmethod
    def _apply_visualization(
        ax,
        img_to_plot: np.ndarray,
        is_grayscale: bool,
        n_channels: int,
        cmap: str | None,
    ) -> None:
        """Apply visualization with appropriate colormap handling."""
        if cmap is not None:
            # User specified a colormap
            if not is_grayscale and n_channels == 3:
                # RGB image with colormap requested - convert to grayscale first
                img_gray = Plotter._rgb_to_grayscale(img_to_plot)
                ax.imshow(img_gray, cmap=cmap)
            else:
                # Grayscale image or single-channel - apply colormap directly
                ax.imshow(img_to_plot.squeeze(), cmap=cmap)
        else:
            # No colormap specified
            if is_grayscale:
                # Display grayscale with 'gray' colormap
                ax.imshow(img_to_plot, cmap="gray")
            else:
                # Display RGB naturally
                ax.imshow(img_to_plot)

    @staticmethod
    def _rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB image to grayscale using the mean of dimensions. A machine learning model
        don't has the human limitations, so the luminance formula is not correct to use"""
        return np.mean(rgb_image, axis=2)
