import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Optional
from ..core.models import ExperimentResult
from ..metrics.utils import normalize_diff


class Plotter:
    """
    Visualize adversarial results matching the specific style requested.
    """

    def plot(
        self,
        results: List[ExperimentResult],
        figsize: Optional[tuple] = None,
        cmap: str = "hot",
        show: bool = True,
        class_names: Optional[Dict[int, str]] = None,
    ) -> Optional[Figure]:
        if not results:
            return None

        n_images = len(results)
        # Determine number of attacks from the first result
        first_result = results[0]
        n_attacks = len(first_result.attacks)

        # Check if we should plot difference maps (if any attack has one)
        has_diffs = any(
            atk.difference_map is not None for res in results for atk in res.attacks
        )

        # Only show aggregated diff if more than 1 attack
        show_aggregated = has_diffs and n_attacks > 1

        # Logic for columns: Original + (Adversarial + Diff) * n_attacks + Aggregated Diff
        cols_per_attack = 2 if has_diffs else 1
        n_cols = 1 + n_attacks * cols_per_attack + (1 if show_aggregated else 0)

        if figsize is None:
            # Adjust figsize to look square-ish per subplot like in the image
            figsize = (2.5 * n_cols, 2.5 * n_images)

        fig, axes = plt.subplots(n_images, n_cols, figsize=figsize, squeeze=False)

        # Styling constants
        FONT_SIZE_TITLE = 9
        FONT_SIZE_LABEL = 8

        for idx, res in enumerate(results):
            # ---------------------------------------------------------
            # 1. Original Image
            # ---------------------------------------------------------
            ax = axes[idx, 0]
            self._plot_image(ax, res.original_image)

            # Title: "Original"
            ax.set_title("Original", fontsize=FONT_SIZE_TITLE, fontweight="bold")

            # XLabel: "GT: class_name (class_idx)"
            gt_text = ""
            if res.ground_truth is not None:
                gt_name = (
                    class_names.get(int(res.ground_truth), "?") if class_names else "?"
                )
                gt_text = f"GT: {gt_name} ({res.ground_truth})"

            ax.set_xlabel(gt_text, fontsize=FONT_SIZE_LABEL)

            diff_accumulator = []  # Store difference maps for aggregation

            current_col = 1
            for atk in res.attacks:
                # ---------------------------------------------------------
                # 2. Adversarial Image
                # ---------------------------------------------------------
                ax_adv = axes[idx, current_col]
                self._plot_image(ax_adv, atk.adversarial_image)

                # Title: "Adversarial (MethodName)"
                ax_adv.set_title(
                    f"Adversarial ({atk.method_name})",
                    fontsize=FONT_SIZE_TITLE,
                    fontweight="bold",
                )

                # XLabel: "Pred: class_name (class_idx)"
                pred_name = (
                    class_names.get(int(atk.prediction), "?") if class_names else "?"
                )
                pred_text = f"Pred: {pred_name} ({atk.prediction})"

                ax_adv.set_xlabel(pred_text, fontsize=FONT_SIZE_LABEL)

                current_col += 1

                # ---------------------------------------------------------
                # 3. Difference Map (Optional)
                # ---------------------------------------------------------
                if has_diffs:
                    ax_diff = axes[idx, current_col]

                    # Logic adjustment: Only show if attack was successful AND has actual differences
                    should_show_diff = False
                    if atk.success and (atk.difference_map is not None):
                        # Check if the difference map has non-zero values
                        if np.any(atk.difference_map > 0):
                            should_show_diff = True

                    if should_show_diff:
                        # Normalize difference map before plotting and aggregation
                        normalized_diff = normalize_diff(atk.difference_map)
                        im = self._plot_diff(ax_diff, normalized_diff, cmap=cmap)

                        # Collect for aggregation
                        diff_accumulator.append(normalized_diff)

                        # Title: "Difference (MethodName)"
                        ax_diff.set_title(
                            f"Difference ({atk.method_name})",
                            fontsize=FONT_SIZE_TITLE,
                            fontweight="bold",
                        )

                        # Add colorbar
                        cbar = plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=6)

                    else:
                        ax_diff.axis("off")  # Hide if no real difference

                    current_col += 1

            # ---------------------------------------------------------
            # 4. Aggregated Difference Map (Last Column)
            # ---------------------------------------------------------
            if show_aggregated:
                ax_agg = axes[idx, current_col]
                if diff_accumulator:
                    # Sum already-normalized difference maps
                    total_diff = np.zeros_like(diff_accumulator[0], dtype=np.float32)
                    for d in diff_accumulator:
                        total_diff += d.astype(np.float32)

                    # Normalize the aggregated sum
                    total_diff = normalize_diff(total_diff)

                    im = self._plot_diff(ax_agg, total_diff, cmap=cmap)
                    ax_agg.set_title(
                        "Aggregated Diff", fontsize=FONT_SIZE_TITLE, fontweight="bold"
                    )
                    cbar = plt.colorbar(im, ax=ax_agg, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=6)
                else:
                    ax_agg.axis("off")

        plt.tight_layout()
        if show:
            plt.show()

        return fig

    def _plot_image(self, ax, img: np.ndarray):
        # Validate type
        img_to_plot = img
        if img.dtype != np.uint8 and img.max() > 1.0:
            img_to_plot = img.astype(np.uint8)

        if len(img_to_plot.shape) > 2 and img_to_plot.shape[-1] == 1:
            ax.imshow(img_to_plot.squeeze(), cmap="gray")
        else:
            ax.imshow(img_to_plot)

        # Remove ticks but keep frame?
        # Image shows frame (box) around images.
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_diff(self, ax, diff: np.ndarray, cmap: str):
        # Normalize for visualization if not already?
        # Metrics return uint8 [0, 255].
        # We should plot it in [0, 255] or [0, 1].
        # imshow handles data range automatically if it floats or ints.

        im_data = diff

        if len(diff.shape) > 2 and diff.shape[-1] == 3:
            # Convert to grayscale magnitude for heatmap
            # Assuming typically differences are magnitude-like.
            im_data = np.mean(diff, axis=-1)
        else:
            im_data = diff.squeeze()

        # Normalize to 0-1 for colorbar consistency usually looks better?
        # Or keep raw values. Let"s keep raw but if it is uint8, matplotlib expects 0-255.
        im = ax.imshow(im_data, cmap=cmap)

        ax.set_xticks([])
        ax.set_yticks([])
        return im
