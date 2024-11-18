#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom plotting utilities with rcParams presets and cycler support.

Author: Secra
Date: 2024-11-17
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
from cycler import cycler

# Check for optional dependencies
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# %% Presets
def default_params():
    """Default parameters for data visualization."""
    return {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 3.0,
        "xtick.major.pad": 8.0,
        "ytick.major.pad": 8.0,
        "xtick.major.size": 8.0,
        "ytick.major.size": 8.0,
        "xtick.minor.size": 8.0,
        "ytick.minor.size": 8.0,
        "xtick.major.width": 3.0,
        "ytick.major.width": 3.0,
        "xtick.labelsize": 16.0,
        "ytick.labelsize": 16.0,
        "font.size": 22.0,
        "legend.fontsize": 16.0,
        "legend.loc": "best",
        "legend.framealpha": 1.0,
        "figure.figsize": (8, 6),
        "grid.linewidth": 1.0,
        "grid.linestyle": "--",
        "lines.linestyle": "-",
        "lines.linewidth": 3.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }


def paper_params():
    """Parameters for publication-quality figures."""
    return {
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 1.5,
        "xtick.major.pad": 8.0,
        "ytick.major.pad": 8.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 4.0,
        "ytick.minor.size": 4.0,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "font.size": 12.0,
        "legend.fontsize": 10.0,
        "legend.loc": "best",
        "legend.framealpha": 1.0,
        "figure.figsize": (4, 4),
        "grid.linewidth": 0.5,
        "grid.linestyle": "--",
        "lines.linestyle": "-",
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    }


# %% Utility Functions
def get_custom_cycler(cmap_name=None, n_colors=9, skip_first=False, use_seaborn=False):
    """
    Create a custom color cycler with optional skipping of the first color.

    Parameters:
    - cmap_name (str): Name of the colormap to use.
    - n_colors (int): Number of colors to include in the cycler.
    - skip_first (bool): Whether to skip the first color in the colormap.
    - use_seaborn (bool): Use Seaborn colormap if True.

    Returns:
    - cycler: A matplotlib cycler object.
    """
    if not cmap_name:
        colors = [
            "b", 
            "lime",
            "r",
            "darkviolet",
            "darkorange",
            "deepskyblue",
            "gold",
            "k",
            "pink",
        ]
    elif use_seaborn and SEABORN_AVAILABLE:
        palette = sns.color_palette(cmap_name, n_colors + (1 if skip_first else 0))
        colors = palette.as_hex()[1:] if skip_first else palette.as_hex()
    else:
        cmap = plt.get_cmap(cmap_name)
        start = 1 / n_colors if skip_first else 0
        colors = [
            mplcolors.to_hex(cmap(start + i * (1 - start) / (n_colors - 1)))
            for i in range(n_colors)
        ]

    return cycler(color=colors)


def match_cycler_lengths(color_cycler, linestyles):
    """
    Ensures the lengths of the color cycler and linestyles match.

    Parameters:
    - color_cycler (cycler): Cycler object for colors.
    - linestyles (list): List of linestyles.

    Returns:
    - cycler: Combined cycler with matching lengths for color and linestyle.
    """
    # Extract colors from the color cycler
    colors = [c["color"] for c in color_cycler]
    n_colors = len(colors)
    n_linestyles = len(linestyles)

    if n_colors > n_linestyles:
        # Repeat linestyles if fewer than colors
        linestyles = (linestyles * (n_colors // n_linestyles + 1))[:n_colors]
    elif n_colors < n_linestyles:
        # Truncate linestyles if more than colors
        linestyles = linestyles[:n_colors]

    # Return combined cycler
    return cycler("color", colors) + cycler("linestyle", linestyles)


def get_rc_params(
    preset="default_params",
    font="DejaVu Serif",
    thin_border=False,
    linestyle_cycle=None,
    custom_cycler=None,  # Accepts a pre-defined color cycler
    custom_linestyles=None,
    update_existing=False,
):
    """
    Generate or update rcParams based on a preset and additional options.

    Parameters:
    - preset (str): Preset to use ('default_params' or 'paper_params').
    - font (str): Font family to use (e.g., 'Times New Roman').
    - thin_border (bool): Use thinner borders if True.
    - linestyle_cycle (bool): Whether to include a linestyle cycler. 
    Nice when just wanting to use default. Can also set manually in custom_plot
    - custom_cycler (cycler): Predefined color cycler (used for colors).
    - custom_linestyles (list): Custom linestyle cycle (used if linestyle_cycle=True).
    - update_existing (bool): Update global rcParams if True.

    Returns:
    - dict: Updated rcParams dictionary.
    """
    presets = {
        "default_params": default_params(),
        "paper_params": paper_params(),
    }

    if preset not in presets:
        raise ValueError(f"Preset '{preset}' not recognized. Use 'default_params' or 'paper_params'.")

    # Start with the selected preset
    rc_params = presets[preset]

    # Update font family
    rc_params["font.family"] = font

    # Apply thin borders
    if thin_border:
        rc_params["axes.linewidth"] /= 2
        rc_params["xtick.major.width"] /= 2
        rc_params["ytick.major.width"] /= 2
        rc_params["grid.linewidth"] /= 2

    # Handle cyclers: Combine linestyles with custom or default color cycler
    if linestyle_cycle:
        default_linestyles = [
            "solid",
            (0, (5, 5)),
            "dashdot",
            (0, (3, 5, 1, 5, 1, 5)),
            "dotted",
            (0, (3, 10, 1, 10, 1, 10)),
            (0, (5, 1)),
            (0, (3, 1, 1, 1)),
            (0, (5, 10)),
        ]
        linestyles = custom_linestyles or default_linestyles
        color_cycler = custom_cycler or get_custom_cycler()  # Use provided cycler or default

        # Ensure lengths match using helper function
        rc_params["axes.prop_cycle"] = match_cycler_lengths(color_cycler, linestyles)
        rc_params["legend.handlelength"] = 3.75 if preset == "default_params" else 3.7

    # Use custom color cycler without linestyles if no linestyle_cycle is specified
    elif custom_cycler:
        rc_params["axes.prop_cycle"] = custom_cycler

    # Update rcParams globally if requested
    if update_existing:
        plt.rcParams.update(rc_params)
        return plt.rcParams
    else:
        return rc_params



# %% Plotting Function
def custom_plot(
    x,
    y_list,
    labels=None,
    xlabel="",
    ylabel="",
    title="",
    xlim=None,
    ylim=None,
    grid=True,
    aspect="auto",
    savefig=None,
    rc_params=None,
    local_rc_params=None,
    save_formats=None,
    show=True,
    title_fontsize=16,
    title_pad=10,
    x_major_locator=None,
    y_major_locator=None,
    xscale="linear",
    yscale="linear",
    xticks=None,
    xtick_labels=None,
    yticks=None,  # Added yticks
    ytick_labels=None,  # Added ytick_labels
    linestyle_dict=None,
    **plot_kwargs,
):
    """
    Plot multiple lines with customizable parameters.

    Parameters:
    - x (array-like): X-axis data.
    - y_list (list of array-like): List of Y-axis data for each line.
    - labels (list of str, optional): Labels for the legend.
    - xlabel (str, optional): Label for the X-axis.
    - ylabel (str, optional): Label for the Y-axis.
    - title (str, optional): Title of the plot.
    - xlim (tuple, optional): X-axis limits (min, max).
    - ylim (tuple, optional): Y-axis limits (min, max).
    - grid (bool, optional): Whether to show grid lines.
    - aspect (str or float, optional): Aspect ratio of the plot.
    - savefig (str, optional): Filename for saving the plot (excluding extension).
    - rc_params (dict, optional): Global rcParams to update.
    - local_rc_params (dict, optional): Temporary rcParams for this plot.
    - save_formats (list of str, optional): File formats to save the plot (e.g., ['png', 'pdf']).
    - show (bool, optional): Whether to display the plot.
    - title_fontsize (int, optional): Font size for the title.
    - title_pad (float, optional): Padding between the title and the plot.
    - x_major_locator (float, optional): Major tick interval for the X-axis.
    - y_major_locator (float, optional): Major tick interval for the Y-axis.
    - xscale (str, optional): Scale for the X-axis ('linear', 'log', etc.).
    - yscale (str, optional): Scale for the Y-axis ('linear', 'log', etc.).
    - xticks (array-like, optional): Custom X-tick positions.
    - xtick_labels (list of str, optional): Custom labels for X-ticks.
    - yticks (array-like, optional): Custom Y-tick positions.
    - ytick_labels (list of str, optional): Custom labels for Y-ticks.
    - linestyle_dict (dict, optional): Line-specific customizations (e.g., colors, markers).
    - **plot_kwargs: Additional arguments passed to `plt.plot`.

    Returns:
    - None
    """
    # Update global rcParams if provided
    if rc_params:
        plt.rcParams.update(rc_params)

    # Use rc_context only if local_rc_params is provided
    context_manager = plt.rc_context(local_rc_params) if local_rc_params else None

    if context_manager:
        with context_manager:
            fig, ax = plt.subplots()
            _plot(
                ax,
                x,
                y_list,
                labels,
                xlabel,
                ylabel,
                title,
                xlim,
                ylim,
                grid,
                aspect,
                xticks,
                xtick_labels,
                yticks,
                ytick_labels,
                xscale,
                yscale,
                linestyle_dict,
                title_fontsize,
                title_pad,
                **plot_kwargs,
            )
    else:
        fig, ax = plt.subplots()
        _plot(
            ax,
            x,
            y_list,
            labels,
            xlabel,
            ylabel,
            title,
            xlim,
            ylim,
            grid,
            aspect,
            xticks,
            xtick_labels,
            yticks,
            ytick_labels,
            xscale,
            yscale,
            linestyle_dict,
            title_fontsize,
            title_pad,
            **plot_kwargs,
        )

    # Save the plot in specified formats
    if savefig and save_formats:
        for fmt in save_formats:
            plt.savefig(f"{savefig}.{fmt}", bbox_inches="tight")

    # Display the plot
    if show:
        plt.show()


def _plot(
    ax,
    x,
    y_list,
    labels,
    xlabel,
    ylabel,
    title,
    xlim,
    ylim,
    grid,
    aspect,
    xticks,
    xtick_labels,
    yticks,
    ytick_labels,
    xscale,
    yscale,
    linestyle_dict,
    title_fontsize,
    title_pad,
    **plot_kwargs,
):
    """
    Internal function to handle the actual plotting logic.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object for the plot.
    - x (array-like): X-axis data.
    - y_list (list of array-like): List of Y-axis data for each line.
    - labels (list of str, optional): Labels for the legend.
    - xlabel (str, optional): Label for the X-axis.
    - ylabel (str, optional): Label for the Y-axis.
    - title (str, optional): Title of the plot.
    - xlim (tuple, optional): X-axis limits (min, max).
    - ylim (tuple, optional): Y-axis limits (min, max).
    - grid (bool, optional): Whether to show grid lines.
    - aspect (str or float, optional): Aspect ratio of the plot.
    - xticks (array-like, optional): Custom X-tick positions.
    - xtick_labels (list of str, optional): Custom labels for X-ticks.
    - yticks (array-like, optional): Custom Y-tick positions.
    - ytick_labels (list of str, optional): Custom labels for Y-ticks.
    - xscale (str, optional): Scale for the X-axis ('linear', 'log', etc.).
    - yscale (str, optional): Scale for the Y-axis ('linear', 'log', etc.).
    - linestyle_dict (dict, optional): Line-specific customizations.
    - title_fontsize (int, optional): Font size for the title.
    - title_pad (float, optional): Padding between the title and the plot.
    - **plot_kwargs: Additional arguments passed to `plt.plot`.

    Returns:
    - None
    """
    for i, y in enumerate(y_list):
        # Start with global plot_kwargs
        line_kwargs = plot_kwargs.copy()

        # Apply per-line properties from linestyle_dict
        if linestyle_dict:
            for key, values in linestyle_dict.items():
                # Dynamically add line-specific properties
                line_kwargs[key] = values[i] if i < len(values) else None

        # Get the label for the current line
        label = labels[i] if labels else None

        # Plot the line with updated kwargs
        ax.plot(x, y, label=label, **line_kwargs)

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the plot title
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)

    # Configure the grid, limits, and aspect ratio
    if grid:
        ax.grid()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_aspect(aspect)

    # Set axis scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Set custom X-ticks and labels
    if xticks is not None:
        ax.set_xticks(xticks)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)

    # Set custom Y-ticks and labels
    if yticks is not None:
        ax.set_yticks(yticks)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)

    # Add legend if labels are provided
    if labels:
        ax.legend()

# %% Example Usage
if __name__ == "__main__":
    # Define data
    x = np.linspace(0, 10, 100)
    y_list = [np.sin(x + i) for i in range(5)]  # 5 sine waves with phase shifts
    labels = [f"Line {i + 1}" for i in range(5)]

    # Define a custom cycler
    my_custom_cycler = get_custom_cycler(cmap_name="plasma", n_colors=5)

    # Generate rcParams with custom cycler and linestyles
    custom_params = get_rc_params(
        preset="default_params",
        custom_cycler=my_custom_cycler,  # Custom color cycler
        linestyle_cycle=True,            # Combine with linestyles
        thin_border=True,
    )

    # Plot using custom parameters
    custom_plot(
        x,
        y_list,
        labels=labels,
        xlabel="Time (s)",
        ylabel="Amplitude",
        title="Custom Cycler and Linestyle Example",
        x_major_locator=2,  # Major ticks every 2 units on x-axis
        y_major_locator=0.5,  # Major ticks every 0.5 units on y-axis
        local_rc_params=custom_params,
        title_fontsize=18,  # Font size for the title
        title_pad=15,  # Padding between title and plot
        aspect=1.5,  # Aspect ratio
        savefig="custom_plot",  # Save the figure with this base filename
        save_formats=["png", "pdf"],  # Save as both PNG and PDF
        grid=True,
        linewidth=2.5,  # Pass additional kwargs to control line width
        marker="o",  # Pass additional kwargs to add markers
        alpha=0.8,  # Pass additional kwargs for line transparency
    )

#Example 2: paper params, default, seaborn and black and white color options
#Times New Roman used, might not be installed

x = np.linspace(0, 1, 100)

# Generate y data
n_lines = 10
y_list = [0.25 * (i + 1) * np.exp(x) for i in range(n_lines)]
labels = [f"Line {i+1}" for i in range(n_lines)]

# Define a custom cycler
my_custom_cycler = get_custom_cycler() #use default color cycler

# Generate rcParams with custom cycler and linestyles
custom_params = get_rc_params(
    preset="paper_params",
    custom_cycler=my_custom_cycler,  # Custom color cycler
    linestyle_cycle=False,            # Combine with linestyles
    thin_border=False,
    font='Times New Roman' # Font selection, must be installed
)

# Plot using custom parameters
custom_plot(
    x,
    y_list,
    labels=labels,
    xlabel="\n" + r"$\mu \ [Pa \cdot s]$", # test latex usage and newline char
    ylabel=r"$q_w$ [W/cm$^2$]" + "\n",
    x_major_locator=.25,  # Major ticks every .25 units on x-axis
    y_major_locator=5.0,  # Major ticks every 5.0 units on y-axis
    local_rc_params=custom_params,
    xlim=(0, 1), # plot axis limits
    ylim=(0, 10),
    aspect='auto',  # Aspect ratio
    savefig="custom_plot2",  # Save the figure with this base filename
    save_formats=["png", "pdf"],  # Save as both PNG and PDF
    grid=True,
)

# Define a custom cycler
my_custom_cycler = get_custom_cycler(cmap_name='Greys',skip_first=True) #use black and white

# Generate rcParams with custom cycler and linestyles
custom_params = get_rc_params(
    preset="paper_params",
    custom_cycler=my_custom_cycler,  # Custom color cycler
    linestyle_cycle=True,            # Combine with linestyles
    thin_border=False,
    font='Times New Roman' # Font selection, must be installed
)

# Plot using custom parameters
custom_plot(
    x,
    y_list,
    labels=labels,
    xlabel="\n" + r"$\mu \ [Pa \cdot s]$", # test latex usage and newline char
    ylabel=r"$q_w$ [W/cm$^2$]" + "\n",
    x_major_locator=.25,  # Major ticks every .25 units on x-axis
    y_major_locator=5.0,  # Major ticks every 5.0 units on y-axis
    local_rc_params=custom_params,
    xlim=(0, 1), # plot axis limits
    ylim=(0, 10),
    aspect='auto',  # Aspect ratio
    savefig="custom_plot3",  # Save the figure with this base filename
    save_formats=["png", "pdf"],  # Save as both PNG and PDF
    grid=True,
)

# y data
y_list = [np.exp(x*(i+1)) for i in range(n_lines)]
labels = [rf"$e^{{{i + 1}x}}$" for i in range(n_lines)]

# Define a custom cycler
my_custom_cycler = get_custom_cycler(cmap_name='magma',use_seaborn=True) #use seaborn cmap

# Generate rcParams with custom cycler and linestyles
custom_params = get_rc_params(
    preset="paper_params",
    custom_cycler=my_custom_cycler,  # Custom color cycler
    linestyle_cycle=False,            # Combine with linestyles
    thin_border=False,
    font='Times New Roman' # Font selection, must be installed
)

# Plot using custom parameters
custom_plot(
    x,
    y_list,
    labels=labels,
    xlabel="\n" + r"$\mu \ [Pa \cdot s]$", # test latex usage and newline char
    ylabel=r"$e^{\text{k}x} \Rightarrow \text{k}x$" + "\n",
    yscale='log',
    # x_major_locator=.25,  # Major ticks every .25 units on x-axis
    # y_major_locator=5.0,  # Major ticks every 5.0 units on y-axis
    local_rc_params=custom_params,
    # xlim=(0, 1), # plot axis limits
    # ylim=(0, 10),
    aspect='auto',  # Aspect ratio
    savefig="custom_plot4",  # Save the figure with this base filename
    save_formats=["png", "pdf"],  # Save as both PNG and PDF
    grid=True,
)