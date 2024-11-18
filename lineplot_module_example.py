import numpy as np
import custom_lineplots as clp

x = np.linspace(0, 2 * np.pi)  # Data from 0 to 2π
y_list = [np.sin(x + i * np.pi / 4) for i in range(3)]  # 3 sine waves

xticks = np.arange(0, np.pi * 2 + np.pi / 2, np.pi / 2)
xtick_labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

linestyle_dict = {
    "markerfacecolor": ["red", "green", "blue"],
    "markeredgecolor": ["black", "darkviolet", "darkorange"],
    "linewidth": [2, 1.5, 1],
    "linestyle": ["-", "--", "-."],
    "alpha": [0.8, 0.6, 0.4],
    "marker": ["o", "s", "D"],
}

my_custom_cycler = clp.get_custom_cycler(cmap_name="cool",n_colors=3)

# Generate global rcParams
custom_params = clp.get_rc_params(
    preset="default_params",
    thin_border=True,
    custom_cycler=my_custom_cycler,
)

clp.custom_plot(
    x,
    y_list,
    labels=[r"$\sin(x)$", r"$\sin(x + \frac{\pi}{4})$", r"$\sin(x + \frac{\pi}{2})$"],
    xlabel=r"$x$",
    ylabel=r"$\sin(x)$",
    title="Sine Waves with Custom Styling and $\pi$ Ticks",
    title_fontsize=18,
    title_pad=15,
    xticks=xticks,
    xtick_labels=xtick_labels,
    grid=True,
    aspect="equal",
    linestyle_dict=linestyle_dict,
    rc_params=custom_params,  # Ensure rcParams are applied
    savefig="sine_waves_custom_styling",
    save_formats=["png", "pdf"],
    )