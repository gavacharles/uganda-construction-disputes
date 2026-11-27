"""Credit line on every visualisation (user requirement): 'Visualisation: Charles Gava' (+ organisation for H&Q work)."""
AUTHOR = "Charles Gava"


def credit(fig, org=None, fontsize=None, color="#3f3d39", x=0.992, y=0.008):
    """Add a clearly legible credit at the bottom right of a figure (scaled to the figure height); it persists across
    ax.clear() in animations."""
    if fontsize is None:
        fontsize = max(8.5, min(12.0, 1.45 * fig.get_figheight()))
    text = f"Visualisation: {AUTHOR}" + (f", {org}" if org else "")
    fig.text(x, y, text, ha="right", va="bottom", fontsize=fontsize, color=color, fontweight="bold", zorder=10)
    return fig
