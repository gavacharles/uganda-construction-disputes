"""Bring older animations (made outside the pipelines) up to standard: resample to a smooth 30 fps (each source frame
held, then dissolved into the next with the ffmpeg 'framerate' filter), burn in the credit ('Visualisation: Charles
Gava', bold, bottom right) and rebuild the GIF preview from the result. Originals are replaced in place; GIF copies
elsewhere (e.g. outputs/writing/substack) are refreshed from the new GIF.

Usage: python -m src.viz.stamp_video outputs/animations [--copies outputs/writing/substack]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.viz.credit import AUTHOR
from src.viz.stamp_credit import FONT


def credit_png(width, height, path, org=None):
    size = max(16, int(height * 0.024))
    text = f"Visualisation: {AUTHOR}" + (f", {org}" if org else "")
    font = ImageFont.truetype(FONT, size)
    tw = int(ImageDraw.Draw(Image.new("RGBA", (1, 1))).textlength(text, font=font))
    pad = int(size * 0.45)
    img = Image.new("RGBA", (tw + 2 * pad, size + 2 * pad), (255, 255, 255, 215))
    ImageDraw.Draw(img).text((pad, pad - 2), text, fill=(63, 61, 57, 255), font=font)
    img.save(path)
    return img.size


def probe(mp4):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(mp4)],
                         capture_output=True, text=True, check=True).stdout.strip().split(",")
    num, den = out[2].split("/")
    return int(out[0]), int(out[1]), float(num) / float(den)


def process(mp4, org=None):
    w, h, fps = probe(mp4)
    with tempfile.TemporaryDirectory() as td:
        badge = Path(td) / "credit.png"
        bw, bh = credit_png(w, h, badge, org)
        margin = int(h * 0.012)
        hold = "interp_start=150:interp_end=250" if fps < 5 else "interp_start=15:interp_end=240"
        tmp = Path(td) / "out.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-i", str(badge), "-filter_complex",
                        f"[0:v]framerate=fps=30:{hold}:scene=100[v];"
                        f"[v][1:v]overlay=W-w-{margin}:H-h-{margin}[o]", "-map", "[o]", "-c:v", "libx264",
                        "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                       check=True)
        tmp.replace(mp4)
    gif = mp4.with_suffix(".gif")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    "fps=15,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];"
                    "[b][p]paletteuse=dither=sierra2_4a", str(gif)], check=True)
    return gif, fps


def main(argv):
    src = Path(argv[0])
    copies = Path(argv[argv.index("--copies") + 1]) if "--copies" in argv else None
    for mp4 in sorted(src.glob("*.mp4")):
        gif, fps = process(mp4)
        print(f"{mp4.name}: {fps:g} fps -> 30 fps, credit burned in; {gif.name} rebuilt")
        if copies and (copies / gif.name).exists():
            (copies / gif.name).write_bytes(gif.read_bytes())
            print(f"  refreshed {copies / gif.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
