"""data/colour_check.py: hue families and near-highlight counts on drawn images."""

from PIL import Image, ImageDraw

from data import colour_check


def _img(colour):
    im = Image.new("RGBA", (60, 40), (0, 0, 0, 0))                      # transparent, like Geometry3K
    ImageDraw.Draw(im).line([(5, 20), (55, 20)], fill=colour, width=3)
    return im


def test_hue_families_and_near_highlight():
    red, teal, hl = _img((220, 30, 30)), _img((0, 142, 68)), _img((0, 200, 0))
    assert colour_check.hue_families(red)["red"] > 0 and colour_check.hue_families(red)["green"] == 0
    assert colour_check.hue_families(teal)["green"] > 0                 # textbook green is green-family ink ...
    assert colour_check.near_colour(teal, (0, 200, 0), 80) == 0          # ... but not near the highlight
    assert colour_check.near_colour(hl, (0, 200, 0), 40) >= colour_check.MIN_PIXELS
    assert colour_check.near_colour(_img((0, 0, 0)), (0, 200, 0), 100) == 0
    rows = {r["test"]: r for r in colour_check.table([red, teal, hl], (0, 200, 0))}
    assert rows["hue_red"]["images_with"] == 1 and rows["hue_green"]["images_with"] == 2
    assert rows["near_highlight_r80"]["images_with"] == 1 and rows["near_highlight_r80"]["pct"] == 33.3
