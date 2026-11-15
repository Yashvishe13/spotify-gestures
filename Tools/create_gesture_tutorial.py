#!/usr/bin/env python3
"""Render the hand-only gesture tutorial GIF shown to users for review.

A stylised hand (with a sleeve entering from the bottom of a camera
viewfinder) performs each gesture, drops out of view, and the command is
confirmed — the same flow the app expects.
"""

import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "gesture_tutorial.gif"
PREVIEW = ROOT / "preview"
W, H, S, FPS, N = 720, 405, 2, 18, 46
SIZE = (W * S, H * S)
FONT = "/System/Library/Fonts/Avenir Next.ttc"

BG = (11, 12, 14)
PANEL = (21, 23, 26)
LINE = (44, 47, 51)
WHITE = (244, 244, 245)
GRAY = (150, 154, 160)
DIM = (82, 86, 92)
ACCENT = (64, 214, 138)
SKIN = (238, 199, 164)
SKIN_SHADE = (214, 168, 132)
SKIN_LINE = (190, 144, 110)
SLEEVE = (58, 64, 74)
SLEEVE_SHADE = (44, 49, 57)
CUFF = (84, 92, 104)

STAGE = (30, 78, 390, 324)
CX, CY = 210, 206
UNIT = 92  # base px per hand unit

SCENES = [
    ("SWIPE LEFT", "Next track", "Raise an open hand and sweep it to the left."),
    ("SWIPE RIGHT", "Previous track", "Raise an open hand and sweep it to the right."),
    ("SWIPE UP", "Volume up", "Lift your open hand upward."),
    ("SWIPE DOWN", "Volume down", "Lower your open hand downward."),
    ("OPEN PALM", "Pause", "Hold your open palm toward the camera."),
    ("THUMBS UP", "Play", "Hold up a thumbs-up toward the camera."),
]


# ---------------------------------------------------------------- helpers
def p(v):
    return round(v * S)


def font(size, weight="regular"):
    index = {"regular": 7, "medium": 5, "demi": 2, "bold": 0, "heavy": 8}[weight]
    return ImageFont.truetype(FONT, p(size), index=index)


def text(draw, xy, value, size, color=WHITE, weight="regular", anchor="lt", alpha=255):
    draw.text((p(xy[0]), p(xy[1])), value, font=font(size, weight),
              fill=(*color, alpha), anchor=anchor)


def clamp(v):
    return max(0.0, min(1.0, v))


def ease(v):
    v = clamp(v)
    return v * v * (3 - 2 * v)


def ease_out(v):
    v = clamp(v)
    return 1 - (1 - v) ** 3


def ease_in(v):
    v = clamp(v)
    return v ** 3


def back_out(v):
    v = clamp(v)
    c = 1.6
    return 1 + (c + 1) * (v - 1) ** 3 + c * (v - 1) ** 2


# ---------------------------------------------------------------- hand sprites
SPRITE_U = 260               # sprite px per hand unit
SPRITE_BOX = (-0.75, -0.85, 0.75, 2.6)   # local units: x0, y0, x1, y1


def _sprite_canvas():
    x0, y0, x1, y1 = SPRITE_BOX
    return (round((x1 - x0) * SPRITE_U), round((y1 - y0) * SPRITE_U))


def _to_px(x, y):
    return ((x - SPRITE_BOX[0]) * SPRITE_U, (y - SPRITE_BOX[1]) * SPRITE_U)


def _capsule(draw, a, b, r, fill=255):
    (ax, ay), (bx, by) = _to_px(*a), _to_px(*b)
    rp = r * SPRITE_U
    draw.line((ax, ay, bx, by), fill=fill, width=round(2 * rp))
    for x, y in ((ax, ay), (bx, by)):
        draw.ellipse((x - rp, y - rp, x + rp, y + rp), fill=fill)


def _rrect(draw, box, r, fill=255):
    x0, y0 = _to_px(box[0], box[1])
    x1, y1 = _to_px(box[2], box[3])
    draw.rounded_rectangle((x0, y0, x1, y1), radius=r * SPRITE_U, fill=fill)


def _ellipse(draw, box, fill=255):
    x0, y0 = _to_px(box[0], box[1])
    x1, y1 = _to_px(box[2], box[3])
    draw.ellipse((x0, y0, x1, y1), fill=fill)


def _shade(mask, base, shade):
    """Flat fill with a soft lower-right rim so the shape reads as 3D."""
    size = mask.size
    shift = round(0.035 * SPRITE_U)
    inner = ImageChops.multiply(mask, ImageChops.offset(mask, -shift, -shift))
    inner = inner.filter(ImageFilter.GaussianBlur(shift * 0.6))
    out = Image.new("RGBA", size, (*shade, 0))
    out.putalpha(mask)
    top = Image.new("RGBA", size, (*base, 0))
    top.putalpha(ImageChops.multiply(inner, mask))
    out.alpha_composite(top)
    return out


def _sleeve(sprite, wrist_x0, wrist_x1):
    size = sprite.size
    m = Image.new("L", size)
    d = ImageDraw.Draw(m)
    _rrect(d, (wrist_x0 - 0.09, 0.66, wrist_x1 + 0.09, 2.7), 0.10)
    sleeve = _shade(m, SLEEVE, SLEEVE_SHADE)
    sprite.alpha_composite(sleeve)
    d = ImageDraw.Draw(sprite)
    _rrect(d, (wrist_x0 - 0.09, 0.66, wrist_x1 + 0.09, 0.76), 0.05, fill=(*CUFF, 255))


def open_hand_sprite():
    size = _sprite_canvas()
    m = Image.new("L", size)
    d = ImageDraw.Draw(m)
    _rrect(d, (-0.31, -0.12, 0.30, 0.44), 0.17)                       # palm
    _rrect(d, (-0.20, 0.30, 0.19, 0.74), 0.10)                         # wrist
    _capsule(d, (-0.215, -0.02), (-0.255, -0.56), 0.066)               # index
    _capsule(d, (-0.070, -0.05), (-0.075, -0.68), 0.068)               # middle
    _capsule(d, (0.075, -0.04), (0.095, -0.61), 0.066)                 # ring
    _capsule(d, (0.205, 0.02), (0.275, -0.41), 0.058)                  # little
    _capsule(d, (-0.22, 0.24), (-0.50, -0.07), 0.074)                  # thumb
    _ellipse(d, (-0.36, 0.06, -0.06, 0.44))                            # thumb base
    sprite = Image.new("RGBA", size)
    _sleeve(sprite, -0.20, 0.19)
    sprite.alpha_composite(_shade(m, SKIN, SKIN_SHADE))
    d = ImageDraw.Draw(sprite)
    lw = max(2, round(0.014 * SPRITE_U))
    # Two soft palm creases — just enough detail to read as a palm.
    for box, a0, a1 in (((-0.30, -0.02, 0.26, 0.34), 200, 320),
                        ((-0.12, 0.06, 0.40, 0.52), 205, 265)):
        x0, y0 = _to_px(box[0], box[1])
        x1, y1 = _to_px(box[2], box[3])
        d.arc((x0, y0, x1, y1), a0, a1, fill=(*SKIN_LINE, 150), width=lw)
    return sprite


def thumbs_up_sprite():
    size = _sprite_canvas()
    m = Image.new("L", size)
    d = ImageDraw.Draw(m)
    _rrect(d, (-0.29, -0.12, 0.25, 0.46), 0.17)                        # fist
    _rrect(d, (-0.20, 0.34, 0.15, 0.74), 0.10)                         # wrist
    for y in (-0.03, 0.105, 0.24, 0.37):                               # curled fingers
        _capsule(d, (0.02, y), (0.29, y), 0.072)
    _capsule(d, (-0.16, 0.04), (-0.14, -0.40), 0.108)                  # thumb
    sprite = Image.new("RGBA", size)
    _sleeve(sprite, -0.19, 0.15)
    sprite.alpha_composite(_shade(m, SKIN, SKIN_SHADE))
    d = ImageDraw.Draw(sprite)
    lw = max(2, round(0.014 * SPRITE_U))
    for y in (0.037, 0.172, 0.305):                                    # knuckle gaps
        a, b = _to_px(0.08, y), _to_px(0.34, y)
        d.line((*a, *b), fill=(*SKIN_LINE, 190), width=lw)
    a, b = _to_px(-0.05, 0.10), _to_px(-0.02, -0.06)                   # thumb fold
    d.line((*a, *b), fill=(*SKIN_LINE, 150), width=lw)
    nail = Image.new("RGBA", size)                                     # thumbnail
    x0, y0 = _to_px(-0.205, -0.47)
    x1, y1 = _to_px(-0.085, -0.33)
    ImageDraw.Draw(nail).rounded_rectangle((x0, y0, x1, y1), radius=0.055 * SPRITE_U,
                                           fill=(248, 220, 194, 255))
    sprite.alpha_composite(nail)
    return sprite


SPRITES = {}


def sprite(kind):
    if kind not in SPRITES:
        SPRITES[kind] = open_hand_sprite() if kind == "open" else thumbs_up_sprite()
    return SPRITES[kind]


def place_hand(layer, kind, x, y, scale=1.0, angle=0.0, alpha=1.0, shadow=True):
    """Composite the hand so its palm centre (local 0,0) lands on base (x, y)."""
    src = sprite(kind)
    px_per_unit = UNIT * S * scale
    factor = px_per_unit / SPRITE_U
    img = src.resize((max(1, round(src.width * factor)), max(1, round(src.height * factor))),
                     Image.Resampling.LANCZOS)
    ox, oy = _to_px(0, 0)
    ox, oy = ox * factor, oy * factor
    if angle:
        # Rotate about the palm centre: pad so that point is the image centre.
        cw, ch = img.size
        pw = round(2 * max(ox, cw - ox))
        ph = round(2 * max(oy, ch - oy))
        padded = Image.new("RGBA", (pw, ph))
        padded.paste(img, (round(pw / 2 - ox), round(ph / 2 - oy)))
        img = padded.rotate(angle, resample=Image.Resampling.BICUBIC)
        ox, oy = pw / 2, ph / 2
    if alpha < 1:
        img.putalpha(img.getchannel("A").point(lambda a: round(a * alpha)))
    pos = (round(p(x) - ox), round(p(y) - oy))
    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sh.putalpha(img.getchannel("A").point(lambda a: round(a * 0.45)))
        sh = sh.filter(ImageFilter.GaussianBlur(p(9)))
        layer.alpha_composite(sh, (pos[0] + p(6), pos[1] + p(10)))
    layer.alpha_composite(img, pos)


# ---------------------------------------------------------------- motion
ENTER, MOVE0, MOVE1, EXIT0, EXIT1, TOAST = 0.13, 0.16, 0.52, 0.62, 0.76, 0.76


def pose(scene, t):
    """Return (kind, x, y, scale, angle) for the hand at time t in [0, 1]."""
    m = ease((t - MOVE0) / (MOVE1 - MOVE0))
    kind, scale, angle = "open", 1.0, 0.0
    if scene == 0:
        x, y = 292 - 164 * m, 214
    elif scene == 1:
        x, y = 128 + 164 * m, 214
    elif scene == 2:
        x, y = CX, 262 - 104 * m
    elif scene == 3:
        x, y = CX, 158 + 104 * m
    elif scene == 4:
        x, y = CX, 212
        scale = 1.0 + 0.12 * ease_out((t - MOVE0) / 0.22)
        scale += 0.012 * math.sin(max(0.0, t - MOVE0) * 26)  # gentle "hold"
    else:
        kind, x, y = "thumb", CX + 6, 210
        scale = 0.92 + 0.10 * back_out((t - MOVE0) / 0.20)
    # Lean slightly into horizontal movement like a real wrist.
    if scene in (0, 1):
        dt = 0.01
        vx = (pose_x(scene, t + dt) - pose_x(scene, t - dt)) / (2 * dt)
        angle = max(-13.0, min(13.0, -vx * 0.028))
    elif scene in (2, 3):
        angle = 4.0 * math.sin(m * math.pi) * (1 if scene == 2 else -1)
    # Enter from below the frame, and drop out of view at the end.
    y += 230 * (1 - ease_out(t / ENTER))
    y += 260 * ease_in((t - EXIT0) / (EXIT1 - EXIT0))
    return kind, x, y, scale, angle


def pose_x(scene, t):
    m = ease((t - MOVE0) / (MOVE1 - MOVE0))
    return 292 - 164 * m if scene == 0 else 128 + 164 * m


# ---------------------------------------------------------------- stage drawing
def stage_mask():
    mask = Image.new("L", SIZE)
    ImageDraw.Draw(mask).rounded_rectangle(tuple(p(v) for v in STAGE), radius=p(14), fill=255)
    return mask


STAGE_MASK = None


def draw_stage_bg(img):
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle(tuple(p(v) for v in STAGE), radius=p(14), fill=(*PANEL, 255),
                        outline=(*LINE, 255), width=p(1))
    glow = Image.new("RGBA", SIZE)
    ImageDraw.Draw(glow).ellipse((p(CX - 120), p(CY - 105), p(CX + 120), p(CY + 105)),
                                 fill=(70, 78, 90, 70))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(p(40))))
    d = ImageDraw.Draw(img, "RGBA")
    # Viewfinder corner brackets.
    x0, y0, x1, y1 = STAGE[0] + 14, STAGE[1] + 14, STAGE[2] - 14, STAGE[3] - 14
    L, c = 16, (*GRAY, 150)
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line((p(x), p(y), p(x + sx * L), p(y)), fill=c, width=p(1.5))
        d.line((p(x), p(y), p(x), p(y + sy * L)), fill=c, width=p(1.5))
    d.ellipse((p(x0 + 8), p(y0 + 9), p(x0 + 14), p(y0 + 15)), fill=(235, 84, 84, 255))
    text(d, (x0 + 20, y0 + 7), "CAMERA", 8.5, GRAY, "demi")


def draw_guide(img, scene, t):
    """Dotted path showing where the hand travels, filled as it passes."""
    show = ease((t - 0.08) / 0.12) * (1 - ease((t - EXIT0) / 0.10))
    if show <= 0:
        return
    d = ImageDraw.Draw(img, "RGBA")
    progress = ease((t - MOVE0) / (MOVE1 - MOVE0))
    if scene in (0, 1, 2, 3):
        if scene == 0:
            a, b = (292, 300), (128, 300)
        elif scene == 1:
            a, b = (128, 300), (292, 300)
        elif scene == 2:
            a, b = (354, 262), (354, 132)
        else:
            a, b = (354, 132), (354, 262)
        count = 15
        for i in range(count):
            f = i / (count - 1)
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            done = f <= progress
            col = ACCENT if done else GRAY
            al = round((230 if done else 90) * show)
            r = 2.1 if done else 1.6
            d.ellipse((p(x - r), p(y - r), p(x + r), p(y + r)), fill=(*col, al))
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        ux, uy = math.cos(ang), math.sin(ang)
        tip = (b[0] + 9 * ux, b[1] + 9 * uy)
        pts = [tip, (tip[0] - 9 * ux - 5 * uy, tip[1] - 9 * uy + 5 * ux),
               (tip[0] - 9 * ux + 5 * uy, tip[1] - 9 * uy - 5 * ux)]
        col = ACCENT if progress > 0.98 else GRAY
        d.polygon([(p(x), p(y)) for x, y in pts], fill=(*col, round(220 * show)))
    else:
        # Hold ring around the hand fills while the pose is held.
        hold = ease((t - (MOVE0 + 0.08)) / (EXIT0 - MOVE0 - 0.12))
        r = 92
        box = (p(CX - r), p(CY - r - 4), p(CX + r), p(CY + r - 4))
        d.ellipse(box, outline=(*LINE, round(255 * show)), width=p(3))
        if hold > 0:
            d.arc(box, -90, -90 + 360 * hold, fill=(*ACCENT, round(255 * show)), width=p(3))


def draw_toast(img, action, t):
    k = ease_out((t - TOAST) / 0.08)
    if k <= 0:
        return
    d = ImageDraw.Draw(img, "RGBA")
    f = font(11, "bold")
    label = action
    tw = d.textlength(label, font=f) / S
    w = tw + 44
    x0 = CX - w / 2
    y0 = CY - 15 + 10 * (1 - k)
    al = round(255 * k)
    d.rounded_rectangle((p(x0), p(y0), p(x0 + w), p(y0 + 30)), radius=p(15),
                        fill=(*ACCENT, al))
    # check mark
    cx, cy = x0 + 17, y0 + 15
    d.line((p(cx - 5), p(cy), p(cx - 1.5), p(cy + 4), p(cx + 5.5), p(cy - 4.5)),
           fill=(*BG, al), width=p(2.2), joint="curve")
    text(d, (x0 + 30, y0 + 15), label, 11, BG, "bold", anchor="lm", alpha=al)


def stage_layer(scene, t):
    layer = Image.new("RGBA", SIZE)
    draw_guide(layer, scene, t)
    kind, x, y, scale, angle = pose(scene, t)
    # Faint motion trail during swipes.
    if scene in (0, 1, 2, 3) and MOVE0 < t < MOVE1 + 0.04:
        for i, a in ((4, 0.05), (3, 0.08), (2, 0.12), (1, 0.18)):
            _, gx, gy, gs, ga = pose(scene, t - 0.022 * i)
            place_hand(layer, kind, gx, gy, gs, ga, alpha=a, shadow=False)
    place_hand(layer, kind, x, y, scale, angle)
    draw_toast(layer, SCENES[scene][1], t)
    alpha = ImageChops.multiply(layer.getchannel("A"), STAGE_MASK)
    layer.putalpha(alpha)
    return layer



# ---------------------------------------------------------------- music player
TRACKS = [
    ("Midnight Drive", "Nova Lane", ((255, 120, 90), (120, 40, 160)), 204),
    ("Golden Hour", "Coral Avenue", ((255, 200, 90), (230, 90, 70)), 187),
]
VOL_LOW, VOL_HIGH = 0.45, 0.72
# Per scene: (track before, track after, playing before, playing after, vol before, vol after)
PLAYER_STATE = [
    (0, 1, True, True, VOL_LOW, VOL_LOW),
    (1, 0, True, True, VOL_LOW, VOL_LOW),
    (0, 0, True, True, VOL_LOW, VOL_HIGH),
    (0, 0, True, True, VOL_HIGH, VOL_LOW),
    (0, 0, True, False, VOL_LOW, VOL_LOW),
    (0, 0, False, True, VOL_LOW, VOL_LOW),
]
CARD = (416, 222, 688, 334)
ART_CACHE = {}


def album_art(index, size):
    key = (index, size)
    if key not in ART_CACHE:
        (c0, c1) = TRACKS[index][2]
        n = size
        grad = Image.new("RGBA", (n, n))
        gd = ImageDraw.Draw(grad)
        for y in range(n):
            f = y / max(1, n - 1)
            gd.line((0, y, n, y), fill=tuple(round(a + (b - a) * f) for a, b in zip(c0, c1)) + (255,))
        if index == 0:
            gd.ellipse((n * 0.28, n * 0.18, n * 0.72, n * 0.62), fill=(255, 236, 200, 230))
            gd.rectangle((0, n * 0.62, n, n), fill=(40, 20, 70, 200))
        else:
            for i in range(4):
                r = n * (0.18 + 0.12 * i)
                gd.ellipse((n / 2 - r, n * 0.95 - r, n / 2 + r, n * 0.95 + r),
                           outline=(255, 250, 230, 170 - 30 * i), width=max(1, n // 40))
        mask = Image.new("L", (n, n))
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, n - 1, n - 1), radius=n // 8, fill=255)
        grad.putalpha(mask)
        ART_CACHE[key] = grad
    return ART_CACHE[key]


def scene_start_position(scene):
    """Playback position (s) at each scene start, carried across the loop."""
    dur = N / FPS
    after_change = 1 + (1 - TOAST - 0.06) * dur          # restarted track, end of scene
    pos = [0.0] * 6
    pos[1] = after_change                                 # after "next"
    pos[2] = after_change                                 # after "previous"
    pos[3] = pos[2] + dur
    pos[4] = pos[3] + dur
    pos[5] = pos[4] + TOAST * dur                         # frozen while paused
    pos[0] = pos[5] + (1 - TOAST) * dur
    return pos[scene]


def _fmt(sec):
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def draw_player(img, scene, t, step):
    tr0, tr1, play0, play1, vol0, vol1 = PLAYER_STATE[scene]
    c = ease((t - TOAST) / 0.12)            # command takes effect with the toast
    x0, y0, x1, y1 = CARD
    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((p(x0), p(y0), p(x1), p(y1)), radius=p(12),
                        fill=(*PANEL, 255), outline=(*LINE, 255), width=p(1))
    playing = play1 if c >= 0.5 else play0

    # Header: "NOW PLAYING" with a live equaliser.
    text(d, (x0 + 14, y0 + 11), "NOW PLAYING", 8.5, GRAY, "demi")
    clock = (scene * N + step) / FPS
    for i in range(4):
        if playing:
            h = 3 + 6 * (0.5 + 0.5 * math.sin(clock * (7 + 2.3 * i) + i * 1.7))
        else:
            h = 2
        bx = x1 - 34 + i * 5
        d.rounded_rectangle((p(bx), p(y0 + 20 - h), p(bx + 3), p(y0 + 20)), radius=p(1),
                            fill=(*(ACCENT if playing else DIM), 255))

    # Album art and title slide over when the track changes.
    art_y = y0 + 30
    art_n = 46
    layer = Image.new("RGBA", img.size)
    ld = ImageDraw.Draw(layer, "RGBA")
    changing = tr0 != tr1
    direction = 1 if scene == 0 else -1
    out_k, in_k = ease(c * 2), ease(c * 2 - 1)   # old slides out, then new slides in
    items = [(tr0, 1 - out_k, -direction * 22 * out_k), (tr1, in_k, direction * 22 * (1 - in_k))] \
        if changing else [(tr0, 1.0, 0.0)]
    for idx, alpha, dx in items:
        if alpha <= 0.01:
            continue
        art = album_art(idx, p(art_n)).copy()
        art.putalpha(art.getchannel("A").point(lambda a: round(a * alpha)))
        layer.alpha_composite(art, (p(x0 + 14 + dx), p(art_y)))
        name, artist, _, _ = TRACKS[idx]
        text(ld, (x0 + 72 + dx, art_y + 8), name, 12, WHITE, "bold", alpha=round(255 * alpha))
        text(ld, (x0 + 72 + dx, art_y + 26), artist, 10, GRAY, "medium", alpha=round(255 * alpha))
    clip = Image.new("L", img.size)
    ImageDraw.Draw(clip).rectangle((p(x0 + 10), p(art_y - 2), p(x0 + 178), p(art_y + art_n + 2)), fill=255)
    layer.putalpha(ImageChops.multiply(layer.getchannel("A"), clip))
    img.alpha_composite(layer)
    d = ImageDraw.Draw(img, "RGBA")

    # Transport controls with a pulse on the button the gesture pressed.
    cy = art_y + 23
    px, mx, nx = x1 - 76, x1 - 48, x1 - 20
    pulse = math.sin(math.pi * clamp((t - TOAST) / 0.16))
    target = {0: nx, 1: px, 4: mx, 5: mx}.get(scene)
    if target is not None and pulse > 0:
        grow = clamp((t - TOAST) / 0.16)
        r = 10 + 8 * grow
        d.ellipse((p(target - 10), p(cy - 10), p(target + 10), p(cy + 10)),
                  fill=(*ACCENT, round(70 * pulse)))
        d.ellipse((p(target - r), p(cy - r), p(target + r), p(cy + r)),
                  outline=(*ACCENT, round(255 * pulse)), width=p(1.5))
    col = lambda which: (*WHITE, 255)
    # previous
    d.rectangle((p(px - 6), p(cy - 5), p(px - 4.6), p(cy + 5)), fill=col(px))
    d.polygon([(p(px + 5), p(cy - 5)), (p(px + 5), p(cy + 5)), (p(px - 4), p(cy))], fill=col(px))
    # next
    d.rectangle((p(nx + 4.6), p(cy - 5), p(nx + 6), p(cy + 5)), fill=col(nx))
    d.polygon([(p(nx - 5), p(cy - 5)), (p(nx - 5), p(cy + 5)), (p(nx + 4), p(cy))], fill=col(nx))
    # play / pause
    d.ellipse((p(mx - 11), p(cy - 11), p(mx + 11), p(cy + 11)), fill=(*WHITE, 255))
    if playing:
        for ox in (-3.6, 1.4):
            d.rounded_rectangle((p(mx + ox), p(cy - 4.5), p(mx + ox + 2.4), p(cy + 4.5)),
                                radius=p(0.8), fill=(*BG, 255))
    else:
        d.polygon([(p(mx - 3), p(cy - 5)), (p(mx - 3), p(cy + 5)), (p(mx + 5), p(cy))],
                  fill=(*BG, 255))

    # Progress keeps moving only while playing; a new track restarts it.
    by = y1 - 16
    cur = tr1 if (changing and c >= 0.5) else tr0
    length = TRACKS[cur][3]
    if changing and c >= 0.5:
        pos = 1 + (t - TOAST - 0.06) * (N / FPS)
    else:
        base = scene_start_position(scene)
        run_until = TOAST if (play0 and not play1) else (t if play0 else TOAST)
        moving = max(0.0, (min(t, run_until) if play0 else max(0.0, t - TOAST)) * (N / FPS))
        pos = base + moving
    pos = max(0.0, pos)
    bx0, bx1 = x0 + 14, x0 + 160
    d.rounded_rectangle((p(bx0), p(by - 1.5), p(bx1), p(by + 1.5)), radius=p(1.5), fill=(*LINE, 255))
    fx = bx0 + (bx1 - bx0) * min(1.0, pos / length)
    d.rounded_rectangle((p(bx0), p(by - 1.5), p(fx), p(by + 1.5)), radius=p(1.5), fill=(*WHITE, 255))
    d.ellipse((p(fx - 3), p(by - 3), p(fx + 3), p(by + 3)), fill=(*WHITE, 255))
    text(d, (bx0, by + 5), _fmt(pos), 7.5, GRAY, "demi")
    text(d, (bx1, by + 5), _fmt(length), 7.5, GRAY, "demi", anchor="rt")

    # Volume.
    vol = vol0 + (vol1 - vol0) * c
    vx = x1 - 96
    vcol = ACCENT if (scene in (2, 3) and 0 < c < 1) or (scene in (2, 3) and pulse > 0.2) else GRAY
    d.rectangle((p(vx), p(by - 2.5), p(vx + 3), p(by + 2.5)), fill=(*vcol, 255))
    d.polygon([(p(vx + 3), p(by - 2.5)), (p(vx + 8), p(by - 6)), (p(vx + 8), p(by + 6)),
               (p(vx + 3), p(by + 2.5))], fill=(*vcol, 255))
    for k, r in enumerate((5, 8)):
        if vol > 0.3 + 0.3 * k:
            d.arc((p(vx + 8 - r), p(by - r), p(vx + 8 + r), p(by + r)), -45, 45,
                  fill=(*vcol, 255), width=p(1.1))
    vb0, vb1 = vx + 22, x1 - 14
    d.rounded_rectangle((p(vb0), p(by - 1.5), p(vb1), p(by + 1.5)), radius=p(1.5), fill=(*LINE, 255))
    vfx = vb0 + (vb1 - vb0) * vol
    d.rounded_rectangle((p(vb0), p(by - 1.5), p(vfx), p(by + 1.5)), radius=p(1.5),
                        fill=(*(ACCENT if scene in (2, 3) else WHITE), 255))
    text(d, (vb1, by + 5), f"{round(vol * 100)}%", 7.5, GRAY, "demi", anchor="rt")

# ---------------------------------------------------------------- frame
def draw_frame(scene, step):
    title, action, instruction = SCENES[scene]
    t = step / (N - 1)
    img = Image.new("RGBA", SIZE, (*BG, 255))
    d = ImageDraw.Draw(img, "RGBA")

    # Header with six-segment progress.
    text(d, (32, 24), "GESTURE CONTROL", 13, WHITE, "bold")
    seg_w, gap, sx = 34, 6, 688 - 6 * 34 - 5 * 6
    for i in range(6):
        x = sx + i * (seg_w + gap)
        d.rounded_rectangle((p(x), p(29), p(x + seg_w), p(32)), radius=p(1.5), fill=(*LINE, 255))
        fill = 1.0 if i < scene else (t if i == scene else 0.0)
        if fill > 0:
            d.rounded_rectangle((p(x), p(29), p(x + seg_w * fill), p(32)), radius=p(1.5),
                                fill=(*(WHITE if i < scene else ACCENT), 255))
    d.line((p(32), p(54), p(688), p(54)), fill=(*LINE, 255), width=p(1))

    draw_stage_bg(img)
    img.alpha_composite(stage_layer(scene, t))
    d = ImageDraw.Draw(img, "RGBA")

    # Right-hand copy.
    text(d, (418, 80), f"{scene + 1:02d} / 06", 11, GRAY, "demi")
    text(d, (416, 98), title, 28, WHITE, "heavy")
    d.line((p(418), p(142), p(686), p(142)), fill=(*LINE, 255), width=p(1))
    d.ellipse((p(419), p(158), p(427), p(166)), fill=(*ACCENT, 255))
    text(d, (436, 162), action, 17, WHITE, "bold", anchor="lm")
    words, lines, cur = instruction.split(), [], ""
    f = font(11, "medium")
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) / S > 262:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    for i, ln in enumerate(lines):
        text(d, (418, 180 + i * 16), ln, 11, GRAY, "medium")
    draw_player(img, scene, t, step)

    # Footer steps.
    d.line((p(32), p(344), p(688), p(344)), fill=(*LINE, 255), width=p(1))
    steps = ["SHOW GESTURE", "MOVE HAND OUT", "COMMAND SENT"]
    active = 0 if t < EXIT0 else 1 if t < TOAST else 2
    for i, words in enumerate(steps):
        x = 33 + i * 224
        on = i == active
        done = i < active
        col = ACCENT if on else (GRAY if done else DIM)
        d.ellipse((p(x), p(356), p(x + 16), p(372)),
                  fill=(*ACCENT, 255) if on else None, outline=(*col, 255), width=p(1.2))
        text(d, (x + 8, 364), str(i + 1), 9, BG if on else col, "bold", anchor="mm")
        text(d, (x + 24, 364), words, 10, WHITE if on else col, "demi", anchor="lm")
        if i < 2:
            d.line((p(x + 190), p(364), p(x + 204), p(364)), fill=(*DIM, 255), width=p(1))

    # Short crossfade at the scene boundaries.
    fade = min(1.0, (step + 1) / 3, (N - step) / 3)
    if fade < 1:
        img = Image.blend(Image.new("RGBA", SIZE, (*BG, 255)), img, fade)
    return img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)


def render():
    global STAGE_MASK
    STAGE_MASK = stage_mask()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    total = len(SCENES) * N
    with tempfile.TemporaryDirectory(prefix="gesture-tutorial-") as temp:
        directory = Path(temp)
        for scene in range(len(SCENES)):
            for step in range(N):
                frame = draw_frame(scene, step)
                frame.save(directory / f"frame_{scene * N + step:04d}.png")
                if step == round(N * 0.45):
                    frame.save(PREVIEW / f"gesture_tutorial_faceless_{scene + 1:02d}.png")
        pattern = str(directory / "frame_%04d.png")
        palette = str(directory / "palette.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", pattern,
                        "-vf", "palettegen=max_colors=192:stats_mode=full", palette], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", pattern,
                        "-i", palette, "-lavfi", "paletteuse=dither=sierra2_4a",
                        "-loop", "0", str(OUTPUT)], check=True)
    print(f"Created {OUTPUT} ({total} frames, {total / FPS:.1f} seconds)")


if __name__ == "__main__":
    render()
