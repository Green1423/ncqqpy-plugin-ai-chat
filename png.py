"""从渲染结果里**按像素**紧裁留白 —— 纯标准库（zlib + struct），不引图像库。

为什么不用页面报的尺寸：页面只能给「元素的盒子」，给不了「墨迹在哪」。
实测页面上报的尺寸算出来的图，右边多 42px、下边多 30px 的空白
（`x^{0}` 的内容只有 22×21，图却有 68×55）。而按内容真实边界裁剪
就必须看像素 —— 顺带也不怕「web 字体晚到导致量窄了把公式切掉」。

Chrome 出的 PNG 是 8bit、非交错、filter 0~4，所以解码只要实现那 5 种
filter 反推；编码用 filter 0（不过滤）最省事，体积对这种小图无所谓。
"""
from __future__ import annotations

import struct
import zlib

#: 判定「有墨迹」的亮度阈值。用 250 而不是更低：要连**抗锯齿边缘**一起收进来，
#: 否则紧裁会把字形的浅色边切掉一圈（看起来像被削过）。
INK_THRESHOLD = 250


class PngError(RuntimeError):
    """不是我们能处理的 PNG（或结构不认识）。"""


def read_png(path: str) -> tuple[int, int, int, list[bytes]]:
    """读 PNG，返回 `(宽, 高, 通道数, 每行的原始像素)`。

    只支持 Chrome 截图的形态：8bit、非交错、颜色类型 0/2/4/6。
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise PngError("不是 PNG")
    pos, idat = 8, bytearray()
    width = height = 0
    channels = 3
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
            interlace = body[12]
            if depth != 8 or interlace != 0:
                raise PngError(f"不支持的 PNG（depth={depth} interlace={interlace}）")
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color, 0)
            if not channels or color == 3:
                raise PngError(f"不支持的颜色类型 {color}")
        elif ctype == b"IDAT":
            idat += body
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if not width or not height:
        raise PngError("没有 IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    rows: list[bytes] = []
    prev = bytearray(stride)
    p = 0
    for _ in range(height):
        ftype = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if ftype == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif ftype != 0:
            raise PngError(f"未知 filter {ftype}")
        rows.append(bytes(line))
        prev = line
    return width, height, channels, rows


def ink_bounds(rows: list[bytes], width: int, channels: int, *,
               threshold: int = INK_THRESHOLD) -> tuple[int, int, int, int] | None:
    """整行/整列有没有墨迹 → 内容外框 `(x0, y0, x1, y1)`（含端点）。

    按**整行整列**判定而不是逐像素取极值：逐像素极值会被一个孤立噪点带偏，
    整行整列只要该行/列有任何墨迹就保留，对「不能切到内容」这个目标更稳。
    """
    rows_hit: list[int] = []
    col_flag = bytearray(width)
    for y, row in enumerate(rows):
        hit = False
        for x in range(width):
            o = x * channels
            if min(row[o], row[o + 1], row[o + 2]) < threshold:
                hit = True
                col_flag[x] = 1
        if hit:
            rows_hit.append(y)
    if not rows_hit:
        return None
    cols_hit = [x for x in range(width) if col_flag[x]]
    if not cols_hit:
        return None
    return cols_hit[0], rows_hit[0], cols_hit[-1], rows_hit[-1]


def _crop_rows(rows: list[bytes], channels: int, x0: int, x1: int) -> list[bytes]:
    out = []
    for row in rows:
        out.append(row[x0 * channels:(x1 + 1) * channels])
    return out


def write_png(path: str, width: int, height: int, channels: int,
              rows: list[bytes]) -> None:
    """写 8bit 非交错 PNG（filter 0）。"""
    color = {1: 0, 2: 4, 3: 2, 4: 6}[channels]
    raw = bytearray()
    for row in rows:
        raw.append(0)
        raw += row
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (struct.pack(">I", len(payload)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))


def crop_to_ink(path: str, *, pad_x: int = 6, pad_y: int = 4,
                threshold: int = INK_THRESHOLD) -> bool:
    """把图片裁到内容边界（四周留 `pad_x`/`pad_y`，不放大）。

    返回是否真的裁过（全白图或裁不动时返回 `False`，调用方保留原图）。
    就地覆写 `path`。
    """
    try:
        width, height, channels, rows = read_png(path)
    except (PngError, OSError, zlib.error):
        return False
    bounds = ink_bounds(rows, width, channels, threshold=threshold)
    if bounds is None:
        return False
    x0, y0, x1, y1 = bounds
    nx0 = max(0, x0 - pad_x)
    ny0 = max(0, y0 - pad_y)
    nx1 = min(width - 1, x1 + pad_x)
    ny1 = min(height - 1, y1 + pad_y)
    new_w = nx1 - nx0 + 1
    new_h = ny1 - ny0 + 1
    if new_w == width and new_h == height:
        return False
    cropped = _crop_rows(rows[ny0:ny1 + 1], channels, nx0, nx1)
    try:
        write_png(path, new_w, new_h, channels, cropped)
    except OSError:
        return False
    return True


#: 整页图裁剪时四周保留的像素。
#:
#: 留白主要靠 CSS 盒子（`FRAGMENT_PADDING` / 行内公式页的 padding），这里只是
#: 「按像素收口」时的一点点余量 —— 目标是**紧贴**，所以给个位数。
#: ⚠️ 不能给 0：KaTeX 的字形墨迹会比元素盒子**稍微偏左**，0 会让字形压在边界上
#: （实测：10px 时右边缘就已经贴到最后两像素）。
PAGE_PAD = 4


def crop_right_to_ink(path: str, *, pad_x: int = PAGE_PAD, min_width: int = 200,
                      threshold: int = INK_THRESHOLD) -> bool:
    """裁掉**右侧**与底部留白（左侧不动，保留排版起点）。

    给「可能被内容撑宽、也可能留了一大片右空白」的图用：缩放后的超宽公式就属于
    这种 —— 页面报的宽度与内容实际宽度经常对不上（量测受容器定宽与 em 缩放影响），
    与其继续跟布局较劲，不如直接看像素。
    """
    try:
        width, height, channels, rows = read_png(path)
    except (PngError, OSError, zlib.error):
        return False
    bounds = ink_bounds(rows, width, channels, threshold=threshold)
    if bounds is None:
        return False
    x0, _y0, x1, y1 = bounds
    # ⚠️ 切点先夹到图片范围内：`x1 + pad` 可能超过 `width - 1`（内容贴边时），
    # 那样 `_crop_rows` 切出来的行会**比声明的宽度短**，写出的 PNG 行长度与
    # IHDR 不一致 —— 解码时 filter 直接读到垃圾字节（实测报「未知 filter 255」）。
    cut = min(width - 1, x1 + pad_x)
    new_w = max(min_width, cut + 1)
    new_h = min(height, y1 + pad_x + 1)
    if new_w >= width and new_h >= height:
        return False
    # 行数据只有 `cut + 1` 宽；声明宽度更大时右侧由 PNG 编码补 0 不可靠，
    # 所以声明宽度就用实际切出来的宽度，保持一致。
    new_w = cut + 1
    cropped = _crop_rows(rows[:new_h], channels, x0=0, x1=cut)
    try:
        write_png(path, new_w, new_h, channels, cropped)
    except OSError:
        return False
    return True


def crop_bottom_to_ink(path: str, *, pad_y: int = PAGE_PAD,
                       threshold: int = INK_THRESHOLD) -> bool:
    """只裁掉**底部**留白（宽度保持不变）。

    给「整页宽」的图用（整段公式、表格）：它们横向本来就该铺满，
    裁宽度没意义；但底部那段预留下的空白（量高度时留的安全余量）必须去掉，
    否则每张图下面都挂着一块空白。保留宽度 = 不做横向裁剪。
    """
    try:
        width, height, channels, rows = read_png(path)
    except (PngError, OSError, zlib.error):
        return False
    bounds = ink_bounds(rows, width, channels, threshold=threshold)
    if bounds is None:
        return False
    _x0, _y0, _x1, y1 = bounds
    new_h = min(height, y1 + pad_y + 1)
    if new_h >= height:
        return False
    try:
        write_png(path, width, new_h, channels, rows[:new_h])
    except OSError:
        return False
    return True
