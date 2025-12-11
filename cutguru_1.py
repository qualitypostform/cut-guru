# === CHUNK 1/7: BEGIN ===
# Cut Guru - Full MaxRects System with True 2D Kerf + SVG Kerf Rendering
# ---------------------------------------------------------------
# CHUNK 1/7: Imports, Data Classes, Kerf Helpers
# ---------------------------------------------------------------

from datetime import datetime
import pytz
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from flask import Flask, request, render_template_string, url_for, redirect, session
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
from functools import wraps
import os
from functools import wraps
from datetime import datetime


# ============================================================
#  DATA CLASSES
# ============================================================

@dataclass
class Part:
    """
    A single cut part before expansion.
    width  = CUT width  (X direction)
    height = CUT length (Y direction)
    """
    name: str
    width: float
    height: float
    quantity: int = 1

    final_width: float = None
    final_length: float = None

    band_width_sides: int = 0
    band_length_sides: int = 0

    can_rotate: bool = True
    grain_group: Optional[str] = None
    grain_order: int = 0

    # NEW
    has_hinge_holes: bool = False


@dataclass
class PlacedPart:
    """
    A placed rectangle inside a board.
    Coordinates stored are CUT coordinates (no kerf embedded).
    """
    name: str
    x: float
    y: float
    width: float
    height: float
    rotated: bool
    board_index: int

    final_width: float
    final_length: float
    band_width_sides: int
    band_length_sides: int
    can_rotate: bool
    grain_group: Optional[str]
    grain_order: int

    # NEW
    has_hinge_holes: bool = False



@dataclass
class FreeRect:
    """
    Represents a FREE space region inside a board (kerf-free internal geometry).
    """
    x: float
    y: float
    width: float
    height: float


@dataclass
class BoardLayout:
    """
    Board layout storing placed parts and remaining free rectangles.
    """
    width: float      # full board width
    height: float     # full board height
    free_rects: List[FreeRect] = field(default_factory=list)
    parts: List[PlacedPart] = field(default_factory=list)


@dataclass
class BoardStock:
    """
    A stock board type the user has in inventory.
    length = usable board length (mm)
    width  = usable board width  (mm)
    quantity = how many boards of this size are available
    """
    length: float
    width: float
    quantity: int = 1


# ============================================================
#  KERF GEOMETRY HELPERS
# ============================================================

def kerf_expand(w: float, h: float, kerf: float) -> Tuple[float, float]:
    """
    Expands a part's size by kerf in both directions so MaxRects ensures spacing.

    For 2D kerf model:
        final_w = w + kerf
        final_h = h + kerf

    This ensures each placed part is isolated by kerf on all sides.
    """
    return w + kerf, h + kerf


def kerf_board_dims(board_w: float, board_h: float, kerf: float) -> Tuple[float, float]:
    """
    Compute the usable interior area for placement considering kerf around edges.

    Full 2D kerf model:
        Usable width  = board_w - kerf*2
        Usable height = board_h - kerf*2
    """
    return board_w - kerf * 2, board_h - kerf * 2
# === CHUNK 1/7: END ===

# === CHUNK 2/7: BEGIN ===

# ============================================================
#  MAXRECTS CORE (BSSF — Best Short Side Fit)
# ============================================================

class MaxRects:
    """
    MaxRects with:
    - Best Short Side Fit
    - Supports rotated or unrotated placement
    - Free-rectangle pruning (removes contained rects)
    - Guaranteed non-overlap (proper split geometry)
    """

    def __init__(self, width: float, height: float):
        self.bin_width = width
        self.bin_height = height
        self.free_rects: List[FreeRect] = [
            FreeRect(0, 0, width, height)
        ]

    # ------------------------------------------------------------
    # Split one free rectangle around a placed block
    # ------------------------------------------------------------
    def split_free_rect(self, fr: FreeRect, x: float, y: float,
                        w: float, h: float):
        """
        Split rectangle 'fr' around placed block at (x,y,w,h).
        Produces up to 4 new rectangles.

        All new rectangles are guaranteed non-overlapping.
        """
        new_rects = []

        fr_right = fr.x + fr.width
        fr_bottom = fr.y + fr.height
        block_right = x + w
        block_bottom = y + h

        # Top segment
        if y > fr.y:
            new_rects.append(
                FreeRect(fr.x, fr.y, fr.width, y - fr.y)
            )

        # Bottom segment
        if block_bottom < fr_bottom:
            new_rects.append(
                FreeRect(fr.x, block_bottom,
                         fr.width, fr_bottom - block_bottom)
            )

        # Overlap zone vertically
        overlap_top = max(fr.y, y)
        overlap_bottom = min(fr_bottom, block_bottom)

        if overlap_bottom > overlap_top:
            # Left strip
            if x > fr.x:
                new_rects.append(
                    FreeRect(fr.x, overlap_top,
                             x - fr.x,
                             overlap_bottom - overlap_top)
                )

            # Right strip
            if block_right < fr_right:
                new_rects.append(
                    FreeRect(block_right, overlap_top,
                             fr_right - block_right,
                             overlap_bottom - overlap_top)
                )

        return new_rects

    # ------------------------------------------------------------
    # Remove rectangles contained in others
    # ------------------------------------------------------------
    def prune(self):
        final = []
        for i, a in enumerate(self.free_rects):
            contained = False
            for j, b in enumerate(self.free_rects):
                if i != j:
                    if (a.x >= b.x and a.y >= b.y and
                        a.x + a.width <= b.x + b.width and
                        a.y + a.height <= b.y + b.height):
                        contained = True
                        break
            if not contained:
                final.append(a)
        self.free_rects = final

    # ------------------------------------------------------------
    # Merge free rectangles that are exactly adjacent
    # (share a full edge). This dramatically improves packing
    # efficiency by restoring large usable areas.
    # ------------------------------------------------------------
    def merge_adjacent(self):
        rects = list(self.free_rects)
        merged_any = True

        while merged_any:
            merged_any = False
            new_rects = []
            used = [False] * len(rects)

            for i in range(len(rects)):
                if used[i]:
                    continue

                a = rects[i]
                ax, ay, aw, ah = a.x, a.y, a.width, a.height
                current = a

                for j in range(i + 1, len(rects)):
                    if used[j]:
                        continue

                    b = rects[j]
                    bx, by, bw, bh = b.x, b.y, b.width, b.height

                    merged = None

                    # ---- Horizontal merge (side-by-side) ----
                    if ay == by and ah == bh:
                        # a next to b
                        if ax + aw == bx:
                            merged = FreeRect(ax, ay, aw + bw, ah)
                        # b next to a
                        elif bx + bw == ax:
                            merged = FreeRect(bx, by, aw + bw, ah)

                    # ---- Vertical merge (stacked) ----
                    if merged is None and ax == bx and aw == bw:
                        # b below a
                        if ay + ah == by:
                            merged = FreeRect(ax, ay, aw, ah + bh)
                        # a below b
                        elif by + bh == ay:
                            merged = FreeRect(ax, by, aw, ah + bh)

                    # Apply merge
                    if merged is not None:
                        current = merged
                        ax, ay, aw, ah = current.x, current.y, current.width, current.height
                        used[j] = True
                        merged_any = True

                used[i] = True
                new_rects.append(current)

            rects = new_rects

        self.free_rects = rects


    # ------------------------------------------------------------
    # Insert rectangle (w,h) with optional rotation
    # ------------------------------------------------------------
    def insert(self, w: float, h: float, allow_rotate: bool) -> Optional[Tuple[float,float,bool]]:
        """
        Returns (x, y, rotated) or None.
        """
        best_score = None
        best_pos = None
        best_rot = False
        best_index = -1

        for try_rot in (False, True):
            if try_rot and not allow_rotate:
                continue

            rw = h if try_rot else w
            rh = w if try_rot else h

            for idx, fr in enumerate(self.free_rects):
                if rw <= fr.width and rh <= fr.height:
                    leftover_w = fr.width - rw
                    leftover_h = fr.height - rh
                    short_side = min(leftover_w, leftover_h)

                    if best_score is None or short_side < best_score:
                        best_score = short_side
                        best_pos = (fr.x, fr.y)
                        best_index = idx
                        best_rot = try_rot

        if best_pos is None:
            return None

        # Split the free rect
        fr = self.free_rects.pop(best_index)
        x, y = best_pos

        new_rects = self.split_free_rect(
            fr, x, y,
            h if best_rot else w,
            w if best_rot else h
        )

        self.free_rects.extend(new_rects)
        self.prune()
        self.merge_adjacent()  # <--- NEW: merge side-by-side and stacked free spaces


        return (x, y, best_rot)


# ============================================================
#  GRAIN GROUP COMPOSITES
# ============================================================

@dataclass
class GrainComposite:
    width: float
    height: float
    subparts: List[Part]
    grain_group: str


def make_grain_composite(parts: List[Part]) -> GrainComposite:
    """
    Grain groups become tall vertical composites that cannot rotate.
    """
    parts_sorted = sorted(parts, key=lambda p: p.grain_order)
    total_h = sum(p.height for p in parts_sorted)
    max_w = max(p.width for p in parts_sorted)
    gname = parts_sorted[0].grain_group

    return GrainComposite(
        width=max_w,
        height=total_h,
        subparts=parts_sorted,
        grain_group=gname
    )


# ============================================================
#  PART EXPANSION & PLACEMENT HELPERS
# ============================================================

def expand_parts(parts):
    out = []
    for p in parts:
        for _ in range(p.quantity):
            out.append(
                Part(
                    name=p.name,
                    width=p.width,
                    height=p.height,
                    quantity=1,
                    final_width=p.final_width,
                    final_length=p.final_length,
                    band_width_sides=p.band_width_sides,
                    band_length_sides=p.band_length_sides,
                    can_rotate=p.can_rotate,
                    grain_group=p.grain_group,
                    grain_order=p.grain_order,
                    has_hinge_holes=p.has_hinge_holes,  # NEW
                )
            )
    return out



def place_composite(maxr: MaxRects, comp: GrainComposite,
                    board_index: int, kerf: float,
                    layout: BoardLayout) -> bool:

    w_exp, h_exp = kerf_expand(comp.width, comp.height, kerf)
    pos = maxr.insert(w_exp, h_exp, allow_rotate=False)
    if pos is None:
        return False

    x, y, _ = pos

    curr_y = y
    for child in comp.subparts:
        cx = x + (comp.width - child.width)/2
        cy = curr_y
        layout.parts.append(
            PlacedPart(
                name=child.name,
                x=cx,
                y=cy,
                width=child.width,
                height=child.height,
                rotated=False,
                board_index=board_index,
                final_width=child.final_width,
                final_length=child.final_length,
                band_width_sides=child.band_width_sides,
                band_length_sides=child.band_length_sides,
                can_rotate=child.can_rotate,
                grain_group=child.grain_group,
                grain_order=child.grain_order,
                # 🔴 ADD THIS LINE
                has_hinge_holes=child.has_hinge_holes,
            )
        )
        curr_y += child.height + kerf  # spacing from composite expansion

    return True



def place_normal_part(maxr: MaxRects, part: Part,
                      board_index: int, kerf: float,
                      layout: BoardLayout) -> bool:

    w_exp, h_exp = kerf_expand(part.width, part.height, kerf)
    pos = maxr.insert(w_exp, h_exp, allow_rotate=part.can_rotate)
    if pos is None:
        return False

    x, y, rotated = pos

    if rotated:
        w = part.height
        h = part.width
    else:
        w = part.width
        h = part.height

    layout.parts.append(
        PlacedPart(
            name=part.name,
            x=x, y=y,
            width=w, height=h,
            rotated=rotated,
            board_index=board_index,
            final_width=part.final_width,
            final_length=part.final_length,
            band_width_sides=part.band_width_sides,
            band_length_sides=part.band_length_sides,
            can_rotate=part.can_rotate,
            grain_group=part.grain_group,
            grain_order=part.grain_order,
            has_hinge_holes=part.has_hinge_holes,  # NEW
        )
    )

    return True


# === CHUNK 2/7: END ===

# === CHUNK 3/7: BEGIN ===

# ============================================================
#  MAXRECTS-DRIVEN NESTING
# ============================================================

# ============================================================
#  MAXRECTS-DRIVEN NESTING
# ============================================================

# ============================================================
#  MAXRECTS-DRIVEN NESTING (single run)
# ============================================================

def nest_with_maxrects(board_w: float,
                       board_h: float,
                       parts: List[Part],
                       kerf: float,
                       shuffle_mode: int = 0):
    """
    Single MaxRects run on an infinite supply of identical boards.

    shuffle_mode controls how we vary the item order:
        0 = deterministic, sort by area descending
        1 = random shuffle
        2 = random shuffle, then sort by height descending
    """

    # Expand quantities
    expanded = expand_parts(parts)

    # Optional randomisation of base list (before grouping)
    if shuffle_mode in (1, 2):
        import random
        random.shuffle(expanded)

    # Separate grain groups
    grain_map = {}
    normals = []

    for p in expanded:
        if p.grain_group:
            grain_map.setdefault(p.grain_group, []).append(p)
        else:
            normals.append(p)

    composites = [make_grain_composite(v) for v in grain_map.values()]

    # Unified list of items (composites first for better packing)
    items = composites + normals

    if shuffle_mode == 0:
        # original behaviour: area descending
        items.sort(key=lambda it: (it.width * it.height), reverse=True)
    elif shuffle_mode == 1:
        # already shuffled above, no extra sort
        pass
    elif shuffle_mode == 2:
        # bias towards taller strips first
        items.sort(key=lambda it: it.height, reverse=True)

    # Compute interior usable space (kerf on outer edges)
    usable_w, usable_h = kerf_board_dims(board_w, board_h, kerf)

    boards: List[BoardLayout] = []
    total_part_area = sum(p.width * p.height for p in expanded)

    for item in items:
        placed = False

        # Try existing boards first
        for bi, board in enumerate(boards):
            maxr = board._maxr
            if isinstance(item, GrainComposite):
                if place_composite(maxr, item, bi, kerf, board):
                    placed = True
                    break
            else:
                if place_normal_part(maxr, item, bi, kerf, board):
                    placed = True
                    break

        # Need new board?
        if not placed:
            b = BoardLayout(board_w, board_h)
            b._maxr = MaxRects(usable_w, usable_h)
            boards.append(b)
            bi = len(boards) - 1

            if isinstance(item, GrainComposite):
                if not place_composite(b._maxr, item, bi, kerf, b):
                    raise ValueError(f"Grain group '{item.grain_group}' too large.")
            else:
                if not place_normal_part(b._maxr, item, bi, kerf, b):
                    raise ValueError(f"Part '{item.name}' too large for board.")

    # Copy free rects
    for b in boards:
        b.free_rects = b._maxr.free_rects

    waste = len(boards) * board_w * board_h - total_part_area
    return boards, total_part_area, waste


# ============================================================
#  MULTI-TRY GLOBAL OPTIMISER (both orientations)
# ============================================================

def best_layout_global(board_length: float,
                       board_width: float,
                       parts: List[Part],
                       kerf: float,
                       tries_per_orientation: int = 16):
    """
    Try multiple randomised layouts on boards with the
    given length/width (no board rotation), and keep
    the best result (fewest boards, then least waste).

    Returns (boards, used_area, waste_area).
    """
    import math

    best_boards = None
    best_used = 0.0
    best_waste = math.inf
    best_metric = None  # (board_count, waste)

    # FIXED orientation:
    #   board_width  = across the board  (2100)
    #   board_length = along the board   (2850)
    bw, bl = board_width, board_length

    for t in range(tries_per_orientation):
        mode = t % 3  # 0,1,2 – same shuffle logic as before

        boards, used_area, waste_area = nest_with_maxrects(
            bw,
            bl,
            parts,
            kerf,
            shuffle_mode=mode,
        )

        # squeeze onto earlier boards if possible
        boards = second_pass_optimiser(boards, kerf)

        metric = (len(boards), waste_area)
        if best_metric is None or metric < best_metric:
            best_metric = metric
            best_boards = boards
            best_used = used_area
            best_waste = waste_area

    return best_boards, best_used, best_waste



# ============================================================
#  NESTING WITH BOARD INVENTORY
# ============================================================

def nest_with_inventory(board_stock: List[BoardStock],
                        parts: List[Part],
                        kerf: float):
    """
    Nest parts onto a finite inventory of boards of possibly different sizes.

    board_stock: list of BoardStock(length,width,quantity)
    Returns (boards, used_area, waste_area).

    If there is not enough board area (or geometry) to place all parts,
    it raises a ValueError.
    """

    # Expand quantities
    expanded = expand_parts(parts)

    # Separate grain groups
    grain_map = {}
    normals = []
    for p in expanded:
        if p.grain_group:
            grain_map.setdefault(p.grain_group, []).append(p)
        else:
            normals.append(p)

    composites = [make_grain_composite(v) for v in grain_map.values()]

    # Unified list of items (composites first for better packing)
    items = composites + normals
    items.sort(key=lambda it: (it.width * it.height), reverse=True)

    # Build concrete boards from stock
    boards: List[BoardLayout] = []
    for bs in board_stock:
        for _ in range(bs.quantity):
            b = BoardLayout(width=bs.width, height=bs.length)
            usable_w, usable_h = kerf_board_dims(bs.width, bs.length, kerf)
            b._maxr = MaxRects(usable_w, usable_h)
            boards.append(b)

    if not boards:
        raise ValueError("No boards in inventory.")

    total_part_area = sum(p.width * p.height for p in expanded)

    # Place items
    for item in items:
        placed = False

        for bi, board in enumerate(boards):
            maxr = board._maxr
            if isinstance(item, GrainComposite):
                if place_composite(maxr, item, bi, kerf, board):
                    placed = True
                    break
            else:
                if place_normal_part(maxr, item, bi, kerf, board):
                    placed = True
                    break

        if not placed:
            name = getattr(item, "grain_group", None) or getattr(item, "name", "unnamed")
            raise ValueError(f"Not enough boards in stock to place '{name}'")

    # Copy free rects
    for b in boards:
        b.free_rects = b._maxr.free_rects

    total_board_area = sum(b.width * b.height for b in boards)
    waste = total_board_area - total_part_area
    return boards, total_part_area, waste


# ============================================================
#  SECOND PASS OPTIMISER
# ============================================================

def second_pass_optimiser(boards: List[BoardLayout], kerf: float):
    if len(boards) <= 1:
        return boards

    improved = True
    while improved:
        improved = False
        last_idx = len(boards) - 1
        last = boards[last_idx]

        movable = [p for p in list(last.parts) if not p.grain_group]

        for part in movable:

            # Create a temporary part object for MaxRects test
            test_part = Part(
                name=part.name,
                width=part.width,
                height=part.height,
                can_rotate=part.can_rotate,
                grain_group=None,
                grain_order=0,
                final_width=part.final_width,
                final_length=part.final_length,
                band_width_sides=part.band_width_sides,
                band_length_sides=part.band_length_sides
            )

            for bi in range(last_idx):
                bd = boards[bi]
                pos = bd._maxr.insert(
                    *(kerf_expand(test_part.width, test_part.height, kerf)),
                    allow_rotate=test_part.can_rotate
                )
                if pos is None:
                    continue

                x, y, rot = pos
                w = test_part.height if rot else test_part.width
                h = test_part.width if rot else test_part.height

                # Place on earlier board
                bd.parts.append(
                    PlacedPart(
                        name=part.name,
                        x=x, y=y,
                        width=w, height=h,
                        rotated=rot,
                        board_index=bi,
                        final_width=part.final_width,
                        final_length=part.final_length,
                        band_width_sides=part.band_width_sides,
                        band_length_sides=part.band_length_sides,
                        can_rotate=part.can_rotate,
                        grain_group=None,
                        grain_order=0
                    )
                )

                # Remove from final board
                last.parts.remove(part)
                improved = True
                break

            if improved:
                if not last.parts:
                    boards.pop()
                break

    return boards


# ============================================================
#  BOARD ORIENTATION COMPARISON (A vs B) - UNUSED NOW
# ============================================================

def run_orientation_test(board_length, board_width, parts, kerf):
    # Orientation A (as-provided)
    A_boards, A_used, A_waste = nest_with_maxrects(board_width, board_length, parts, kerf)
    A_boards = second_pass_optimiser(A_boards, kerf)

    # Orientation B (rotated)
    B_boards, B_used, B_waste = nest_with_maxrects(board_length, board_width, parts, kerf)
    B_boards = second_pass_optimiser(B_boards, kerf)

    metricA = (len(A_boards), A_waste)
    metricB = (len(B_boards), B_waste)

    if metricB < metricA:
        return B_boards, B_used, B_waste, board_length, board_width
    else:
        return A_boards, A_used, A_waste, board_width, board_length

# === CHUNK 3/7: END ===


# === CHUNK 4/7: BEGIN ===

# ============================================================
#  BOOLEAN & DIMENSION PARSER
# ============================================================

def parse_bool(v: str) -> bool:
    if not v:
        return True
    v = v.strip().lower()
    return v in ("y","yes","true","1","on")

def parse_dimension_mm(val: str) -> float:
    """
    Supports:
        - 600
        - 600mm
        - 23 1/2"
        - 23⅝"
        - 1/16"
        - 2-3/8"
    If ends with ", treat as inches.
    """

    # Convert unicode fractions into " n/d "
    def normalize_unicode(s: str) -> str:
        mapping = {
            "¼":"1/4","½":"1/2","¾":"3/4",
            "⅐":"1/7","⅑":"1/9","⅒":"1/10",
            "⅓":"1/3","⅔":"2/3",
            "⅕":"1/5","⅖":"2/5","⅗":"3/5","⅘":"4/5",
            "⅙":"1/6","⅚":"5/6",
            "⅛":"1/8","⅜":"3/8","⅝":"5/8","⅞":"7/8",
        }
        for u, f in mapping.items():
            s = s.replace(u, " "+f)
        return s

    v = (val or "").strip()
    if not v:
        raise ValueError("Empty dimension")

    v = normalize_unicode(v)

    # If ends with ", process as inches
    if v.endswith('"'):
        v = v[:-1].strip()
        v = v.replace("-", " ")
        toks = v.split()
        total_in = 0.0
        for t in toks:
            if "/" in t:
                # fraction
                n, d = t.split("/", 1)
                total_in += float(n)/float(d)
            else:
                total_in += float(t)
        return total_in * 25.4

    # Otherwise mm
    if v.lower().endswith("mm"):
        v = v[:-2].strip()

    return float(v)


# ============================================================
#  PARTS PARSER
# ============================================================

def parse_parts_from_form(form, edge_thickness):
    names = form.getlist("part_name")
    lens  = form.getlist("final_length")
    wids  = form.getlist("final_width")
    bls   = form.getlist("band_len")
    bws   = form.getlist("band_wid")
    rots  = form.getlist("can_rotate")
    groups= form.getlist("grain_group")
    orders= form.getlist("grain_order")
    qtys  = form.getlist("quantity")

    rows_out = []
    parts = []
    errors = []

    row_count = max(len(names), len(lens), len(wids))

    for i in range(row_count):
        name = names[i] if i < len(names) else ""
        L    = lens[i]  if i < len(lens)  else ""
        W    = wids[i]  if i < len(wids)  else ""
        bl   = bls[i]   if i < len(bls)   else "0"
        bw   = bws[i]   if i < len(bws)   else "0"
        rot  = rots[i]  if i < len(rots)  else "yes"
        grp  = groups[i]if i < len(groups)else ""
        ordv = orders[i]if i < len(orders)else "0"
        qty  = qtys[i]  if i < len(qtys)  else "1"

        # NEW — hinge hole checkbox value
        hinge_val = form.get(f"hinge_holes_{i}", "0")
        has_hinge = (hinge_val == "1")

        row_out = {
            "name": name,
            "final_length": L,
            "final_width": W,
            "band_len": bl,
            "band_wid": bw,
            "can_rotate": rot,
            "grain_group": grp,
            "grain_order": ordv,
            "quantity": qty,
            "hinge_holes": has_hinge,   # NEW
        }
        rows_out.append(row_out)

        if not name and not L and not W:
            continue

        try:
            Lmm = parse_dimension_mm(L)
            Wmm = parse_dimension_mm(W)
            qty_i = int(qty or "0")
            bl_i = int(bl or "0")
            bw_i = int(bw or "0")
            rot_b = parse_bool(rot)
            grp_v = grp if grp.strip() else None
            ord_i = int(ordv or "0")

            cut_len = Lmm - edge_thickness * bw_i
            cut_wid = Wmm - edge_thickness * bl_i

            parts.append(
                Part(
                    name=name,
                    width=cut_wid,
                    height=cut_len,
                    quantity=qty_i,
                    final_width=Wmm,
                    final_length=Lmm,
                    band_width_sides=bw_i,
                    band_length_sides=bl_i,
                    can_rotate=rot_b,
                    grain_group=grp_v,
                    grain_order=ord_i,
                    has_hinge_holes=has_hinge,  # NEW
                )
            )

        except Exception as e:
            errors.append(f"Row {i+1} ({name or 'unnamed'}): {e}")


    return parts, errors, rows_out


# ============================================================
#  BOARD INVENTORY PARSER
# ============================================================

def parse_board_inventory_from_form(form):
    """
    Reads optional board inventory from the form.

    Fields expected (array inputs):
        inv_length
        inv_width
        inv_qty
    """
    lens = form.getlist("inv_length")
    wids = form.getlist("inv_width")
    qtys = form.getlist("inv_qty")

    row_count = max(len(lens), len(wids), len(qtys))
    rows_out = []
    inventory = []
    errors = []

    for i in range(row_count):
        L = lens[i] if i < len(lens) else ""
        W = wids[i] if i < len(wids) else ""
        Q = qtys[i] if i < len(qtys) else ""

        rows_out.append({
            "length": L,
            "width": W,
            "qty": Q
        })

        # Completely blank row → ignore
        if not L and not W and not Q:
            continue

        # If length or width is empty, treat row as "unused"
        # whenever quantity is empty or zero.
        if (not L or not W):
            # try to read quantity; if <= 0, just ignore row
            try:
                q_val = int(Q or "0")
            except ValueError:
                q_val = 0
            if q_val <= 0:
                continue
            # positive quantity but missing dimension → real error
            errors.append(f"Board row {i+1}: Empty dimension")
            continue

        # Now we have both L and W present; quantity defaults to 1
        try:
            Lmm = parse_dimension_mm(L)
            Wmm = parse_dimension_mm(W)
            Qi  = int(Q or "1")
            if Qi <= 0:
                # zero or negative quantity with valid dims → just ignore
                continue

            inventory.append(
                BoardStock(length=Lmm, width=Wmm, quantity=Qi)
            )
        except Exception as e:
            errors.append(f"Board row {i+1}: {e}")

    return inventory, rows_out, errors



# ============================================================
#  FREE-RECT MERGING FOR OFFCUTS
# ============================================================

def merge_free_rects(free_rects):
    """
    Merge axis-aligned free rectangles that are exactly adjacent
    (share a full edge) in the internal coordinate system:

        x      = horizontal offset (left)
        y      = vertical offset (top)
        width  = horizontal size
        height = vertical size

    This is ONLY for nicer SVG offcut display; it does NOT affect
    the packing itself.
    """
    rects = list(free_rects)
    merged_any = True

    while merged_any:
        merged_any = False
        new_rects = []
        used = [False] * len(rects)

        for i in range(len(rects)):
            if used[i]:
                continue

            a = rects[i]
            ax, ay, aw, ah = a.x, a.y, a.width, a.height
            current = a

            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue

                b = rects[j]
                bx, by, bw, bh = b.x, b.y, b.width, b.height

                merged = None

                # ---- Horizontal merge (side-by-side) ----
                # Same vertical span, same height, touching in x
                if ay == by and ah == bh:
                    # b is immediately to the right of a
                    if ax + aw == bx:
                        merged = FreeRect(ax, ay, aw + bw, ah)
                    # a is immediately to the right of b
                    elif bx + bw == ax:
                        merged = FreeRect(bx, by, aw + bw, ah)

                # ---- Vertical merge (stacked) ----
                # Same horizontal span, same width, touching in y
                if merged is None and ax == bx and aw == bw:
                    # b is immediately below a
                    if ay + ah == by:
                        merged = FreeRect(ax, ay, aw, ah + bh)
                    # a is immediately below b
                    elif by + bh == ay:
                        merged = FreeRect(ax, by, aw, ah + bh)

                if merged is not None:
                    current = merged
                    ax, ay, aw, ah = current.x, current.y, current.width, current.height
                    used[j] = True
                    merged_any = True

            used[i] = True
            new_rects.append(current)

        rects = new_rects

    return rects

def filter_used_boards(boards: List[BoardLayout]) -> List[BoardLayout]:
    """
    Keep only boards that actually have parts placed on them
    (used area > 0), and renumber board_index on the parts
    so SVG labels stay sequential (Board 1, Board 2, ...).

    This is purely a presentation step; it does not change the
    nesting itself.
    """
    used: List[BoardLayout] = []

    # First, keep only boards with at least one part
    for b in boards:
        used_area = sum(p.width * p.height for p in b.parts)
        if used_area > 0:
            used.append(b)

    # Now renumber board_index for all remaining parts
    for new_idx, b in enumerate(used):
        for p in b.parts:
            p.board_index = new_idx

    return used


# === CHUNK 4/7: END ===
# === CHUNK 5/7: BEGIN ===

# ============================================================
#  SVG GENERATION
# ============================================================

def generate_svg_layout(boards, board_length, board_width, parts_file_name=None):
    """
    Generates SVG for ALL boards in one string.

    Each board may have its own size; we scale all using the
    longest board length so they are comparable.
    """

    # Timestamp for printout
    from datetime import datetime
    import pytz
    cet = pytz.timezone("Europe/Berlin")
    timestamp_str = datetime.now(cet).strftime("%Y-%m-%d %H:%M")



    padding = 40
    max_board_length_px = 800.0  # horizontal space for LENGTH

    # Small text for headers / yield
    FS_BOARD_LABEL = 12   # "Board 1/3 (xxxx×yyyy)"
    FS_FILE_NAME   = 12   # file name
    FS_YIELD       = 12   # "Yield: 72.8%"

    # Part / offcut labels
    FS_TEXT   = 13
    FS_OFFCUT = 11

    # Strip extension once, so SVG shows just base name
    base_file_name = None
    if parts_file_name:
        base_file_name = parts_file_name.rsplit(".", 1)[0]

    if boards:
        max_length = max(b.height for b in boards)
    else:
        max_length = board_length or 1.0

    scale = max_board_length_px / max_length

    all_svgs = []

    for b_index, board in enumerate(boards):
        board_len = board.height   # treat height as LENGTH
        board_wid = board.width    # width as WIDTH

        board_len_px = board_len * scale
        board_wid_px = board_wid * scale

        svg_w = board_len_px + 2 * padding
        svg_h = board_wid_px + 2 * padding

        out = []
        out.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{svg_w:.0f}" height="{svg_h:.0f}">'
        )
        out.append('<style>text{font-family:Arial,Helvetica,sans-serif;}</style>')

        # Fine-dot pattern for offcuts
        out.append("""
        <defs>
            <pattern id="fineDots" patternUnits="userSpaceOnUse" width="6" height="6">
                <circle cx="1" cy="1" r="0.6" fill="#cccccc" />
            </pattern>
        </defs>
        """)

        ox = padding
        oy = padding

        # --- Board outline ---
        out.append(
            f'<rect x="{ox:.1f}" y="{oy:.1f}" '
            f'width="{board_len_px:.1f}" height="{board_wid_px:.1f}" '
            f'style="fill:white;stroke:black;stroke-width:2"/>'
        )

        # Merge free rectangles for nicer offcut display
        merged_offcuts = merge_free_rects(board.free_rects)

        # --- OFFCUTS (fine dotted fill) ---
        for fr in merged_offcuts:
            sx = ox + fr.y * scale
            sy = oy + fr.x * scale
            sw = fr.height * scale
            sh = fr.width  * scale

            out.append(
                f'<rect x="{sx - 0.5:.1f}" y="{sy - 0.5:.1f}" '
                f'width="{sw + 1:.1f}" height="{sh + 1:.1f}" '
                f'style="fill:url(#fineDots);stroke:#cccccc;stroke-width:0.5"/>'
            )

        # ---------- HEADER TEXTS (top-left) ----------

        total_boards = len(boards)
        board_text = f"Board {b_index+1}/{total_boards} ({board_len:.0f}x{board_wid:.0f})"

        # We put FILE NAME on the first line, BOARD TEXT on the second line
        header_y1 = oy - 18   # first line
        header_y2 = oy - 4    # second line (a bit above the board outline)

        if base_file_name:
            # Line 1: file name (no extension)
            out.append(
                f'<text x="{ox:.1f}" y="{header_y1:.1f}" '
                f'font-size="{FS_FILE_NAME}" fill="black">{base_file_name}</text>'
            )
            # Line 2: board label
            out.append(
                f'<text x="{ox:.1f}" y="{header_y2:.1f}" '
                f'font-size="{FS_BOARD_LABEL}" fill="black">{board_text}</text>'
            )
        else:
            # No file name → just show board text slightly above the board
            out.append(
                f'<text x="{ox:.1f}" y="{header_y2:.1f}" '
                f'font-size="{FS_BOARD_LABEL}" fill="black">{board_text}</text>'
            )

        # ---------- PARTS ----------
        for p in board.parts:
            if p.board_index != b_index:
                continue

            # Map internal coords to SVG
            x = ox + p.y * scale
            y = oy + p.x * scale
            w = p.height * scale   # CUT length
            h = p.width  * scale   # CUT width

            # Part rectangle
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" '
                f'width="{w:.1f}" height="{h:.1f}" '
                f'style="fill:white;stroke:black;stroke-width:1"/>'
            )

            # Dimension strings with banding markers (#)
            # NOTE:
            #   - Horizontal side in the SVG has length p.height (we map board-length → SVG width)
            #   - Vertical side in the SVG has length p.width  (we map board-width  → SVG height)
            #
            # So horiz_val must always be p.height, vert_val always p.width.
            # Only the banding counts swap when the part is rotated.
            horiz_val = p.height      # length of horizontal side in SVG
            vert_val  = p.width       # length of vertical side in SVG

            if not p.rotated:
                # Unrotated: board length aligns with part "length"
                horiz_band = p.band_length_sides
                vert_band  = p.band_width_sides
            else:
                # Rotated: board length now aligns with part "width"
                horiz_band = p.band_width_sides
                vert_band  = p.band_length_sides


            len_str = f"{horiz_val:.0f}" + ("#" * horiz_band)
            wid_str = f"{vert_val:.0f}"  + ("#" * vert_band)

            # Length text (top centre)
            tx = x + w / 2
            ty = y + FS_TEXT + 2
            out.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" '
                f'font-size="{FS_TEXT}" text-anchor="middle" fill="black">{len_str}</text>'
            )

            # Width text (left centre)
            wx = x + 4
            wy = y + h / 2
            out.append(
                f'<text x="{wx:.1f}" y="{wy:.1f}" '
                f'font-size="{FS_TEXT}" text-anchor="start" fill="black">{wid_str}</text>'
            )

            # NEW — hinge holes symbol (centered)
            if getattr(p, "has_hinge_holes", False):
                hinge_symbol = "⬤_⬤"   # the symbol you requested
                hx = x + w / 2
                hy = y + h / 2 + FS_TEXT / 2
                out.append(
                    f'<text x="{hx:.1f}" y="{hy:.1f}" '
                    f'font-size="{FS_TEXT}" text-anchor="middle" fill="black">{hinge_symbol}</text>'
                )


            # Name label (bottom-right)
           
            if p.name:
                raw = p.name
                if "_" in raw:
                    base, last = raw.rsplit("_", 1)
                    label = base if last.isdigit() else raw
                else:
                    label = raw

                lx = x + w - 4
                ly = y + h - 4
                out.append(
                    f'<text x="{lx:.1f}" y="{ly:.1f}" '
                    f'font-size="{FS_TEXT}" text-anchor="end" fill="black">{label}</text>'
                )
            else:
                lx = x + w - 4
                ly = y + h - 4

            # Icons / grain group labels
            icon_x = lx - 18
            icon_y = ly - 2 * FS_TEXT + 4

            if p.grain_group:
                # Show grain-group name, nudged left
                group_label = p.grain_group
                out.append(
                    f'<text x="{icon_x - 19:.1f}" y="{icon_y + FS_TEXT:.1f}" '
                    f'font-size="{FS_TEXT}" text-anchor="start" fill="black">≈ {group_label}</text>'
                )

            # NOTE: we intentionally do NOT draw anything for non-rotating parts now.



        # --- OFFCUT LABELS ---
        for fr in merged_offcuts:
            sx = ox + fr.y * scale
            sy = oy + fr.x * scale
            sw = fr.height * scale
            sh = fr.width  * scale

            off_len = fr.height
            off_wid = fr.width
            label = f"{off_len:.0f}×{off_wid:.0f}"

            tx = sx + sw / 2
            ty = sy + sh / 2 + 4

            out.append(
                f'<text x="{tx:.1f}" y="{ty:.1f}" '
                f'font-size="{FS_OFFCUT}" text-anchor="middle" '
                f'fill="#999999">{label}</text>'
            )

        # --- YIELD for this board ---
        board_area = board.width * board.height
        used_area = sum(pp.width * pp.height for pp in board.parts
                        if pp.board_index == b_index)
        yield_pct = 100 * used_area / board_area if board_area else 0

        # Position of the yield label (bottom-right inside the board)
        yield_x = ox + board_len_px - 4
        yield_y = oy + board_wid_px - 4

        # Yield text
        out.append(
            f'<text x="{yield_x:.1f}" '
            f'y="{yield_y:.1f}" '
            f'font-size="{FS_YIELD}" text-anchor="end" fill="black">'
            f'Yield: {yield_pct:.1f}%</text>'
        )

        # Tagline just below the board, under the yield label
        tagline_y = yield_y + FS_YIELD + 4
        out.append(
            f'<text x="{yield_x:.1f}" '
            f'y="{tagline_y:.1f}" '
            f'font-size="{FS_YIELD}" text-anchor="end" fill="black">'
            f'Cut Guru by Quality Postform</text>'
        )

        # --- DATE/TIME STAMP (same line as tagline, left side) ---
        stamp_x = ox
        stamp_y = tagline_y

        out.append(
            f'<text x="{stamp_x:.1f}" '
            f'y="{stamp_y:.1f}" '
            f'font-size="{FS_YIELD}" text-anchor="start" fill="black">'
            f'Printed: {timestamp_str}</text>'
        )


        out.append("</svg>")
        all_svgs.append("\n".join(out))


    return "\n".join(all_svgs)


# ============================================================
#  TEXT REPORT
# ============================================================

def generate_text_report(boards,
                         board_width,
                         board_height,
                         used_area,
                         waste_area,
                         kerf,
                         edge_thickness):

    from collections import Counter

    lines = []

    all_parts = [p for b in boards for p in b.parts]
    total_pieces = len(all_parts)
    hinge_parts = sum(1 for p in all_parts if getattr(p, "has_hinge_holes", False))


    # Banding length
    overhang = 80.0
    band_len_total = 0.0

    for p in all_parts:
        if p.final_length is None or p.final_width is None:
            continue
        band_len_total += p.band_length_sides * (p.final_length + overhang)
        band_len_total += p.band_width_sides  * (p.final_width  + overhang)

    band_len_meters = band_len_total / 1000.0

    size_counts = Counter((b.height, b.width) for b in boards)

    # HEADER
    lines.append("Cut Guru - Nesting Report")
    lines.append("=================================")
    lines.append("")

    # ----------------------------------------------------
    # ✅ SUMMARY FIRST (BOLD)
    # ----------------------------------------------------
    lines.append(f"<strong>Boards used          : {len(boards)}</strong>")
    lines.append(f"<strong>Total pieces         : {total_pieces}</strong>")
    lines.append(f"<strong>Total banding length : {band_len_meters:.2f} m</strong>")
    lines.append(f"<strong>Hinge hole parts     : {hinge_parts}</strong>")
    lines.append("")

    # ----------------------------------------------------
    # ✅ BOARD SIZE / KERF / EDGE THICKNESS AFTERWARD (NOT BOLD)
    # ----------------------------------------------------
    if len(size_counts) == 1:
        (L, W), count = next(iter(size_counts.items()))
        lines.append(
            f"Board size           : length {L:.1f} mm, width {W:.1f} mm (x{count})"
        )
    else:
        lines.append("Board sizes          :")
        for (L, W), count in sorted(size_counts.items()):
            lines.append(f"  - {count} × {L:.1f} mm × {W:.1f} mm")

    lines.append(f"Saw kerf             : {kerf:.1f} mm")
    lines.append(f"Edge band thickness  : {edge_thickness:.1f} mm")
    lines.append("")

    return "\n".join(lines)

# === CHUNK 5/7: END ===


# === CHUNK 6/7: BEGIN ===

# ============================================================
#  FLASK APPLICATION - TEMPLATE
# ============================================================

app = Flask(__name__)

# Secret key used for sessions (login cookie)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# Password checker – reads APP_PASSWORD from environment
def check_password(pwd: str) -> bool:
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        # If no password is set, never let anyone in
        return False
    return pwd == expected

TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cut Guru - MaxRects</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .row { display: flex; gap: 20px; flex-wrap: wrap; }
        .card { border: 1px solid #ccc; padding: 15px; border-radius: 8px; margin-bottom: 20px; background: #fff; }
        .svg-container { border: 1px solid #ccc; padding: 10px; overflow: auto; background: #fafafa; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; font-size: 13px; }
        th, td { border: 1px solid #ddd; padding: 4px; text-align: center; }
        th { background: #f0f0f0; }
        input[type="text"], input[type="number"], select {
            width: 100%;
            box-sizing: border-box;
            font-size: 12px;
        }
        pre { background: #f5f5f5; padding: 10px; border-radius: 5px; max-height: 400px; overflow:auto; }
        button { cursor: pointer; }

        .cutlist-btn {
            width: 150px;                 /* exact width */
            display: inline-flex;         /* make button + label render identically */
            justify-content: center;      /* center text horizontally */
            align-items: center;          /* center text vertically */
            padding: 6px 0;               /* even padding */
            font-size: 14px;
            background: #eee;
            border: 1px solid #ccc;
            border-radius: 6px;
            cursor: pointer;
            box-sizing: border-box;       /* ensures exact size */
        }



        .rotate-yes {
            background: #dfd !important;  /* light green */
            border: 1px solid #8c8; 
        }


        .rotate-no {
            background: #fdd !important;  /* light red */
            border: 1px solid #c88;
        }

    </style>
</head>
<body>

<!-- Header with logo + title -->
<div style="display:flex; align-items:center; gap:30px; margin-bottom:25px;">

    <!-- Quality Postform logo -->
    <img src="{{ url_for('static', filename='logo.jpg') }}"
         alt="Company Logo"
         style="height:80px; object-fit:contain;">

    <!-- Thin divider line -->
    <div style="width:1px; height:60px; background:#ccc;"></div>

    <!-- Cut Guru logo (slightly smaller) -->
    <img src="{{ url_for('static', filename='cutguru_logo.png') }}"
         alt="Cut Guru Logo"
         style="height:55px; object-fit:contain;">

</div>


<form method="post">
    <div class="row">
        <!-- Board settings -->
        <div class="card" style="flex: 1 1 250px;">
            <h3>Board settings</h3>
            <label>Board length (mm):<br>
                <input type="number" step="0.1" name="board_length" value="{{ board_length }}" required>
            </label><br><br>
            <label>Board width (mm):<br>
                <input type="number" step="0.1" name="board_width" value="{{ board_width }}" required>
            </label><br><br>
            <label>Edge band thickness (mm):<br>
                <select name="edge_thickness" style="width:100%;">
                    <option value="1" {% if edge_thickness == 1 %}selected{% endif %}>1 mm</option>
                    <option value="2" {% if edge_thickness == 2 %}selected{% endif %}>2 mm</option>
                </select>
            </label><br><br>
            <label>Kerf (mm):<br>
                <input type="number" step="0.1" name="kerf" value="6" readonly
                       style="background:#f0f0f0; cursor:not-allowed;">
            </label>

        </div>

        <!-- Board inventory (optional) -->
        <div class="card" style="flex: 1 1 250px;">
            <h3>Board inventory (optional)</h3>
            <p style="font-size:12px;">
                Leave empty to assume unlimited boards of the size above.<br>
                Otherwise, list each board size you have in stock.
            </p>
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
                <thead>
                    <tr>
                        <th style="border:1px solid #ddd;">Length</th>
                        <th style="border:1px solid #ddd;">Width</th>
                        <th style="border:1px solid #ddd;">Qty</th>
                    </tr>
                </thead>
                <tbody id="board-inv-body">
                    {% for row in board_inv_rows %}
                    <tr>
                        <td style="border:1px solid #ddd;">
                            <input type="text" name="inv_length" value="{{ row.length }}">
                        </td>
                        <td style="border:1px solid #ddd;">
                            <input type="text" name="inv_width" value="{{ row.width }}">
                        </td>
                        <td style="border:1px solid #ddd;">
                            <input type="number" name="inv_qty" value="{{ row.qty or '0' }}" min="0">
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            <div style="margin-top:8px; display:flex; gap:10px;">
                <button type="button" onclick="clearBoardInventory()" style="background:#fdd;">Clear all</button>
            </div>
        </div>

        <!-- Parts list -->
        <div class="card" style="flex: 3 1 600px;">

            <!-- Title + input units button on same row -->
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <h3 style="margin:0;">Parts list</h3>

                <button
                    type="button"
                    id="unit-mode-btn"
                    onclick="toggleInputUnits()"
                    style="padding:10px 18px; font-size:14px; font-weight:bold; border-radius:6px;">
                    Input set to mm
                </button>


            </div>

            <p>
                One row per part. Leave blank rows unused.<br>
                Name may be blank.<br>
                Length/width: mm (e.g. <code>600</code>, <code>600mm</code>)
                or inches with fractions (e.g. <code>23 1/2"</code>, <code>23⅝"</code>).
            </p>

            <table id="parts-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Final length</th>
                            <th>Band len<br>(0/1/2)</th>
                            <th>Final width</th>
                            <th>Band wid<br>(0/1/2)</th>
                            <th>Qty</th>
                            <th>Can rotate</th>
                            <th>Hinge holes</th>
                            <th>Grain group</th>
                            <th>Grain order</th>
                        </tr>
                    </thead>

                <tbody id="parts-body">
                    {% for row in parts_rows %}
                    <tr>
                        <td>
                            <input type="text" name="part_name" value="{{ row.name }}">
                        </td>
                        <td>
                            <input type="text" name="final_length" value="{{ row.final_length }}">
                        </td>
                        <td>
                            <input type="number" name="band_len" value="{{ row.band_len or '0' }}" min="0" max="2">
                        </td>
                        <td>
                            <input type="text" name="final_width" value="{{ row.final_width }}">
                        </td>
                        <td>
                            <input type="number" name="band_wid" value="{{ row.band_wid or '0' }}" min="0" max="2">
                        </td>
                        <td>
                            <input type="number" name="quantity" value="{{ row.quantity or '0' }}" min="0">
                        </td>
                        <td>
                            <select name="can_rotate">
                                <option value="yes" {% if row.can_rotate|lower in ['yes','true','1'] %}selected{% endif %}>Yes</option>
                                <option value="no"  {% if row.can_rotate|lower in ['','no','false','0'] %}selected{% endif %}>No</option>
                            </select>
                        </td>

                        <!-- NEW HINGE HOLES COLUMN -->
                        <td>
                            <input type="checkbox"
                                   name="hinge_holes_{{ loop.index0 }}"
                                   value="1"
                                   {% if row.hinge_holes %}checked{% endif %}>
                            <input type="hidden"
                                   name="hinge_holes_{{ loop.index0 }}"
                                   value="0">
                        </td>

                        <td>
                            <input type="text" name="grain_group" value="{{ row.grain_group }}">
                        </td>
                        <td>
                            <input type="number" name="grain_order" value="{{ row.grain_order or '0' }}">
                        </td>
                    </tr>

                    {% endfor %}
                </tbody>
            </table>

            <br>

        <!-- FIRST ROW: Clear + Rotate -->
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">

            <!-- LEFT SIDE -->
            <button type="button" onclick="clearAllParts()" style="background:#fdd;">Clear all</button>

            <!-- RIGHT SIDE -->
            <div style="display:flex; align-items:center; gap:10px;">
                <span>Rotate all:</span>
                <button
                    type="button"
                    id="rotate-toggle-btn"
                    onclick="toggleRotateAll()"
                    style="padding:4px 12px;">
                    Rotate: ON
                </button>
            </div>

        </div>

            <!-- SECOND ROW: Save + Load -->
            <div style="display:flex; align-items:center; gap:20px; margin-bottom:10px;">

                <!-- Save cutlist -->
                <button type="button" onclick="exportPartsList()" class="cutlist-btn">
                    Save cutlist
                </button>

                <!-- Load cutlist -->
                <label for="parts-file-input"
                       class="cutlist-btn"
                       style="cursor:pointer; text-align:center;">
                    Load cutlist
                </label>

                <input
                    type="file"
                    id="parts-file-input"
                    accept=".json"
                    onchange="importPartsList(event)"
                    style="display:none;"
                >

                <!-- ✅ NOW inside the flex row -->
                <span id="parts-file-name-label"
                      style="font-size:12px; color:#333;">
                    {{ parts_file_name or '' }}
                </span>

            </div>

            <!-- hidden field stays where it was -->
            <input
                type="hidden"
                id="parts-file-name-hidden"
                name="parts_file_name"
                value="{{ parts_file_name or '' }}"
            >



        </div>  <!-- close Parts list .card -->
    </div>      <!-- close .row -->


    <button type="submit" style="padding:10px 20px; font-size:16px;">Calculate layout</button>
</form>

{% if errors %}
<div class="card error">
    <h3>Errors</h3>
    <ul>
    {% for e in errors %}
        <li>{{ e }}</li>
    {% endfor %}
    </ul>
</div>
{% endif %}

{% if report %}
<div class="row">
    <div class="card" style="flex:1 1 25%;">
        <h3>Report</h3>
        <pre>{{ report|safe }}</pre>
    </div>

    <div class="card svg-container" style="flex:3 1 75%;">
        <h3>SVG Layouts</h3>
        {{ svg|safe }}

        <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
            <button type="button" onclick="printLayouts()">Print 1 per page</button>
            <button type="button" onclick="printLayoutsTwoPerPage()">Print 2 per page</button>
        </div>


    </div>
</div>
{% endif %}

<script>
// =============== PARTS TABLE HELPERS ===============

// Create a new empty parts row (Qty default 0; changes to 1 on focus)
function addRow() {
    const tbody = document.getElementById('parts-body');
    const index = tbody.children.length;   // 0-based index for this row
    const tr = document.createElement('tr');

    tr.innerHTML =
        '<td><input type="text" name="part_name"></td>' +
        '<td><input type="text" name="final_length"></td>' +
        '<td><input type="number" name="band_len" value="0" min="0" max="2"></td>' +
        '<td><input type="text" name="final_width"></td>' +
        '<td><input type="number" name="band_wid" value="0" min="0" max="2"></td>' +
        '<td><input type="number" name="quantity" value="0" min="0"></td>' +
        '<td><select name="can_rotate">' +
            '<option value="yes">Yes</option>' +
            '<option value="no" selected>No</option>' +
        '</select></td>' +
        // Hinge holes column: checkbox + hidden, same naming scheme as template
        '<td style="text-align:center;">' +
            '<input type="checkbox" name="hinge_holes_' + index + '" value="1">' +
            '<input type="hidden"  name="hinge_holes_' + index + '" value="0">' +
        '</td>' +
        '<td><input type="text" name="grain_group"></td>' +
        '<td><input type="number" name="grain_order" value="0"></td>';

    tbody.appendChild(tr);
    return tr;
}



// Clear all parts and leave ONE empty row, focus Final length
function clearAllParts() {
    if (!confirm("Clear all parts?")) return;

    const tbody = document.getElementById('parts-body');
    tbody.innerHTML = "";
    const tr = addRow();
    const lenInput = tr.querySelector('input[name="final_length"]');
    if (lenInput) {
        lenInput.focus();
        lenInput.select();
    }
}

// Export current parts as JSON and update file name label
function exportPartsList() {
    const rows = document.querySelectorAll('#parts-body tr');
    const data = [];
    rows.forEach(row => {
        const obj = {};
        row.querySelectorAll('input, select').forEach(input => {
            obj[input.name] = input.value;
        });
        data.push(obj);
    });

    if (!data.length) {
        alert("No parts to export.");
        return;
    }

    const fileName = "parts.json";

    const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // update label + hidden field with base name (no extension)
    const baseName = fileName.replace(/\\.[^.]+$/, '');
    const hidden = document.getElementById('parts-file-name-hidden');
    const label  = document.getElementById('parts-file-name-label');
    if (hidden) hidden.value = baseName;
    if (label)  label.textContent = baseName;
}

// Import parts from JSON and update file name
function importPartsList(ev) {
    const file = ev.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const arr = JSON.parse(e.target.result);
            if (!Array.isArray(arr)) {
                alert("Invalid JSON (expected an array).");
                return;
            }

            const tbody = document.getElementById('parts-body');
            tbody.innerHTML = "";

            arr.forEach(row => {
                const tr = addRow();  // create row with correct defaults

                tr.querySelector('input[name="part_name"]').value     = row.part_name || row.name || "";
                tr.querySelector('input[name="final_length"]').value  = row.final_length || "";
                tr.querySelector('input[name="band_len"]').value      = row.band_len || "0";
                tr.querySelector('input[name="final_width"]').value   = row.final_width || "";
                tr.querySelector('input[name="band_wid"]').value      = row.band_wid || "0";
                tr.querySelector('input[name="quantity"]').value      = (row.quantity ?? "0");
                tr.querySelector('select[name="can_rotate"]').value   = (row.can_rotate || "no");
                tr.querySelector('input[name="grain_group"]').value   = row.grain_group || "";
                tr.querySelector('input[name="grain_order"]').value   = row.grain_order || "0";
            });

            if (!arr.length) {
                addRow();
            }

            // Update hidden + label with imported file name (base name only)
            const baseName = file.name.replace(/\\.[^.]+$/, '');
            const hidden = document.getElementById('parts-file-name-hidden');
            const label  = document.getElementById('parts-file-name-label');
            if (hidden) hidden.value = baseName;
            if (label)  label.textContent = baseName;

            alert("Parts list imported.");
        } catch (err) {
            alert("Error reading JSON: " + err.message);
        }
    };
    reader.readAsText(file);
}

// Set all rotate flags to yes/no
function setAllRotate(v) {
    document.querySelectorAll('select[name="can_rotate"]').forEach(sel => {
        sel.value = v;
        updateRotateColour(sel); // <-- NEW
    });
}


function toggleRotateAll() {
    const btn = document.getElementById('rotate-toggle-btn');
    const isOn = btn.dataset.state === "on";

    if (isOn) {
        setAllRotate("no");
        btn.dataset.state = "off";
        btn.textContent = "Rotate: OFF";
        btn.style.background = "#fdd";   // light red
    } else {
        setAllRotate("yes");
        btn.dataset.state = "on";
        btn.textContent = "Rotate: ON";
        btn.style.background = "#dfd";   // light green
    }
}

// =============== INPUT UNIT MODE (mm <-> inches) ===============

let inputUnitsMode = "mm";  // default

function toggleInputUnits() {
    const btn = document.getElementById('unit-mode-btn');
    const orange = "#D68A4A";

    if (inputUnitsMode === "mm") {
        inputUnitsMode = "in";
        if (btn) {
            btn.textContent = "Input set to inches";
            btn.style.background = orange;
            btn.style.color = "white";
            btn.style.border = "2px solid #b26d34";
        }
    } else {
        inputUnitsMode = "mm";
        if (btn) {
            btn.textContent = "Input set to mm";
            btn.style.background = "#eee";
            btn.style.color = "black";
            btn.style.border = "1px solid #ccc";
        }
    }
}





// =============== BOARD INVENTORY HELPERS ===============

function addBoardRow() {
    const tbody = document.getElementById('board-inv-body');
    if (!tbody) return;
    const tr = document.createElement('tr');
    tr.innerHTML =
        '<td style="border:1px solid #ddd;"><input type="text" name="inv_length"></td>' +
        '<td style="border:1px solid #ddd;"><input type="text" name="inv_width"></td>' +
        '<td style="border:1px solid #ddd;"><input type="number" name="inv_qty" value="0" min="0"></td>';
    tbody.appendChild(tr);
    return tr;
}

function clearBoardInventory() {
    if (!confirm("Clear all board inventory rows?")) return;
    const tbody = document.getElementById('board-inv-body');
    tbody.innerHTML = "";
    addBoardRow(); // one blank row with qty 0
}

// =============== PRINTING HELPERS (1 per page) ===============
function printLayouts() {
    const svgs = document.querySelectorAll(".svg-container svg");
    if (!svgs.length) {
        alert("Nothing to print.");
        return;
    }

    const win = window.open("", "_blank");

    win.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Print layouts</title>
            <style>
                @page {
                    size: A4 landscape;
                    margin: 5mm;
                }
                body {
                    margin: 0;
                    padding: 0;
                }
                .sheet {
                    page-break-after: always;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    width: 100%;
                    height: 100vh;
                }
                svg {
                    width: 100%;
                    height: 100%;
                    max-width: 100%;
                    max-height: 100%;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
    `);

    svgs.forEach((svg) => {
        const cloned = svg.cloneNode(true);

        const w = parseFloat(cloned.getAttribute("width") || "1000");
        const h = parseFloat(cloned.getAttribute("height") || "1000");

        cloned.removeAttribute("width");
        cloned.removeAttribute("height");

        if (!cloned.getAttribute("viewBox")) {
            cloned.setAttribute("viewBox", `0 0 ${w} ${h}`);
        }

        win.document.write(`
            <div class="sheet">
                ${cloned.outerHTML}
            </div>
        `);
    });

    win.document.write("</body></html>");
    win.document.close();

    setTimeout(() => {
        win.focus();
        win.print();
    }, 200);
}

// =============== PRINTING HELPERS (2 per page, portrait) ===============
function printLayoutsTwoPerPage() {
    const svgs = document.querySelectorAll(".svg-container svg");
    if (!svgs.length) {
        alert("Nothing to print.");
        return;
    }

    const win = window.open("", "_blank");

    win.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Print layouts (2 per page)</title>
            <style>
                @page {
                    size: A4 portrait;
                    margin: 5mm;
                }
                body {
                    margin: 0;
                    padding: 0;
                }
                .sheet {
                    page-break-after: always;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-around;
                    align-items: center;
                    min-height: 100vh;
                }
                .board-wrapper {
                    width: 100%;
                    flex: 1;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .board-wrapper svg {
                    width: 95%;
                    height: auto;
                    max-height: 48vh;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
    `);

    // Group SVGs 2 per page
    const svgArray = Array.from(svgs);
    for (let i = 0; i < svgArray.length; i += 2) {
        win.document.write('<div class="sheet">');

        for (let j = i; j < i + 2 && j < svgArray.length; j++) {
            const cloned = svgArray[j].cloneNode(true);

            const w = parseFloat(cloned.getAttribute("width") || "1000");
            const h = parseFloat(cloned.getAttribute("height") || "1000");

            // Use viewBox for scaling instead of fixed size
            cloned.removeAttribute("width");
            cloned.removeAttribute("height");

            if (!cloned.getAttribute("viewBox")) {
                cloned.setAttribute("viewBox", `0 0 ${w} ${h}`);
            }

            win.document.write(`
                <div class="board-wrapper">
                    ${cloned.outerHTML}
                </div>
            `);
        }

        win.document.write('</div>');
    }

    win.document.write("</body></html>");
    win.document.close();

    setTimeout(() => {
        win.focus();
        win.print();
    }, 200);
}


// =============== ENTER-BASED NAVIGATION + QTY DEFAULTS ===============


function updateRotateColour(selectEl) {
    if (!selectEl) return;
    if (selectEl.value.toLowerCase() === "yes") {
        selectEl.classList.add("rotate-yes");
        selectEl.classList.remove("rotate-no");
    } else {
        selectEl.classList.add("rotate-no");
        selectEl.classList.remove("rotate-yes");
    }
}

document.addEventListener("change", function(e) {
    if (e.target && e.target.name === "can_rotate") {
        updateRotateColour(e.target);
    }
});


document.addEventListener('DOMContentLoaded', function() {
    // Colour all rotate selects on page load
    document.querySelectorAll('select[name="can_rotate"]').forEach(sel => {
        updateRotateColour(sel);
    });

    // Make sure at least one parts row exists
    const partsBody = document.getElementById('parts-body');
    if (partsBody && partsBody.children.length === 0) {
        addRow();
    }

        // Set initial label for input units button
    const unitBtn = document.getElementById('unit-mode-btn');
    if (unitBtn) {
        unitBtn.textContent = "Input set to mm";
        unitBtn.style.background = "";
    }

    // On form submit: if inputUnitsMode is inches, auto-append " to bare numeric dimensions
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function() {
            if (inputUnitsMode !== "in") return;

            // These fields are treated as dimensions
            const selectors = [
                'input[name="final_length"]',
                'input[name="final_width"]',
                'input[name="inv_length"]',
                'input[name="inv_width"]'
            ];

            selectors.forEach(sel => {
                document.querySelectorAll(sel).forEach(inp => {
                    let v = (inp.value || "").trim();
                    if (!v) return;

                    // If user already explicitly used inches or mm or letters, leave it alone
                    if (v.endsWith('"')) return;
                    if (/[a-zA-Z]/.test(v)) return;  // e.g. "600mm"

                    // Otherwise, treat as inches input and append "
                    inp.value = v + '"';
                });
            });
        });
    }


    // Set initial state of rotate toggle
    const rotateBtn = document.getElementById('rotate-toggle-btn');
    if (rotateBtn) {
        rotateBtn.dataset.state = "on";
        rotateBtn.style.background = "#dfd";
    }

    // Default qty behaviour on focus:
    //  - parts list: quantity -> if 0 or empty, set to 1
    //  - board inventory: inv_qty -> if 0 or empty, set to 1
    document.addEventListener('focusin', function(e) {
        const t = e.target;
        if (!t) return;

        if (t.name === 'quantity') {
            if (t.value === '' || t.value === '0') {
                t.value = '1';
            }
        }

        if (t.name === 'inv_qty') {
            if (t.value === '' || t.value === '0') {
                t.value = '1';
            }
        }
    });

    // ENTER navigation in PARTS table
    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter') return;
        const target = e.target;
        if (!target.closest('#parts-body')) return;

        const row = target.closest('tr');
        if (!row) return;

        const nameInput  = row.querySelector('input[name="part_name"]');
        const lenInput   = row.querySelector('input[name="final_length"]');
        const bandLen    = row.querySelector('input[name="band_len"]');
        const widInput   = row.querySelector('input[name="final_width"]');
        const bandWid    = row.querySelector('input[name="band_wid"]');
        const qtyInput   = row.querySelector('input[name="quantity"]');

        e.preventDefault();

        // If we're on QTY: enforce non-empty length & width first
        if (target === qtyInput) {
            if (lenInput && !lenInput.value.trim()) {
                lenInput.focus();
                lenInput.select();
                return;
            }
            if (widInput && !widInput.value.trim()) {
                widInput.focus();
                widInput.select();
                return;
            }

            // Length & width are filled → add new row and go to its Final length
            const newRow = addRow();
            const newLen = newRow.querySelector('input[name="final_length"]');
            if (newLen) {
                newLen.focus();
                newLen.select();
            }
            return;
        }

        // Name → Final length
        if (target === nameInput && lenInput) {
            lenInput.focus();
            lenInput.select();
            return;
        }

        // Final length → Band len
        if (target === lenInput && bandLen) {
            bandLen.focus();
            bandLen.select();
            return;
        }

        // Band len → Final width
        if (target === bandLen && widInput) {
            widInput.focus();
            widInput.select();
            return;
        }

        // Final width → Band wid
        if (target === widInput && bandWid) {
            bandWid.focus();
            bandWid.select();
            return;
        }

        // Band wid → Qty
        if (target === bandWid && qtyInput) {
            qtyInput.focus();
            qtyInput.select();
            return;
        }
    });

    // ENTER navigation in BOARD INVENTORY table
    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter') return;
        const target = e.target;
        if (!target.closest('#board-inv-body')) return;

        const row = target.closest('tr');
        if (!row) return;

        const lenInput = row.querySelector('input[name="inv_length"]');
        const widInput = row.querySelector('input[name="inv_width"]');
        const qtyInput = row.querySelector('input[name="inv_qty"]');

        e.preventDefault();

        // Length → Width
        if (target === lenInput && widInput) {
            widInput.focus();
            widInput.select();
            return;
        }

        // Width → Qty
        if (target === widInput && qtyInput) {
            qtyInput.focus();
            qtyInput.select();
            return;
        }

        // On Qty: check length & width; if OK, add a new board row and focus its length
        if (target === qtyInput) {
            if (lenInput && !lenInput.value.trim()) {
                lenInput.focus();
                lenInput.select();
                return;
            }
            if (widInput && !widInput.value.trim()) {
                widInput.focus();
                widInput.select();
                return;
            }

            const newRow = addBoardRow();
            const newLen = newRow.querySelector('input[name="inv_length"]');
            if (newLen) {
                newLen.focus();
                newLen.select();
            }
            return;
        }
    });
});
</script>

</body>
</html>
"""
LOGIN_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cut Guru - Login</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            display:flex;
            justify-content:center;
            align-items:center;
            height:100vh;
            margin:0;
        }
        .card {
            background:white;
            padding:25px 30px;
            border-radius:10px;
            box-shadow:0 2px 8px rgba(0,0,0,0.15);
            min-width:280px;
            text-align:center;
        }
        h2 {
            margin-top:10px;
            margin-bottom:20px;
        }
        input[type="password"] {
            width:100%;
            padding:10px;
            font-size:14px;
            margin-top:12px;
            border:1px solid #ccc;
            border-radius:6px;
        }
        button {
            margin-top:15px;
            padding:10px 16px;
            font-size:14px;
            cursor:pointer;
            background:#4CAF50;
            color:white;
            border:none;
            border-radius:6px;
        }
        .error {
            color:#b00020;
            font-size:13px;
            margin-top:10px;
        }
        .logo-group {
            display:flex;
            justify-content:center;
            align-items:center;
            gap:20px;
            margin-bottom:20px;
        }
        .logo-group img {
            height:60px;
            object-fit:contain;
        }
        .divider {
            width:1px;
            height:40px;
            background:#ccc;
        }
    </style>
</head>
<body>
    <div class="card">

        <div class="logo-group">
            <img src="{{ url_for('static', filename='logo.jpg') }}" alt="QP Logo">
            <div class="divider"></div>
            <img src="{{ url_for('static', filename='cutguru_logo.png') }}" alt="CG Logo">
        </div>

        <h2>Cut Guru Login</h2>

        <form method="post">
            <label>
                <input type="password" name="password" placeholder="Enter password" autofocus>
            </label><br>
            <button type="submit">Enter</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
    </div>
</body>
</html>
"""

# === CHUNK 6/7: END ===


# === CHUNK 7/7: BEGIN ===

# ============================================================
#  AUTH HELPERS
# ============================================================

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


# ============================================================
#  ROUTES: LOGIN / LOGOUT
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        pwd = request.form.get("password", "")
        if check_password(pwd):
            session["logged_in"] = True
            return redirect("/")
        else:
            error = "Invalid password"
    return render_template_string(LOGIN_TEMPLATE, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ============================================================
#  ROUTE: INDEX (protected)
# ============================================================

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    # Default demo rows (you can change these)
    default_rows = [
        {
            "name": "",
            "final_length": "",
            "final_width": "",
            "band_len": "0",
            "band_wid": "0",
            "can_rotate": "no",
            "grain_group": "",
            "grain_order": "0",
            "quantity": "1",
        },
    ]
    # ... keep the rest of your index() code exactly as it was ...

    ctx = {
        "board_length": 2850,
        "board_width": 2100,
        "edge_thickness": 1.0,
        "kerf": 6.0,
        "parts_rows": default_rows,
        "board_inv_rows": [{"length": "", "width": "", "qty": ""}],
        "report": None,
        "svg": None,
        "errors": [],
        "parts_file_name": "",   # <-- ensure this exists
    }

    if request.method == "POST":
        try:
            board_length = float(request.form.get("board_length", "0"))
            board_width = float(request.form.get("board_width", "0"))
            edge_thickness = float(request.form.get("edge_thickness", "1"))
            kerf = 6.0
            ctx["kerf"] = kerf

            ctx["board_length"] = board_length
            ctx["board_width"] = board_width
            ctx["edge_thickness"] = edge_thickness
            ctx["kerf"] = kerf
            
            # NEW: store uploaded/exported parts filename
            ctx["parts_file_name"] = request.form.get("parts_file_name", "")

            # Parts
            parts, errors, rows = parse_parts_from_form(request.form, edge_thickness)
            ctx["parts_rows"] = rows
            ctx["errors"] = errors

            # Optional board inventory
            inv_boards, inv_rows, inv_errors = parse_board_inventory_from_form(request.form)
            ctx["board_inv_rows"] = inv_rows
            ctx["errors"].extend(inv_errors)

            if not ctx["errors"] and parts:
                # ---------- NESTING ----------
                if inv_boards:
                    # Use finite inventory of (possibly different) boards
                    boards, used_area, waste_area = nest_with_inventory(
                        inv_boards,
                        parts,
                        kerf,
                    )
                else:
                    # Infinite boards of a single size – explore multiple
                    # randomised layouts in both orientations and keep the best.
                    boards, used_area, waste_area = best_layout_global(
                        board_length,
                        board_width,
                        parts,
                        kerf,
                        tries_per_orientation=16,   # you can increase to 32/48 if still fast enough
                    )


                # NEW: remove any completely unused boards and renumber them
                boards = filter_used_boards(boards)

                # ---------- REPORT + SVG ----------
                report = generate_text_report(
                    boards,
                    board_width,   # width (for info only)
                    board_length,  # length (for info only)
                    used_area,
                    waste_area,
                    kerf,
                    edge_thickness,
                )
                svg = generate_svg_layout(
                    boards,
                    board_length,
                    board_width,
                    ctx.get("parts_file_name")  # <-- this bit
                )

                ctx["report"] = report
                ctx["svg"] = svg

        except Exception as e:
            ctx["errors"].append(str(e))

    return render_template_string(TEMPLATE, **ctx)



# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=5001)

# === CHUNK 7/7: END ===
