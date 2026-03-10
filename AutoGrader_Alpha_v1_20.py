#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# OMR Reader - version alpha1.20
__version__ = "alpha1.20"

"""
AutoGrader Alpha 1.20

# ================================================================
# AutoGrader Alpha
#
# Version: 1.20
# Release Date: 2026-03-09
#
# Changes from v1.18
# ------------------------------------------------
# NEW FEATURE 1: Process all PDFs in the directory given
#
# When given a directory instead of a single PDF, the script will merge all PDFs
# into a single one, then process all pages in it.
#
# NEW FEATURE 2: Automatic Skip for Unreadable Pages
#
# Added an optional command line argument:
#
#     --auto-skip-unreadable
#
# When enabled, the grader will automatically skip pages that cannot
# be processed due to image recognition or fiducial detection errors,
# such as:
#
#     RuntimeError: not enough fiducial candidates
#
# This typically occurs when the script encounters a page that is not
# an OMR form (e.g., extra instructions page, cover sheet, or blank page).
#
# Behavior:
#
# 1. If a page fails during grading, the script catches the exception
#    and marks the page as "skipped".
#
# 2. The skipped page is rendered as a normal PDF page without grading.
#
# 3. The skipped page is appended to the *previous successfully graded
#    exam page* in the output PDF so that the student packet remains intact.
#
# 4. Grading then continues with the next page in the input PDF.
#
# Additional Notes:
#
# • This feature works alongside the existing fixed skip logic:
#
#       --skip-n N
#
#   which skips pages in a regular pattern (e.g., every other page).
#
# • The new auto-skip logic is intended to handle unexpected pages
#   dynamically without stopping the grading process.
#
# • If skipped pages appear before the first graded page, they are
#   temporarily queued and attached to the first successfully graded
#   output file.
#
# • All other features and behavior from v1.18 remain unchanged.
#
# ================================================================

Dependencies:
    pip install opencv-python numpy pymupdf Pillow PyPDF2
    (optional) pip install pytesseract   # if you want OCR for email
"""

import argparse
import json
import os
import re
import sys
import math
import glob
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

from PIL import Image
from PyPDF2 import PdfReader, PdfWriter, PdfMerger

# ----------------------------- PDF RENDER -------------------------------------

def _normalize_page_index(requested: int, n: int) -> int:
    if n <= 0:
        raise IndexError("Empty PDF")
    if 0 <= requested < n:
        return requested
    if 1 <= requested <= n:
        return requested - 1
    raise IndexError(f"Bad page index {requested}; pages: 0..{n-1} or 1..{n}")

def get_pdf_page_count(pdf_path: str) -> int:
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("Rendering PDF requires PyMuPDF. Install with: pip install pymupdf") from e
    doc = fitz.open(pdf_path)
    n = len(doc)
    doc.close()
    return n

def render_pdf_page(pdf_path: str, page_index_raw: int, dpi: int = 300) -> Tuple[np.ndarray, float]:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("Rendering PDF requires PyMuPDF. Install with: pip install pymupdf") from e
    doc = fitz.open(pdf_path)
    i = _normalize_page_index(page_index_raw, len(doc))
    page = doc[i]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img, zoom

# ----------------------------- SPEC -------------------------------------------

@dataclass
class Fiducial:
    x: float
    y: float
    size: float

@dataclass
class Bubble:
    x: float
    y: float
    r: float

@dataclass
class RowSpec:
    question: int
    y: float
    col: int
    i_in_col: int
    bubbles: Dict[str, Bubble]

def load_spec(path: str):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    page = spec.get("page", {})
    W = int(page.get("width", 2550))
    H = int(page.get("height", 3300))

    fraw = spec.get("fiducials", [])
    if len(fraw) != 4:
        raise ValueError("Spec must have exactly 4 corner fiducials.")
    fids = [Fiducial(float(d["x"]), float(d["y"]), float(d["size"])) for d in fraw]

    rows = []
    for r in spec["rows"]:
        bubbles = {k: Bubble(float(r["bubbles"][k]["x"]),
                             float(r["bubbles"][k]["y"]),
                             float(r["bubbles"][k]["r"])) for k in ["A", "B", "C", "D"]}
        rows.append(RowSpec(int(r["question"]), float(r["y"]),
                            int(r.get("col", 0)), int(r.get("i_in_col", 0)), bubbles))
    col_defs = spec["columns"]["defs"]

    # Optional: form-version bubbles (A..F) for auto-detect
    form_meta = spec.get("form_version", {})
    form_bubbles = form_meta.get("bubbles", None)  # dict of letters to {x,y,r}
    form_choices = form_meta.get("choices", ["A","B","C","D","E","F"])

    # Optional: email ROI (design-space coordinates)
    email_roi = spec.get("email_roi", None)  # {"x":..,"y":..,"w":..,"h":..}

    return spec, rows, (W, H), fids, col_defs, form_bubbles, form_choices, email_roi

# ----------------------------- GEOMETRY HELPERS -------------------------------

def rotate_bound(img: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate by angle keeping full content. Returns (rotated_img, 3x3 homography)."""
    if abs(angle_deg) < 1e-3:
        return img, np.eye(3, dtype=np.float32)
    (h, w) = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2.0) - c[0]
    M[1, 2] += (nH / 2.0) - c[1]
    rotated = cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    H_aff = np.eye(3, dtype=np.float32)
    H_aff[0:2, :] = M
    return rotated, H_aff

def _angle_deg(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

# ----------------------------- FIDUCIAL DETECTION (GLOBAL) --------------------

def _fiducial_candidates(gray: np.ndarray):
    """Find square-ish blobs anywhere on the page; return (cx, cy, w, h)."""
    H, W = gray.shape[:2]
    area_img = float(H * W)

    def bins(g):
        b = cv2.GaussianBlur(g, (5, 5), 0)
        yield cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        yield cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    cand = []
    for thr in bins(gray):
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), 1)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 0.0005 * area_img or a > 0.05 * area_img:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if min(w, h) / max(w, h) < 0.80:
                continue
            hull = cv2.convexHull(c)
            solidity = a / (cv2.contourArea(hull) + 1e-6)
            if solidity < 0.9:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            cand.append((cx, cy, float(w), float(h)))

    # Keep the 12 biggest; then de-dup close centers
    cand.sort(key=lambda t: t[2] * t[3], reverse=True)
    cand = cand[:12]
    uniq = []
    for c in cand:
        if all(abs(c[0] - d[0]) > 6 or abs(c[1] - d[1]) > 6 for d in uniq):
            uniq.append(c)
    return uniq

def _assign_corners(cands, W, H):
    """
    Choose the OUTERMOST four squares and classify corners by extrema:
      TL=min(x+y), TR=max(x−y), BL=min(x−y), BR=max(x+y).
    """
    if len(cands) < 4:
        raise RuntimeError("not enough fiducial candidates")
    pts = np.array([[c[0], c[1]] for c in cands], np.float32)

    center = np.array([W / 2.0, H / 2.0], np.float32)
    d = np.linalg.norm(pts - center, axis=1)
    keep_idx = np.argsort(-d)[:6]
    pts = pts[keep_idx]

    sumxy = pts[:, 0] + pts[:, 1]
    diffxy = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(sumxy)]
    br = pts[np.argmax(sumxy)]
    bl = pts[np.argmin(diffxy)]
    tr = pts[np.argmax(diffxy)]

    # sanity ordering
    out = {"TL": tl, "TR": tr, "BL": bl, "BR": br}
    tl, tr, bl, br = out["TL"], out["TR"], out["BL"], out["BR"]
    if tl[0] > tr[0]: tl, tr = tr, tl
    if bl[0] > br[0]: bl, br = br, bl
    if tl[1] > bl[1]: tl, bl = bl, tl
    if tr[1] > br[1]: tr, br = br, tr
    return {"TL": tuple(tl), "TR": tuple(tr), "BL": tuple(bl), "BR": tuple(br)}

def _top_edge_residual_after(rot_deg, gray, corners):
    # Transform TL/TR with the rotation; measure residual angle
    _, Haff = rotate_bound(gray, rot_deg)
    pts_h = np.array([[corners[k][0], corners[k][1], 1.0] for k in ["TL", "TR"]], np.float32).T
    new = (Haff @ pts_h).T
    new = new[:, :2] / new[:, 2:3]
    return abs(_angle_deg(new[0], new[1]))

def deskew_by_fiducials(gray: np.ndarray, bgr: Optional[np.ndarray] = None,
                        debug_dir: Optional[str] = None):
    """Detect corners, try ±angle, pick rotation with smallest residual."""
    Hh, Ww = gray.shape[:2]
    cand = _fiducial_candidates(gray)
    corners = _assign_corners(cand, gray.shape[1], gray.shape[0])
    ang = _angle_deg(corners["TL"], corners["TR"])

    r1, r2 = -ang, +ang
    e1 = _top_edge_residual_after(r1, gray, corners)
    e2 = _top_edge_residual_after(r2, gray, corners)
    rot = r1 if e1 <= e2 else r2

    gray_out, H_aff = rotate_bound(gray, rot)
    bgr_out = None if bgr is None else rotate_bound(bgr, rot)[0]

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        dbg = cv2.cvtColor(gray_out, cv2.COLOR_GRAY2BGR)
        pts_h = np.array([[corners[k][0], corners[k][1], 1.0] for k in ["TL", "TR", "BL", "BR"]], np.float32).T
        new = (H_aff @ pts_h).T
        new = new[:, :2] / new[:, 2:3]
        for p in new:
            cv2.circle(dbg, (int(p[0]), int(p[1])), 14, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, "deskew_fiducials_debug.png"), dbg)

    print(f"[deskew] measured top-edge angle={ang:+.2f}°, "
          f"applied rotation={rot:+.2f}°, residual={min(e1, e2):.3f}°")
    return gray_out, bgr_out, corners, rot

# -------------------------- CANONICAL CORNER ORDERING -------------------------

def _classify_corners_xy(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Return dict {TL,TR,BL,BR} using extrema of x+y and x−y (robust to rotation/scale)."""
    pts = np.asarray(points, dtype=np.float32)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    bl = pts[np.argmin(d)]
    tr = pts[np.argmax(d)]
    return {"TL": tl, "TR": tr, "BL": bl, "BR": br}

def _centers_from_spec(fiducials: List[Fiducial]) -> np.ndarray:
    return np.array([[f.x + f.size / 2.0, f.y + f.size / 2.0] for f in fiducials], dtype=np.float32)

def _pair_src_dst_canon(src_pts_any_order: np.ndarray,
                        dst_pts_any_order: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (src, dst) both ordered [TL, TR, BL, BR]."""
    src_c = _classify_corners_xy(src_pts_any_order)
    dst_c = _classify_corners_xy(dst_pts_any_order)
    order = ["TL", "TR", "BL", "BR"]
    src_ord = np.stack([src_c[k] for k in order]).astype(np.float32)
    dst_ord = np.stack([dst_c[k] for k in order]).astype(np.float32)
    return src_ord, dst_ord

# ----------------------------- SAMPLING / SCORING -----------------------------

def sample_bubble_fill(binary, center, r, inset=1.0):
    """Return a fill score ~[0..1], higher = darker/more filled."""
    cx, cy = center
    rr = max(1.0, float(r) - inset)
    x0 = int(max(0, cx - rr))
    y0 = int(max(0, cy - rr))
    x1 = int(min(binary.shape[1], cx + rr))
    y1 = int(min(binary.shape[0], cy + rr))
    roi = binary[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    h, w = roi.shape[:2]
    Y, X = np.ogrid[:h, :w]
    mask = (X - (cx - x0))**2 + (Y - (cy - y0))**2 <= rr**2
    if not np.any(mask):
        return 0.0
    return float(1.0 - (roi[mask].mean() / 255.0))

# New, richer chooser for BLANK/MULTI
@dataclass
class ChoiceResult:
    status: str            # "OK" | "BLANK" | "MULTI"
    chosen: Optional[str]  # when status=="OK"
    winners: List[str]     # all options >= threshold (or close-tie)
    score_vec: Dict[str, float]

def choose_answer(scores: Dict[str, float], abs_thresh: float, margin: float) -> ChoiceResult:
    if not scores:
        return ChoiceResult("BLANK", None, [], {})
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    winners = [opt for opt, s in ordered if s >= abs_thresh]
    if len(winners) == 0:
        return ChoiceResult("BLANK", None, [], scores)
    if len(winners) > 1:
        return ChoiceResult("MULTI", None, winners, scores)
    # single candidate above threshold; check separation
    best_opt, best_val = ordered[0]
    runner_val = ordered[1][1] if len(ordered) > 1 else 0.0
    if (best_val - runner_val) < margin and runner_val >= abs_thresh:
        return ChoiceResult("MULTI", None, [best_opt, ordered[1][0]], scores)
    return ChoiceResult("OK", best_opt, [best_opt], scores)

# ----------------------------- FIDUCIAL TRANSLATION ---------------------------

def _find_local_dark_square_center(img_gray, cx_exp, cy_exp, search_radius=60):
    h, w = img_gray.shape[:2]
    x0 = max(0, int(cx_exp - search_radius))
    x1 = min(w, int(cx_exp + search_radius))
    y0 = max(0, int(cy_exp - search_radius))
    y1 = min(h, int(cy_exp + search_radius))
    roi = img_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = x0 + M["m10"]/M["m00"]
    cy = y0 + M["m01"]/M["m00"]
    return (float(cx), float(cy))

def _get_expected_fiducials_from_json(spec):
    if isinstance(spec, dict):
        for key in ("fiducials", "anchors", "markers"):
            if key in spec and isinstance(spec[key], (list, tuple)) and len(spec[key]) >= 2:
                pts = []
                for f in spec[key]:
                    x = f.get("x") if isinstance(f, dict) else None
                    y = f.get("y") if isinstance(f, dict) else None
                    if x is not None and y is not None:
                        pts.append((float(x), float(y)))
                if len(pts) >= 2:
                    return pts
        W = int(spec.get("width", 2550))
        H = int(spec.get("height", 3300))
    else:
        W, H = 2550, 3300
    return [(80, 80), (W-80, 80)]

def translate_align_by_fiducials(warped_gray, warped_bgr, spec_dict, debug=False):
    H, W = warped_gray.shape[:2]
    expected = _get_expected_fiducials_from_json(spec_dict)
    found = []
    for (ex, ey) in expected:
        pt = _find_local_dark_square_center(warped_gray, ex, ey, search_radius=70)
        if pt is not None:
            found.append(((ex, ey), pt))

    if not found:
        if debug:
            print("[fidshift] No fiducials found near expected positions; skipping translation.")
        return warped_gray, warped_bgr, (0.0, 0.0)

    dxs, dys = [], []
    for (ex, ey), (cx, cy) in found:
        dxs.append(cx - ex)
        dys.append(cy - ey)
    dx = float(np.median(dxs))
    dy = float(np.median(dys))

    M = np.array([[1, 0, -dx],
                  [0, 1, -dy]], dtype=np.float32)

    shifted_gray = cv2.warpAffine(warped_gray, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    shifted_bgr  = cv2.warpAffine(warped_bgr,  M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if debug:
        print(f"[fidshift] Translation applied: dx={dx:.2f}, dy={dy:.2f}")
    return shifted_gray, shifted_bgr, (dx, dy)

# ----------------------------- GRADED PDF (ALIGNED) ---------------------------

def write_designspace_graded_pdf(warped_bgr: np.ndarray, dpi: int,
                                 mark_icons_design: List[Tuple[float, float, bool, str]],
                                 correct_boxes_design: List[Tuple[float, float, float, float]],
                                 status_texts: List[Tuple[float, float, str]],
                                 score_text: str, out_base_from: str,
                                 jpeg_quality: int = 85,
                                 grayscale: bool = False) -> str:
    """
    Draws:
      - green ✓ (True) / red ✗ (False) at mark_icons_design[(x,y,is_correct,'OK'|'BLANK'|'MULTI')]
      - black squares around correct answer bubbles using correct_boxes_design[(x0,y0,x1,y1)]
      - small status annotations near rows (status_texts: (x,y,'BLANK'/'MULTI'))
      - score text
    Background image is embedded as JPEG (optionally grayscale) to reduce file size.
    """
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("Writing graded PDF requires PyMuPDF. Install with: pip install pymupdf") from e

    H, W = warped_bgr.shape[:2]
    zoom = dpi / 72.0
    page_w_pt, page_h_pt = W / zoom, H / zoom

    doc = fitz.open()
    page = doc.new_page(width=page_w_pt, height=page_h_pt)

    # ---- background image: JPEG (optionally grayscale) ----
    if grayscale:
        bg = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)  # keep 3 channels for JPEG
    else:
        rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)

    quality = int(np.clip(jpeg_quality, 1, 100))
    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("encode JPEG failed")
    page.insert_image(fitz.Rect(0, 0, page_w_pt, page_h_pt), stream=buf.tobytes())

    # ---- squares around correct bubbles ----
    if correct_boxes_design:
        boxes = page.new_shape()
        for (x0, y0, x1, y1) in correct_boxes_design:
            boxes.draw_rect(fitz.Rect(x0/zoom, y0/zoom, x1/zoom, y1/zoom))
        boxes.finish(color=(0, 0, 0), width=1.2)
        boxes.commit()

    # ---- ✓ / ✗ icons and status texts ----
    for (x, y, is_ok, status) in mark_icons_design:
        shape = page.new_shape()
        x_pt, y_pt = x / zoom, y / zoom
        s = 8.0
        if status == "OK":
            if is_ok:
                # green check
                shape.draw_line((x_pt - s,   y_pt),       (x_pt - s/3, y_pt + s/2))
                shape.draw_line((x_pt - s/3, y_pt + s/2), (x_pt + s,   y_pt - s))
                shape.finish(color=(0, 0.7, 0), width=1.5)
            else:
                # red X
                shape.draw_line((x_pt - s, y_pt - s), (x_pt + s, y_pt + s))
                shape.draw_line((x_pt + s, y_pt - s), (x_pt - s, y_pt + s))
                shape.finish(color=(0.85, 0, 0), width=1.5)
        else:
            # For BLANK/MULTI, draw a neutral mark (gray dot)
            shape.draw_circle((x_pt, y_pt), s/3)
            shape.finish(color=(0.2, 0.2, 0.2), fill=(0.8,0.8,0.8), width=1.0)
        shape.commit()

    for (x, y, text) in status_texts:
        page.insert_text((x/zoom, y/zoom), text, color=(0.8, 0.5, 0.0), fontsize=10, fontname="Helvetica")

    # score text
    if score_text:
        page.insert_text((36, 24), score_text, color=(0, 0, 1),
                         fontsize=24, fontname="Helvetica")

    out = out_base_from
    doc.save(out)
    doc.close()
    return out

# ----------------------------- KEY (MULTI-FORM) -------------------------------

def parse_key_file_multi_form(key_path: str) -> Dict[str, Dict[int, set]]:
    """
    Parse a key file containing sections [A]..[F] followed by lines like:
      1<TAB>A
    Returns: { 'A': {1:{'A'}, 2:{'B'}, ...}, 'B': {...}, ... }
    """
    forms: Dict[str, Dict[int, set]] = {}
    if not key_path:
        return forms
    current_form = None
    with open(key_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # section header like [A]
            msec = re.match(r"^\[([A-F])\]$", s, flags=re.IGNORECASE)
            if msec:
                current_form = msec.group(1).upper()
                if current_form not in forms:
                    forms[current_form] = {}
                continue
            if s.startswith("#"):
                continue
            # content line: q and answer(s)
            parts = s.split("\t")
            if len(parts) < 2:
                parts = s.split()
                if len(parts) < 2:
                    continue
            try:
                qnum = int(parts[0])
            except ValueError:
                continue
            ans_field = parts[1].strip()
            letters = [a.strip().upper() for a in ans_field.replace(";", ",").split(",") if a.strip()]
            if not letters:
                continue
            if current_form is None:
                # If no section has been set, assume 'A'
                current_form = "A"
                if current_form not in forms:
                    forms[current_form] = {}
            forms[current_form][qnum] = set(letters)
    return forms

# ----------------------------- FORM DETECTION ---------------------------------

def detect_form_letter(binary: np.ndarray,
                       form_bubbles: Optional[Dict[str, Dict[str, float]]],
                       abs_thresh: float = 0.22,
                       margin: float = 0.05) -> Optional[str]:
    """
    Sample the designated form bubbles (A..F) and return the detected letter.
    Returns None if unable to confidently detect.
    """
    if not form_bubbles:
        return None
    scores = {}
    for letter, b in form_bubbles.items():
        try:
            x, y, r = float(b["x"]), float(b["y"]), float(b["r"])
        except Exception:
            continue
        sc = sample_bubble_fill(binary, (x, y), r, inset=1.0)
        scores[str(letter).upper()] = sc

    # Simple winner selection with margin (reuse BLANK logic if needed)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ordered:
        return None
    topL, topV = ordered[0]
    secV = ordered[1][1] if len(ordered) > 1 else 0.0
    if topV < abs_thresh or (topV - secV) < margin:
        return None
    return topL

# ----------------------------- BUNDLE HELPERS ---------------------------------

def _bgr_to_pil_rgb(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def _render_raw_pages_as_temp_pdfs(pdf_path, dpi, start_idx, end_idx_excl):
    temps = []
    for p in range(start_idx, end_idx_excl):
        try:
            bgr, _ = render_pdf_page(pdf_path, p, dpi)
            img = _bgr_to_pil_rgb(bgr)
            import tempfile as _tf
            tmpf = _tf.NamedTemporaryFile(prefix="omr_skip_", suffix=".pdf", delete=False)
            tmpf.close()
            img.save(tmpf.name, format="PDF", resolution=dpi)
            temps.append(tmpf.name)
        except Exception as e:
            print(f"[bundle] failed to render page {p+1}: {e}")
    return temps

def _render_raw_page_as_temp_pdf(pdf_path, dpi, page_index):
    paths = _render_raw_pages_as_temp_pdfs(pdf_path, dpi, page_index, page_index + 1)
    return paths[0] if paths else ""

def _append_skipped_to_graded_pdf(graded_pdf_path, skipped_pdf_paths):
    if not graded_pdf_path or not skipped_pdf_paths:
        return graded_pdf_path
    writer = PdfWriter()
    with open(graded_pdf_path, "rb") as f:
        r = PdfReader(f)
        for pg in r.pages:
            writer.add_page(pg)
    for sp in skipped_pdf_paths:
        try:
            with open(sp, "rb") as f:
                r = PdfReader(f)
                for pg in r.pages:
                    writer.add_page(pg)
        except Exception as e:
            print(f"[bundle] failed to append {sp}: {e}")
    tmp_out = graded_pdf_path + ".tmp.pdf"
    with open(tmp_out, "wb") as f:
        writer.write(f)
    os.replace(tmp_out, graded_pdf_path)
    for sp in skipped_pdf_paths:
        try: os.remove(sp)
        except: pass
    return graded_pdf_path

# ----------------------------- EMAIL OCR HELPERS -------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

def sanitize_for_filename(s: str, fallback: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._\-@]", "_", s)
    return s if s else fallback

def email_local_part(email: str) -> str:
    if not email:
        return ""
    at = email.find("@")
    return email[:at] if at != -1 else email

def extract_text_ocr(img_bgr: np.ndarray, psm: int = 7) -> str:
    try:
        import pytesseract
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thr = cv2.medianBlur(thr, 3)
        cfg = f"--oem 1 --psm {psm}"
        return pytesseract.image_to_string(thr, config=cfg) or ""
    except Exception:
        return ""

def extract_email_from_design_roi(warped_bgr: np.ndarray, roi: Optional[dict]) -> Optional[str]:
    if not roi:
        return None
    try:
        x, y, w, h = [int(roi[k]) for k in ("x","y","w","h")]
    except Exception:
        return None
    H, W = warped_bgr.shape[:2]
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    w = max(1, min(W-x, w)); h = max(1, min(H-y, h))
    crop = warped_bgr[y:y+h, x:x+w]
    raw = extract_text_ocr(crop, psm=7)
    if not raw:
        return None
    m = EMAIL_RE.search(raw.replace("\n", " ").strip())
    return m.group(0) if m else None

# ----------------------------- PAGE PROCESSOR ---------------------------------

def process_one_page(pdf_path, dpi, page_index,
                     spec, rows, design_size, fiducials, col_defs,
                     forms_key_multi: Dict[str, Dict[int, set]],
                     write_graded_pdf, outdir,
                     perpage_csv_path,
                     debug_dir,
                     abs_thresh, margin,
                     jpeg_quality, grayscale,
                     form_bubbles=None,
                     email_roi=None,
                     forced_form_letter: Optional[str]=None, points_per_question=1.0,
                     output_base_label: Optional[str]=None):
    """
    Returns (out_pdf_path_or_empty, correct, total_keyed, chosen_map, form_letter, email_local, blank_count, multi_count, needs_manual)
    """
    design_w, design_h = design_size

    # Render & grayscale
    bgr, _ = render_pdf_page(pdf_path, page_index, dpi)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Deskew (rotation only)
    gray, bgr, corners, rot_deg = deskew_by_fiducials(gray, bgr, debug_dir or None)

    # GLOBAL fiducials → homography (canonical pairing)
    cand = _fiducial_candidates(gray)
    img_corners = _assign_corners(cand, gray.shape[1], gray.shape[0])
    src_pts_any = np.array([img_corners[k] for k in ["TL", "TR", "BL", "BR"]], np.float32)

    dst_pts_any = _centers_from_spec(fiducials)
    src_pts, dst_pts = _pair_src_dst_canon(src_pts_any, dst_pts_any)

    H_img2design, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.5)
    if H_img2design is None:
        raise RuntimeError("Homography failed on page {}".format(page_index))
    # Warp to design rectangle
    warped_gray = cv2.warpPerspective(gray, H_img2design, (design_w, design_h), flags=cv2.INTER_LINEAR)
    warped_bgr  = cv2.warpPerspective(bgr,  H_img2design, (design_w, design_h), flags=cv2.INTER_LINEAR)

    # Translation-only refinement (no scale/shear)
    try:
        warped_gray, warped_bgr, _t = translate_align_by_fiducials(warped_gray, warped_bgr, spec, debug=True)
    except Exception as _e:
        try:
            print(f"[fidshift] skipped due to error: {_e}")
        except Exception:
            pass

    # Threshold in design space
    blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # ---- Detect form letter (A..F) ----
    form_letter = (forced_form_letter or "").strip().upper() or detect_form_letter(binary, form_bubbles)
    if not form_letter or form_letter not in forms_key_multi:
        fallback = "A" if "A" in forms_key_multi else (sorted(forms_key_multi.keys())[:1] or ["A"])[0]
        print(f"[form] Unable to confidently detect form on page {page_index+1}; using '{fallback}'")
        form_letter = fallback
    else:
        print(f"[form] Detected exam form: {form_letter}")

    key_multi = forms_key_multi.get(form_letter, {})
    key_map = {q: sorted(list(s))[0] for q, s in key_multi.items()}  # first of possibly multiple correct

    # ---- Extract email (design space ROI) and local part ----
    email_full = extract_email_from_design_roi(warped_bgr, email_roi) or ""
    email_loc = sanitize_for_filename(email_local_part(email_full).lower(), "no_email")

    # Evaluate answers & overlay prep
    blank_count = 0
    multi_count = 0
    needs_manual = False
    correct = 0
    total_keyed = 0
    chosen_map: Dict[int, str] = {}

    mark_icons = []      # (x,y,is_correct,status)
    status_texts = []    # (x,y,"BLANK"/"MULTI")
    correct_boxes = []   # around key answers

    for row in rows:
        # Skip non-keyed questions entirely (by requirement)
        if row.question not in key_map:
            continue
        total_keyed += 1

        scores = {}
        for ch, b in row.bubbles.items():
            sc = sample_bubble_fill(binary, (b.x, b.y), b.r, inset=1.0)
            scores[ch] = sc

        chres = choose_answer(scores, abs_thresh, margin)

        # left-of-question indicator position
        col_left = float(col_defs[row.col]["left"]) if row.col < len(col_defs) else 0.0
        qnum_x = col_left + 70.0
        icon_x = qnum_x - 40.0
        icon_y = row.y + 12.0

        if chres.status == "OK":
            chosen_map[row.question] = chres.chosen
            is_ok = (chres.chosen in key_multi.get(row.question, set()))
            if is_ok:
                correct += 1
            mark_icons.append((icon_x, icon_y, bool(is_ok), "OK"))
        elif chres.status == "BLANK":
            chosen_map[row.question] = "BLANK"
            blank_count += 1
            needs_manual = needs_manual or False  # not manual, but still incorrect
            mark_icons.append((icon_x, icon_y, False, "BLANK"))
            status_texts.append((qnum_x, row.y - 2.0, "BLANK"))
        else:  # MULTI
            chosen_map[row.question] = f"MULTI({','.join(chres.winners)})"
            multi_count += 1
            needs_manual = True
            mark_icons.append((icon_x, icon_y, False, "MULTI"))
            status_texts.append((qnum_x, row.y - 2.0, "MULTI"))

        # square around the key-correct bubble(s)
        answers = key_multi.get(row.question, set())
        for _k in sorted(list(answers)):
            if _k in row.bubbles:
                b = row.bubbles[_k]
                s = b.r * 1.25
                correct_boxes.append((b.x - s, b.y - s, b.x + s, b.y + s))

    score_points = correct * points_per_question
    percent = (100.0 * correct / total_keyed) if total_keyed else 0.0

    # Build output filename (with form and email local part)
    base = output_base_label or os.path.splitext(os.path.basename(pdf_path))[0]
    outdir_eff = outdir or os.path.dirname(os.path.abspath(pdf_path))
    os.makedirs(outdir_eff, exist_ok=True)
    out_pdf = ""
    if write_graded_pdf and key_map:
        prefix = "MULTI_" if multi_count > 0 else ""
        outbase = f"{prefix}{int(score_points)}pts_{base}_p{page_index+1:03d}.pdf"
        out_pdf = os.path.join(outdir_eff, outbase)
        score_text = f"Form {form_letter} – {correct}/{len(key_multi)} correct (blank={blank_count}, multi={multi_count})"
        out_pdf = write_designspace_graded_pdf(
            warped_bgr, dpi,
            mark_icons_design=mark_icons,
            correct_boxes_design=correct_boxes,
            status_texts=status_texts,
            score_text=score_text,
            out_base_from=out_pdf,
            jpeg_quality=jpeg_quality,
            grayscale=grayscale
        )

    # Per-page CSV (optional, trimmed to keyed questions)
    if perpage_csv_path:
        import csv as _csv
        with open(perpage_csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["Form", form_letter])
            w.writerow(["Question", "Choice/Status", "A", "B", "C", "D"])
            for q in sorted(key_map.keys()):
                st = chosen_map.get(q, "")
                row_bubble = next((r for r in rows if r.question == q), None)
                A = B = C = D = 0.0
                if row_bubble is not None:
                    # recompute shown scores for clarity
                    scA = sample_bubble_fill(binary, (row_bubble.bubbles["A"].x, row_bubble.bubbles["A"].y), row_bubble.bubbles["A"].r, 1.0)
                    scB = sample_bubble_fill(binary, (row_bubble.bubbles["B"].x, row_bubble.bubbles["B"].y), row_bubble.bubbles["B"].r, 1.0)
                    scC = sample_bubble_fill(binary, (row_bubble.bubbles["C"].x, row_bubble.bubbles["C"].y), row_bubble.bubbles["C"].r, 1.0)
                    scD = sample_bubble_fill(binary, (row_bubble.bubbles["D"].x, row_bubble.bubbles["D"].y), row_bubble.bubbles["D"].r, 1.0)
                    A,B,C,D = scA,scB,scC,scD
                w.writerow([q, st, f"{A:.4f}", f"{B:.4f}", f"{C:.4f}", f"{D:.4f}"])

    print(f"[Alpha1.20] page={page_index+1} form={form_letter} email={email_loc} "
          f"correct={correct}/{total_keyed} blanks={blank_count} multi={multi_count} "
          f"manual={needs_manual} out={os.path.basename(out_pdf) if out_pdf else '-'}")

    return out_pdf, correct, total_keyed, chosen_map, form_letter, email_loc, blank_count, multi_count, needs_manual

# ----------------------------- INPUT HELPERS ----------------------------------

def resolve_input_pdf(input_path: str, outdir_hint: str = "") -> Tuple[str, Optional[str], str]:
    """
    Accept either a single PDF path or a directory containing PDFs.

    Returns:
        (pdf_to_grade, temp_pdf_to_cleanup_or_None, output_base_label)
    """
    if os.path.isfile(input_path):
        if input_path.lower().endswith('.pdf'):
            return input_path, None, os.path.splitext(os.path.basename(input_path))[0]
        raise RuntimeError(f"Input file is not a PDF: {input_path}")

    if not os.path.isdir(input_path):
        raise RuntimeError(f"Input path does not exist: {input_path}")

    pdf_files = sorted(glob.glob(os.path.join(input_path, '*.pdf')))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in directory: {input_path}")

    merge_dir = outdir_hint or input_path
    os.makedirs(merge_dir, exist_ok=True)
    tmpf = tempfile.NamedTemporaryFile(prefix='autograder_alpha_v1_20_merged_', suffix='.pdf', delete=False, dir=merge_dir)
    tmpf.close()

    merger = PdfMerger()
    try:
        for pdf in pdf_files:
            merger.append(pdf)
        with open(tmpf.name, 'wb') as f:
            merger.write(f)
    finally:
        merger.close()

    label = os.path.basename(os.path.normpath(input_path)) or 'merged_input'
    print(f"[input] merged {len(pdf_files)} PDF file(s) from directory: {input_path}")
    print(f"[input] temporary merged PDF: {tmpf.name}")
    return tmpf.name, tmpf.name, label

# --------------------------------- MAIN ---------------------------------------

def main():
    ap = argparse.ArgumentParser("OMR (multi-page, design-aligned export) — Alpha1.20")
    ap.add_argument("pdf"); ap.add_argument("spec")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--page", type=int, default=0, help="Process a single page (0- or 1-based). Ignored if --all-pages.")
    ap.add_argument("--all-pages", action="store_true", help="Process every page in the input PDF.")
    ap.add_argument("--bubble-inset", type=float, default=1.0)
    ap.add_argument("--abs-thresh", type=float, default=0.25,
                    help="Absolute fill threshold to accept a bubble as filled (0..1).")
    ap.add_argument("--margin", type=float, default=0.08,
                    help="Required margin (top - second-best) to avoid ambiguous marks.")
    ap.add_argument("--key", default="", help="Key file with sections [A]..[F]; lines '01<TAB>A'")
    ap.add_argument("--csv", default="", help="Write per-page CSV (suffix _p### added to filename) — trimmed to keyed questions")
    ap.add_argument("--summary-csv", default="", help="Write one CSV summarizing all pages (choice + correctness columns + form, plus email/flags)")
    ap.add_argument("--write-graded-pdf", action="store_true")
    ap.add_argument("--debug-dir", default="")
    ap.add_argument("--outdir", default="", help="Directory to write output PDFs/CSVs")
    ap.add_argument("--jpeg-quality", type=int, default=85,
                    help="JPEG quality (1–100) for background image in graded PDF; lower = smaller file.")
    ap.add_argument("--grayscale", action="store_true",
                    help="Embed background as grayscale (smaller file). Overlays remain colored.")
    ap.add_argument("--skip-n", type=int, default=0,
                    help="After grading one page, skip N pages before grading the next (default 0).")
    ap.add_argument("--auto-skip-unreadable", action="store_true",
                    help="If a page cannot be graded (for example, no fiducials found), skip it, continue grading, and append the skipped page to the previous graded output PDF.")
    ap.add_argument("--points-per-question", type=float, default=1.0, help="Points per question")
    ap.add_argument("--force-form", default="",
                    help="Optional: force the form letter (A–F), bypassing detection.")
    args = ap.parse_args()

    original_input_path = args.pdf
    merge_outdir_hint = args.outdir or (original_input_path if os.path.isdir(original_input_path) else os.path.dirname(os.path.abspath(original_input_path)))
    resolved_pdf_path, temp_merged_pdf, output_base_label = resolve_input_pdf(original_input_path, merge_outdir_hint)
    args.pdf = resolved_pdf_path

    spec, rows, (design_w, design_h), fiducials, col_defs, form_bubbles, form_choices, email_roi = load_spec(args.spec)

    # parse multi-form key file
    forms_key_multi = parse_key_file_multi_form(args.key)
    if not forms_key_multi:
        print("[warn] No keys loaded; grading will still run but score will be 0 and no ✓/✗ drawn.")

    # choose pages (respect --all-pages and --skip-n behavior)
    total_pages = get_pdf_page_count(args.pdf)
    pages = list(range(total_pages)) if args.all_pages else [_normalize_page_index(args.page, total_pages)]

    outdir_eff = args.outdir or (original_input_path if os.path.isdir(original_input_path) else os.path.dirname(os.path.abspath(args.pdf)))
    os.makedirs(outdir_eff, exist_ok=True)

    # summary
    summary_rows = []  # (page_index, form, correct, total_keyed, choices dict, email, blanks, multi, needs_manual, filename)

    # --- Grade 1, skip N pages, repeat ---
    skip_n = max(0, int(args.skip_n))
    auto_skip_unreadable = bool(args.auto_skip_unreadable)
    i = 0
    pending_skipped_pdf_paths = []
    last_successful_out_pdf = ""

    while i < total_pages:
        per_csv = ""
        if args.csv:
            base = os.path.splitext(os.path.basename(args.csv))[0]
            per_csv = os.path.join(outdir_eff, f"{base}_p{i+1:03d}.csv")

        dbg_dir = os.path.join(args.debug_dir, f"p{i+1:03d}") if args.debug_dir else ""

        try:
            out_pdf, correct, total_keyed, chosen, form_letter, email_loc, blank_count, multi_count, needs_manual = process_one_page(
                args.pdf, args.dpi, i,
                spec, rows, (design_w, design_h), fiducials, col_defs,
                forms_key_multi,
                args.write_graded_pdf,
                outdir_eff,
                per_csv if args.csv else None,
                dbg_dir if args.debug_dir else None,
                args.abs_thresh, args.margin,
                args.jpeg_quality,
                args.grayscale,
                form_bubbles=form_bubbles,
                email_roi=email_roi,
                forced_form_letter=(args.force_form or None), points_per_question=args.points_per_question,
                output_base_label=output_base_label)
        except Exception as e:
            if not auto_skip_unreadable:
                raise

            print(f"[auto-skip] page {i+1}/{total_pages} could not be graded; skipping. Reason: {e}")
            raw_skip_pdf = _render_raw_page_as_temp_pdf(args.pdf, args.dpi, i)
            if raw_skip_pdf:
                if last_successful_out_pdf:
                    _append_skipped_to_graded_pdf(last_successful_out_pdf, [raw_skip_pdf])
                    print(f"[auto-skip] appended skipped page {i+1} to {os.path.basename(last_successful_out_pdf)}")
                else:
                    pending_skipped_pdf_paths.append(raw_skip_pdf)
                    print(f"[auto-skip] queued skipped page {i+1} until a graded output exists")
            else:
                print(f"[auto-skip] warning: failed to render skipped page {i+1} for bundling")
            i += 1
            continue

        summary_rows.append((i+1, form_letter, correct, total_keyed, chosen, email_loc, blank_count, multi_count, bool(needs_manual), os.path.basename(out_pdf) if out_pdf else ""))

        if out_pdf:
            if pending_skipped_pdf_paths:
                _append_skipped_to_graded_pdf(out_pdf, pending_skipped_pdf_paths)
                print(f"[auto-skip] attached {len(pending_skipped_pdf_paths)} queued skipped page(s) to {os.path.basename(out_pdf)}")
                pending_skipped_pdf_paths = []

            end_skip = min(total_pages, i + 1 + skip_n)
            skipped_count = max(0, end_skip - (i + 1))
            if skipped_count > 0:
                skipped_pdf_paths = _render_raw_pages_as_temp_pdfs(args.pdf, args.dpi, i + 1, end_skip)
                _append_skipped_to_graded_pdf(out_pdf, skipped_pdf_paths)
            print(f"[graded-pdf] page {i+1}/{total_pages}: {out_pdf} (+{skipped_count} skipped)")
            last_successful_out_pdf = out_pdf
        elif pending_skipped_pdf_paths:
            # Keep waiting for the first actual graded PDF if graded-PDF writing is disabled or unavailable.
            print(f"[auto-skip] note: {len(pending_skipped_pdf_paths)} skipped page(s) remain queued because no graded PDF has been written yet")

        i += (skip_n + 1)

    if pending_skipped_pdf_paths:
        print(f"[auto-skip] warning: {len(pending_skipped_pdf_paths)} skipped page(s) could not be attached because no graded PDF was produced before them")
    # write summary CSV (choice + correctness columns) — ONLY for questions that appear in the selected key set(s)
    if args.summary_csv:
        import csv as _csv
        path = os.path.join(outdir_eff, args.summary_csv)

        # Determine the union of all question numbers across all forms (kept small in practice)
        all_qs = sorted({q for fm in forms_key_multi.values() for q in fm.keys()}) if forms_key_multi else []

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            header = ["Page", "Form", "Email", "Correct", "Total", "Blanks", "Multi", "NeedsManual", "OutFile"]
            for q in all_qs:
                header.append(f"Q{q:02d}")
                header.append(f"A{q:02d}")
            w.writerow(header)
            for (pidx, form_letter, corr, tot, choices, email_loc, blanks, multi, manual, outfile) in summary_rows:
                row = [pidx, form_letter, email_loc if email_loc != "unknown_email" else "", corr, tot, blanks, multi, int(manual), outfile]
                for q in all_qs:
                    choice = choices.get(q, "")
                    row.append(choice or "")
                    # per-form correctness check
                    is_correct = 0
                    if forms_key_multi and form_letter in forms_key_multi and q in forms_key_multi[form_letter]:
                        is_correct = 1 if (choice and (choice in forms_key_multi[form_letter][q])) else 0
                    row.append(is_correct)
                w.writerow(row)
        print(f"[summary] wrote {path}")

if __name__ == "__main__":
    main()
