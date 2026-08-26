# -*- coding: utf-8 -*-
"""
wrinkle_region_analysis_spline.py

Takes a face image (24-bit color) and a wrinkle mask (8-bit, wrinkle=255 /
background=0) as input, and for each of the 12 clinical wrinkle regions
(Forehead lines, Frown lines, Bunny lines, Crow's feet_L/R, Tear Trough_L/R,
Nasolabial Fold_L/R, Marionette line_L/R, Mental Crease):

  1) Computes (wrinkle pixel count / total pixel count of that region) * 100 (%)
     as wrinkle_percent, and (wrinkle pixel count / total pixel count of the
     whole face region) * 100 (%) as wrinkle_percent_whole, and saves them to
     a CSV file
  2) Saves an image with each region overlaid in a different color on top of
     the original color image

Difference from v2 (triangle mesh method)
------------------------------------------
v2 built region boundaries as "straight-edged polygons" (landmarks joined by
straight lines) or as the union of interior triangles. This version instead
takes REGION_BOUNDARY (the ordered boundary landmarks per region) from
face_wrinkle_regions_12.py and smoothly connects them into a "curved outline"
using scipy's periodic B-spline (closed periodic spline), then fills the
inside of that outline to create the mask. Since real skin wrinkle regions
are rounded curves rather than angular polygons, this approach produces a
more clinically natural-looking boundary.

Note: a closed spline can slightly overshoot outward in places where control
      points are sparse, which can cause a small overlap (usually under 1-2%
      of the total) with neighboring regions. If your research use case
      requires strictly normalized region areas, use v2's triangle-mesh mask
      instead (build_region_mask, which guarantees zero overlap); this script
      is better suited for visualization / presentation purposes.

Usage example:
    python wrinkle_region_analysis_spline.py --face face.jpg --wrinkle wrinkle_mask.png \
        --out_csv result.csv --out_overlay overlay.png
"""

import csv

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.interpolate import splprep, splev

from face2age_20260826_face_wrinkle_regions_openVer_en_1 import REGION_BOUNDARY1
from os import path

mp_face_mesh = mp.solutions.face_mesh

REGION_ORDER = [
    "Forehead", "Frown", "Bunny",
    "CrowsFeet_L", "CrowsFeet_R",
    "TearTrough_L", "TearTrough_R",
    "Nasolabial_L", "Nasolabial_R",
    "Marionette_L", "Marionette_R",
    "MentalCrease",
]

REGION_COLORS = {
    "Forehead": (66, 133, 244),
    "Frown": (55, 68, 219),
    "Bunny": (255, 50, 255),
    "CrowsFeet_L": (244, 180, 0),
    "CrowsFeet_R": (244, 180, 0),
    "TearTrough_L": (0, 200, 150),
    "TearTrough_R": (0, 200, 150),
    "Nasolabial_L": (88, 157, 15),
    "Nasolabial_R": (88, 157, 15),
    "Marionette_L": (188, 71, 171),
    "Marionette_R": (188, 71, 171),
    "MentalCrease": (128, 128, 128),
}

# Landmark index list defining the whole face region (contour) - MediaPipe's
# official FACEMESH_FACE_OVAL
FACE_OVAL_IDX = [
    472, 473, 474, 475, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 469, 470, 471,
]


def build_face_mask(landmarks_xy: np.ndarray, shape) -> np.ndarray:
    """
    Builds the whole-face-region mask as the convex hull of the face contour
    (FACE_OVAL_IDX) landmarks.
    FACE_OVAL_IDX already includes the extended forehead landmarks (469~475),
    so landmarks_xy must be the (476, 2) array produced by extend_landmarks().
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    oval_pts = landmarks_xy[FACE_OVAL_IDX].astype(np.int32)
    hull = cv2.convexHull(oval_pts)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


# ---------------------------------------------------------------------------
# Extract landmarks
# ---------------------------------------------------------------------------
def get_landmarks(image_bgr: np.ndarray) -> np.ndarray:
    """Returns the pixel coordinates (x, y) of the 468 landmarks found in the image"""
    h, w = image_bgr.shape[:2]
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as fm:
        results = fm.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise ValueError("Could not detect a face. Please check the input image.")
        lm = results.multi_face_landmarks[0].landmark
        pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    return pts


# ---------------------------------------------------------------------------
# 1) Add virtual landmarks (469~475) to extend the forehead region
# ---------------------------------------------------------------------------
# (new_index, base_index, opposite_index): new point = base point + (base point - opposite point)
# In other words, we take the vector from the opposite point to the base
# point and add it to the base point once more, pushing the base point
# further out in that direction. (This extends the existing upper-forehead
# boundary landmarks slightly further outward/upward, to alleviate the
# problem of the forehead region being detected too narrowly.)
EXTRA_LANDMARK_DEFS = [
    (469, 103, 104),
    (470, 67, 69),
    (471, 109, 108),
    (472, 10, 151),
    (473, 338, 337),
    (474, 297, 299),
    (475, 332, 333),
]

def extend_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    Takes the (468, 2) landmark array returned by get_landmarks() and returns
    a (476, 2) array with 7 additional virtual landmarks (469~475) appended.

    469 = 103 + (103 - 104)
    470 = 67  + (67  - 69)
    471 = 109 + (109 - 108)
    472 = 10  + (10  - 151)
    473 = 338 + (338 - 337)
    474 = 297 + (297 - 299)
    475 = 332 + (332 - 333)

    Indices 0~467 of the returned array are identical to the original
    mediapipe landmarks, index 468 is an unused slot (filled with the
    original landmark 10's coordinates), and indices 469~475 are the newly
    computed extension landmarks.
    """
    pts = np.asarray(pts, dtype=np.float32)
    extended = np.zeros((476, 2), dtype=np.float32)
    extended[:468] = pts
    extended[468] = pts[10]  # unused slot (placeholder to keep indices aligned)

    for new_idx, base_idx, other_idx in EXTRA_LANDMARK_DEFS:
        extended[new_idx] = pts[base_idx] + (pts[base_idx] - pts[other_idx]) * 1.5

    return extended

EXTRA_LANDMARK_DEFS1 = [
    (468, 103, 104),
    (469, 67, 69),
    (470, 109, 108),
    (471, 10, 151),
    (472, 338, 337),
    (473, 297, 299),
    (474, 332, 333),
]

def extend_landmarks1(pts: np.ndarray) -> np.ndarray:
    """
    Takes the (468, 2) landmark array returned by get_landmarks() and returns
    a (476, 2) array with 7 additional virtual landmarks (469~475) appended.

    469 = 103 + (103 - 104)
    470 = 67  + (67  - 69)
    471 = 109 + (109 - 108)
    472 = 10  + (10  - 151)
    473 = 338 + (338 - 337)
    474 = 297 + (297 - 299)
    475 = 332 + (332 - 333)

    Indices 0~467 of the returned array are identical to the original
    mediapipe landmarks, index 468 is an unused slot (filled with the
    original landmark 10's coordinates), and indices 469~475 are the newly
    computed extension landmarks.
    """
    pts = np.asarray(pts, dtype=np.float32)
    extended = np.zeros((476, 2), dtype=np.float32)
    extended[:468] = pts
    #extended[468] = pts[10]  # unused slot (placeholder to keep indices aligned)

    for new_idx, base_idx, other_idx in EXTRA_LANDMARK_DEFS1:
        extended[new_idx] = pts[base_idx] + (pts[base_idx] - pts[other_idx]) * 1.5

    return extended

# ---------------------------------------------------------------------------
# 2) Resolve overlaps by absorbing pixels that overlap a given priority region
#    into that region
# ---------------------------------------------------------------------------
# Applied in this order: an item later in the list takes priority over the
# result so far (i.e. if two regions overlap, the region processed last ends
# up owning that pixel).
OVERLAP_PRIORITY = [
    "Frown",
    "TearTrough_L",
    "TearTrough_R",
    "Nasolabial_L",
    "Nasolabial_R",
    "MentalCrease",
]


def resolve_region_overlaps(region_masks: dict) -> dict:
    """
    Wherever a region in OVERLAP_PRIORITY overlaps another region, the
    overlapping pixels are entirely absorbed into that priority region
    (used to clean up overlaps between regions caused by spline overshoot,
    etc).

    Example) If Frown and Bunny overlap -> the overlapping pixels become
             Frown (removed from Bunny)
             If TearTrough_L and CrowsFeet_L overlap -> the overlapping
             pixels become TearTrough_L

    Returns a new dict; the original dict is not modified.
    """
    resolved = {name: mask.copy() for name, mask in region_masks.items()}

    for priority_name in OVERLAP_PRIORITY:
        if priority_name not in resolved:
            continue
        priority_mask = resolved[priority_name]
        inv_priority = cv2.bitwise_not(priority_mask)
        for name in resolved:
            if name == priority_name:
                continue
            resolved[name] = cv2.bitwise_and(resolved[name], inv_priority)

    return resolved


# ---------------------------------------------------------------------------
# Interpolate the boundary landmarks into a closed periodic spline curve
# ---------------------------------------------------------------------------
def spline_curve(points_xy, n_samples: int = 300, smoothing: float = 0.0) -> np.ndarray:
    """
    points_xy : the landmark pixel coordinates that form a region's boundary
                (must already be in order around the boundary)
    n_samples : number of sample points used to approximate the curve as a
                dense polygon
    smoothing : scipy splprep's s parameter. 0 means the curve passes exactly
                through every control point; a value greater than 0 makes it
                slightly blunter but reduces overshoot (bulging outward).
    """
    pts = np.asarray(points_xy, dtype=np.float64)

    # Remove consecutive duplicate coordinates (prevents spline-fitting errors)
    keep = [0]
    for i in range(1, len(pts)):
        if not np.allclose(pts[i], pts[keep[-1]]):
            keep.append(i)
    pts = pts[keep]

    n = len(pts)
    if n < 3:
        # Too few points to build a spline, so return as-is (straight polygon)
        return pts

    k = 3 if n > 3 else n - 1  # cubic spline; lower the degree if there aren't enough points
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing, per=True, k=k)
    unew = np.linspace(0, 1, n_samples)
    xs, ys = splev(unew, tck)
    return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------
# Build a region mask based on the spline curve
# ---------------------------------------------------------------------------
def build_region_mask(landmarks_xy: np.ndarray, shape, region_name: str,
                       n_samples: int = 300, smoothing: float = 0.0) -> np.ndarray:
    h, w = shape[:2]
    idxs = REGION_BOUNDARY1[region_name]
    boundary_pts = landmarks_xy[idxs]
    curve = spline_curve(boundary_pts, n_samples=n_samples, smoothing=smoothing)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [curve.astype(np.int32)], 255)
    return mask


def build_all_region_masks(landmarks_xy: np.ndarray, shape, smoothing: float = 0.0):
    return {
        name: build_region_mask(landmarks_xy, shape, name, smoothing=smoothing)
        for name in REGION_ORDER
    }


# ---------------------------------------------------------------------------
# Sanity check: verify whether regions overlap each other (small overlaps can
# occur due to spline overshoot)
# ---------------------------------------------------------------------------
def sanity_check(region_masks: dict) -> dict:
    union = None
    overlap_pixels = 0
    total = 0
    for m in region_masks.values():
        if union is None:
            union = np.zeros_like(m)
        overlap_pixels += int(np.count_nonzero(cv2.bitwise_and(union, m)))
        union = cv2.bitwise_or(union, m)
        total += int(np.count_nonzero(m))
    return {
        "overlap_pixel_count": overlap_pixels,
        "overlap_ratio_percent": round(overlap_pixels / total * 100, 3) if total else 0.0,
        "total_region_pixel_count": int(np.count_nonzero(union)),
    }


def plot_points_with_index(
    image,
    points,
    point_color=(0, 255, 0),
    text_color=(0, 255, 255),
    radius=None,
    font_scale=None,
    thickness=2,
    offset=(5, -5),
    show=True,
    window_name="Points",
):
    output = image.copy()

    h, w = output.shape[:2]

    # Auto-set based on image size
    if radius is None:
        radius = max(2, int(min(h, w) * 0.004))

    if font_scale is None:
        font_scale = max(0.4, min(h, w) / 1200)

    points = np.asarray(points)

    for idx, pt in enumerate(points):
        x = int(round(pt[0]))
        y = int(round(pt[1]))

        # point
        cv2.circle(
            output,
            (x, y),
            radius,
            point_color,
            -1,
            lineType=cv2.LINE_AA,
        )

        # index
        cv2.putText(
            output,
            str(idx),
            #str(idx+1),
            (x + offset[0], y + offset[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    if show:
        cv2.imshow(window_name, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output

# ---------------------------------------------------------------------------
# Compute per-region wrinkle ratio -> save to CSV
# ---------------------------------------------------------------------------
def compute_wrinkle_ratios(wrinkle_mask_bin: np.ndarray, region_masks: dict, out_csv: str,
                            face_mask: np.ndarray):
    """
    face_mask : the whole-face-region mask (0/255, same shape as region_masks)
                produced by build_face_mask(). Used to compute
                wrinkle_percent_whole and whole_wrinkle_percent_whole.

    Returns: (results, df, whole_wrinkle_percent_whole)
      - results : a list of per-region result dicts (region, region_pixel_count,
                  wrinkle_pixel_count, wrinkle_percent, wrinkle_percent_whole)
      - df      : results transferred as-is into a pandas DataFrame
      - whole_wrinkle_percent_whole : a single wrinkle-ratio(%) value computed
                  over the whole face regardless of region (not included in
                  results/df; returned separately)
    """
    face_whole_region_area = int(np.count_nonzero(face_mask))

    whole_wrinkle_pixel_count = int(np.count_nonzero(wrinkle_mask_bin))

    # Regardless of region, the number of pixels marked as wrinkles over the
    # whole face / the total pixel count of the whole face region
    # whole_wrinkle_pixel_count = int(np.count_nonzero(cv2.bitwise_and(wrinkle_mask_bin, face_mask)))
    whole_wrinkle_percent_whole = (
        round(whole_wrinkle_pixel_count / face_whole_region_area * 100.0, 3)
        if face_whole_region_area > 0 else 0.0
    )

    results = []
    for name in REGION_ORDER:
        rmask = region_masks[name]
        region_area = int(np.count_nonzero(rmask))
        wrinkle_in_region = int(np.count_nonzero(cv2.bitwise_and(wrinkle_mask_bin, rmask)))

        pct = (wrinkle_in_region / region_area * 100.0) if region_area > 0 else 0.0
        pct_whole = (wrinkle_in_region / face_whole_region_area * 100.0) if face_whole_region_area > 0 else 0.0

        results.append(
            {
                "region": name,
                "whole_pixel_count": face_whole_region_area,
                "region_pixel_count": region_area,
                "wrinkle_pixel_count": wrinkle_in_region,
                "wrinkle_percent": round(pct, 3),
                "wrinkle_percent_whole": round(pct_whole, 3),
            }
        )

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["region", "whole_pixel_count", "region_pixel_count", "wrinkle_pixel_count",
                           "wrinkle_percent", "wrinkle_percent_whole"]
        )
        writer.writeheader()
        writer.writerows(results)

    df = pd.DataFrame(results, columns=["region", "whole_pixel_count", "region_pixel_count", "wrinkle_pixel_count",
                                         "wrinkle_percent", "wrinkle_percent_whole"])

    return results, df, whole_wrinkle_percent_whole


# ---------------------------------------------------------------------------
# Save the per-region color overlay image
# ---------------------------------------------------------------------------
def save_overlay(image_bgr: np.ndarray, region_masks: dict, out_path: str, alpha: float = 0.45):
    canvas = image_bgr.copy()
    for name in REGION_ORDER:
        rmask = region_masks[name]
        if np.count_nonzero(rmask) == 0:
            continue
        color_layer = np.full_like(image_bgr, REGION_COLORS[name])
        blended = cv2.addWeighted(image_bgr, 1 - alpha, color_layer, alpha, 0)
        mask_bool = rmask.astype(bool)
        canvas[mask_bool] = blended[mask_bool]

    for name in REGION_ORDER:
        contours, _ = cv2.findContours(region_masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def analyze(face_image_path: str, wrinkle_mask_path: str, out_csv: str, out_overlay: str,
            smoothing: float = 0.0):
    image = cv2.imread(face_image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read the face image: {face_image_path}")

    wrinkle = cv2.imread(wrinkle_mask_path, cv2.IMREAD_GRAYSCALE)
    if wrinkle is None:
        raise FileNotFoundError(f"Could not read the wrinkle mask image: {wrinkle_mask_path}")

    if wrinkle.shape[:2] != image.shape[:2]:
        wrinkle = cv2.resize(
            wrinkle, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    _, wrinkle_bin = cv2.threshold(wrinkle, 127, 255, cv2.THRESH_BINARY)

    pts = get_landmarks(image)
    pts = extend_landmarks1(pts)
    image_on_pts = plot_points_with_index(image, pts, radius=3, font_scale=0.7,thickness=1,offset=(2, -2))
    cv2.imwrite("image_landmarks_overlay.png", image_on_pts)

    region_masks = build_all_region_masks(pts, image.shape, smoothing=smoothing)
    region_masks = resolve_region_overlaps(region_masks)

    face_mask = build_face_mask(pts, image.shape)
    face_whole_region_area = int(np.count_nonzero(face_mask))

    check = sanity_check(region_masks)
    print(f"[Check] Overlap between regions: {check['overlap_pixel_count']} px "
          f"({check['overlap_ratio_percent']}% of total region pixels) "
          f"- a small amount of overlap from spline overshoot is normal.")
    print(f"[Info] Whole face region (oval) pixel count: {face_whole_region_area}")

    results, df, whole_wrinkle_percent_whole = compute_wrinkle_ratios(
        wrinkle_bin, region_masks, out_csv, face_mask
    )
    save_overlay(image, region_masks, out_overlay)

    white_img = np.full(image.shape, 255, dtype=image.dtype)
    save_overlay(white_img, region_masks, 'overlay_on_white.png')

    print(f"\nCSV saved -> {out_csv}")
    print(f"Overlay image saved -> {out_overlay}\n")
    print(f"{'Region':<18}{'RegionPixels':>12}{'WrinklePixels':>12}{'VsRegion(%)':>12}{'VsWhole(%)':>12}")
    for r in results:
        print(f"{r['region']:<18}{r['region_pixel_count']:>12}{r['wrinkle_pixel_count']:>12}"
              f"{r['wrinkle_percent']:>12}{r['wrinkle_percent_whole']:>12}")
    print(f"\n[Overall] Wrinkle ratio over the whole face, regardless of region: {whole_wrinkle_percent_whole}%")

    return results, df, whole_wrinkle_percent_whole


def main():

    img_color_path = './images/'
    img_wrinkle_path = './images/'
    img_face = '00656.png'
    img_wrinkle = '00656_mask.png'

    img_fn = img_color_path + img_face
    img_fn2 = img_wrinkle_path + img_wrinkle

    fnbase = path.basename(img_fn)
    filename_only, file_extension = path.splitext(fnbase)

    out_csv = 'wrinkle_percent_' + filename_only + '.csv'
    img_out_overlay = img_color_path + 'wrinkle_subregions_' + filename_only + '.png'
    smoothing = 0.0
    analyze(img_fn, img_fn2, out_csv, img_out_overlay, smoothing=smoothing)


if __name__ == "__main__":
    main()
