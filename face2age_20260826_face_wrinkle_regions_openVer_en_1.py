# -*- coding: utf-8 -*-
"""
face_wrinkle_regions_12.py

MediaPipe FaceMesh 468 landmark indices and triangle (mesh) connectivity
information for the 12 clinical wrinkle regions (botox/filler treatment
zones).

Region list:
  Forehead lines, Frown lines (Glabellar), Bunny lines,
  Crow's feet (L/R), Tear Trough (L/R), Nasolabial Fold (L/R),
  Marionette line (L/R), Mental Crease

[Source / how this was generated]
- Coordinates: the official mediapipe GitHub canonical_face_model.obj
  (468 landmark 3D reference coordinates; the v (vertex) order corresponds
  exactly to landmark index 0~467)
- Triangles: the f (face) lines from the same obj file (the official
  898-triangle tessellation)
- Only eye (eyelid) / lips landmarks are excluded (FACEMESH_LEFT_EYE,
  RIGHT_EYE, LIPS). Eyebrow landmarks are kept, since they help form a
  natural boundary between the forehead / glabella / eye-area regions.

[Region classification criteria (canonical coordinates, y = up is +)]
  Forehead      : y > 5.109 (eyebrow line, average of landmarks 105/334)
  Frown         : 3.271 <= y <= 5.109  and |x| <= 1.5
  Bunny         : 1.0   <= y <= 3.271  and 1.5 < |x| <= 2.3
  CrowsFeet_L/R : 2.585 <= y <= 5.109  and |x| > 4.05
  TearTrough_L/R: -0.5  <= y <= 2.585  and 2.3 < |x| <= 4.05
  Nasolabial_L/R: -5.365<= y <= -0.5   and |x| > 1.886
  MentalCrease  : -6.965<= y <= -5.365 and |x| <= 1.8
  Marionette_L/R: -9.403<= y <= -5.365 and |x| > 1.8

NOTE: These boundary values are not official coordinates published by
  mediapipe or any dermatology paper - they are my own approximation of
  the anatomical location of each clinical term (forehead lines / frown
  (glabellar) lines / bunny lines / crow's feet / tear trough / nasolabial
  fold / marionette line / mental crease) on the canonical 3D model.
  Before using this in practice, please check it roughly against
  region_schematic_v3.png, and re-validate by overlaying it on an actual
  face photo to see if it looks clinically reasonable. If needed, just
  adjust the numbers above and recompute.
"""

# Full landmark index list per region (includes both boundary and interior points)
REGION_LANDMARKS = {
    "Forehead": [10, 21, 54, 67, 68, 69, 103, 104, 105, 108, 109, 151, 251, 284, 297, 298, 299, 332, 333, 334, 337, 338],
    "Frown": [8, 9, 55, 107, 168, 285, 336],
    "Bunny": [26, 56, 112, 121, 190, 232, 233, 243, 256, 286, 341, 350, 414, 452, 453, 463],
    "CrowsFeet_L": [260, 276, 283, 293, 300, 301, 342, 353, 359, 368, 383, 389, 444, 445, 467],
    "CrowsFeet_R": [30, 46, 53, 63, 70, 71, 113, 124, 130, 139, 156, 162, 224, 225, 247],
    "TearTrough_L": [252, 253, 254, 329, 330, 348, 349, 450, 451],
    "TearTrough_R": [22, 23, 24, 100, 101, 119, 120, 230, 231],
    "Nasolabial_L": [266, 273, 280, 287, 288, 292, 306, 307, 322, 323, 325, 352, 361, 366, 367, 376, 401, 407, 408, 410, 411, 416, 423, 425, 426, 427, 432, 433, 434, 435, 436],
    "Nasolabial_R": [36, 43, 50, 57, 58, 62, 76, 77, 92, 93, 96, 123, 132, 137, 138, 147, 177, 183, 184, 186, 187, 192, 203, 205, 206, 207, 212, 213, 214, 215, 216],
    "Marionette_L": [262, 335, 364, 365, 369, 378, 379, 394, 395, 397, 400, 418, 422, 424, 430, 431],
    "Marionette_R": [32, 106, 135, 136, 140, 149, 150, 169, 170, 172, 176, 194, 202, 204, 210, 211],
    "MentalCrease": [18, 83, 182, 201, 313, 406, 421],
}

REGION_BOUNDARY = {
    "Forehead": [251, 334, 296, 336, 151, 107, 66, 105, 21, 54, 469, 470, 471, 472, 473, 474, 475, 284, 251],
    "Frown": [285, 8, 55, 151, 285],
    "Bunny": [8, 413, 464, 412, 188, 244, 189, 8],
    "CrowsFeet_L": [359, 261, 346, 352, 264, 368, 301, 300, 276, 359],
    "CrowsFeet_R": [130,  31, 117, 123,  34, 139,  71,  70,  46, 130],
    "TearTrough_L": [463, 464, 357, 277, 329, 330, 346, 261, 359, 255, 339, 254, 253, 252, 256, 341],
    "TearTrough_R": [243, 244, 128,  47, 100, 101, 117,  31, 130,  25, 110,  24,  23,  22,  26, 112],
    "Nasolabial_L": [322, 410, 287, 422, 434, 427, 425, 266, 371, 355, 420, 360, 344, 455],
    "Nasolabial_R": [ 92, 186,  57, 202, 214, 207, 205,  36, 142, 126, 198, 131, 115, 235],
    "Marionette_L": [291, 375, 335, 431, 430, 422, 287, 291],
    "Marionette_R": [ 61, 146, 106, 211, 210, 202,  57,  61],
    "MentalCrease": [ 17, 314, 424, 262, 428, 199, 208, 32, 204, 84, 17],
}

REGION_BOUNDARY1 = {
    "Forehead": [251, 334, 296, 336, 151, 107, 66, 105, 21, 54, 468, 469, 470, 471, 472, 473, 474, 284, 251],
    "Frown": [285, 8, 55, 151, 285],
    "Bunny": [8, 413, 464, 412, 188, 244, 189, 8],
    "CrowsFeet_L": [359, 261, 346, 352, 264, 368, 301, 300, 276, 359],
    "CrowsFeet_R": [130,  31, 117, 123,  34, 139,  71,  70,  46, 130],
    "TearTrough_L": [463, 464, 357, 277, 329, 330, 346, 261, 359, 255, 339, 254, 253, 252, 256, 341],
    "TearTrough_R": [243, 244, 128,  47, 100, 101, 117,  31, 130,  25, 110,  24,  23,  22,  26, 112],
    "Nasolabial_L": [322, 410, 287, 422, 434, 427, 425, 266, 371, 355, 420, 360, 344, 455],
    "Nasolabial_R": [ 92, 186,  57, 202, 214, 207, 205,  36, 142, 126, 198, 131, 115, 235],
    "Marionette_L": [291, 375, 335, 431, 430, 422, 287, 291],
    "Marionette_R": [ 61, 146, 106, 211, 210, 202,  57,  61],
    "MentalCrease": [ 17, 314, 424, 262, 428, 199, 208, 32, 204, 84, 17],
}

# Interior triangle (mesh) list per region -
# (landmark_idx_a, landmark_idx_b, landmark_idx_c)
# Only includes official mediapipe triangles whose all 3 vertices belong to
# that region.
REGION_TRIANGLES = {
    "Forehead": [[299, 333, 297], [69, 67, 104], [332, 297, 333], [103, 104, 67], [333, 298, 332], [104, 103, 68], [284, 332, 298], [54, 68, 103], [337, 299, 338], [108, 109, 69], [297, 338, 299], [67, 69, 109], [333, 299, 334], [104, 105, 69], [151, 337, 10], [151, 10, 108], [338, 10, 337], [109, 108, 10]],
    "Frown": [[8, 285, 9], [8, 9, 55], [336, 9, 285], [107, 55, 9]],
    "Bunny": [[256, 341, 452], [26, 232, 112], [453, 452, 341], [233, 112, 232], [341, 463, 453], [112, 233, 243]],
    "CrowsFeet_L": [[467, 359, 342], [276, 353, 300], [383, 300, 353], [283, 276, 293], [300, 293, 276], [445, 342, 276], [353, 276, 342], [444, 445, 283], [276, 283, 445], [300, 383, 301], [368, 301, 383], [445, 444, 260], [260, 467, 445], [342, 445, 467]],
    "CrowsFeet_R": [[247, 113, 130], [46, 70, 124], [156, 124, 70], [53, 63, 46], [70, 46, 63], [225, 46, 113], [124, 113, 46], [224, 53, 225], [46, 225, 53], [70, 71, 156], [139, 156, 71], [225, 30, 224], [30, 225, 247], [113, 247, 225]],
    "TearTrough_L": [[348, 450, 349], [451, 349, 450], [330, 348, 329], [349, 329, 348], [253, 450, 254], [450, 253, 451], [252, 451, 253]],
    "TearTrough_R": [[119, 120, 230], [231, 230, 120], [101, 100, 119], [120, 119, 100], [23, 24, 230], [230, 231, 23], [22, 23, 231]],
    "Nasolabial_L": [[280, 425, 411], [427, 411, 425], [425, 266, 426], [423, 426, 266], [425, 426, 427], [436, 427, 426], [287, 273, 432], [306, 292, 307], [325, 307, 292], [427, 436, 434], [432, 434, 436], [280, 411, 352], [376, 352, 411], [426, 322, 436], [410, 436, 322], [366, 401, 323], [361, 323, 401], [408, 407, 306], [292, 306, 407], [436, 410, 432], [287, 432, 410], [434, 416, 427], [411, 427, 416], [352, 376, 366], [401, 366, 376], [367, 435, 416], [433, 416, 435], [376, 433, 401], [435, 401, 433], [411, 416, 376], [433, 376, 416], [361, 401, 288], [435, 288, 401]],
    "Nasolabial_R": [[50, 187, 205], [207, 205, 187], [205, 206, 36], [203, 36, 206], [205, 207, 206], [216, 206, 207], [57, 212, 43], [76, 77, 62], [96, 62, 77], [207, 214, 216], [212, 216, 214], [50, 123, 187], [147, 187, 123], [206, 216, 92], [186, 92, 216], [137, 93, 177], [132, 177, 93], [184, 76, 183], [62, 183, 76], [216, 212, 186], [57, 186, 212], [214, 207, 192], [187, 192, 207], [123, 137, 147], [177, 147, 137], [138, 192, 215], [213, 215, 192], [147, 177, 213], [215, 213, 177], [187, 147, 192], [213, 192, 147], [132, 58, 177], [215, 177, 58]],
    "Marionette_L": [[262, 431, 418], [424, 418, 431], [365, 364, 379], [394, 379, 364], [431, 262, 395], [369, 395, 262], [430, 422, 431], [424, 431, 422], [424, 422, 335], [394, 430, 395], [431, 395, 430], [395, 369, 378], [400, 378, 369], [379, 394, 378], [395, 378, 394], [394, 364, 430]],
    "Marionette_R": [[32, 194, 211], [204, 211, 194], [136, 150, 135], [169, 135, 150], [211, 170, 32], [140, 32, 170], [210, 211, 202], [204, 202, 211], [204, 106, 202], [169, 170, 210], [211, 210, 170], [170, 149, 140], [176, 140, 149], [150, 149, 169], [170, 169, 149], [169, 210, 135]],
    "MentalCrease": [[313, 421, 406], [83, 182, 201]],
}


def build_region_mask(landmarks_xy, image_shape, region_name):
    """
    Builds the actual pixel mask for a given region as the union of its
    interior triangles.

    landmarks_xy : (468, 2) ndarray, pixel coordinates from mediapipe FaceMesh
    image_shape  : (h, w, ...) shape of the original image
    region_name  : one of the keys in REGION_TRIANGLES
    """
    import numpy as np
    import cv2

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for (a, b, c) in REGION_TRIANGLES[region_name]:
        tri = np.array([landmarks_xy[a], landmarks_xy[b], landmarks_xy[c]], dtype=np.int32)
        cv2.fillConvexPoly(mask, tri, 255)
    return mask
