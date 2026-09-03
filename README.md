# Facial wrinkle ratio (%) calculation
Calculate the wrinkle ratio (%) for 12 facial sub-regions from the face image and the corresponding wrinkle image.
<img width="332" height="638" alt="Image" src="https://github.com/user-attachments/assets/795a4fb1-8299-48e6-80a1-57e1ee3c11ee" />


# Usage
Run <kbd> face2age_20260826_wrinkle_region_analysis_spline_openVer_en_1.py</kbd>.

Input: Face image, corresponding wrinkle image

Output: Face wrinkle ratio (%) file (.csv)


# Facial landmarks detection



# Facial sub-regions segmentation
Twelve regions prone to wrinkle formation were extracted using these facial landmarks. 



# Facial wrinkle ratio calculation
The facial wrinkle ratio for each sub-region was calculated by dividing the number of wrinkle pixels within that sub-region by the total number of pixels in the whole facial region, then multiplying by 100 to express the result as a percentage.
<img width="400" height="253" alt="The_whole-face_region_and_12_sub-regions_small" src="https://github.com/user-attachments/assets/7adb1870-8d43-4c9f-902e-7c3f0f845ebb" />


The facial wrinkle ratio results are output as a CSV file.
