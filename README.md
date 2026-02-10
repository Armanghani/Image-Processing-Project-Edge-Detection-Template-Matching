# Image-Processing-Project-Edge-Detection-Template-Matching
This repository presents the implementation of classical image processing algorithms developed as part of the Image Processing course taught by Dr. Yousefpour. The project includes Sobel and Prewitt edge detection as well as character detection using normalized cross-correlation, implemented using basic Python operations.
# Image Processing Project – Edge Detection & Character Recognition

## 📌 Course Information
- **Course**: Image Processing  
- **Instructor**: Dr. Yousefpour  
- **Project**: Project 1  

This repository contains the implementation of Project 1 for the Image Processing course.  
The project is divided into two main tasks:

1. **Task 1**: Edge Detection using Sobel and Prewitt operators  
2. **Task 2**: Character Detection using Normalized Cross-Correlation (NCC)

All algorithms are implemented from scratch using basic Python operations, without relying on high-level image processing functions.

---

## 🧩 Task 1: Edge Detection

### 🎯 Objective
The goal of Task 1 is to detect edges in a grayscale image using classical gradient-based operators.  
Edges represent regions of strong intensity change and are essential for understanding image structure.

---

### 🔍 Edge Detection Operators

#### Sobel Operator
The Sobel operator emphasizes edges by combining smoothing and differentiation.  
It uses two kernels:
- Horizontal kernel (detects vertical edges)
- Vertical kernel (detects horizontal edges)

#### Prewitt Operator
The Prewitt operator is similar to Sobel but uses uniform weights.  
It also consists of horizontal and vertical kernels.

---

### 🧠 Methodology

1. The input image is converted to grayscale.
2. Zero-padding is applied to preserve image size.
3. Convolution is performed between the image and each kernel.
4. Edge responses in the x and y directions are computed.
5. Edge magnitude is calculated using:
   
   \[
   \text{Edge Magnitude} = \sqrt{G_x^2 + G_y^2}
   \]

6. The result is normalized to the range [0, 1] for visualization.

---

### 📤 Outputs (Task 1)

For each operator (Sobel / Prewitt), the following images are generated:
- Edge response in x-direction
- Edge response in y-direction
- Edge magnitude image

All results are saved in the `results/` directory.

---

## 🧩 Task 2: Character Detection using NCC

### 🎯 Objective
The goal of Task 2 is to detect occurrences of a given character (template image) inside a larger image.

This task uses **Normalized Cross-Correlation (NCC)** to measure similarity between the template and regions of the image.

---

### 🧠 Normalized Cross-Correlation (NCC)

Normalized Cross-Correlation is defined as:

\[
NCC = \frac{\sum (T \cdot P)}{\|T\| \cdot \|P\|}
\]

Where:
- `T` is the zero-mean template
- `P` is the zero-mean image patch
- `|| ||` denotes the Euclidean norm

The NCC value lies in the range **[-1, 1]**, where higher values indicate higher similarity.

---

### 🔍 Detection Procedure

1. Compute the mean intensity of the template and subtract it (zero-mean template).
2. Slide the template over the image using a sliding window approach.
3. For each window:
   - Subtract the mean of the window (zero-mean patch).
   - Compute the NCC score.
4. If the NCC score exceeds a predefined threshold (0.78), the location is considered a valid detection.
5. Store all detected coordinates.

---

### 📤 Outputs (Task 2)

The detection results are saved in a JSON file with the following structure:

```json
{
  "coordinates": [[x1, y1], [x2, y2], ...],
  "templat_size": [height, width]
}
Project-1/
├── task1.py
├── task2.py
├── utils.py
├── data/
│   ├── a.pgm
│   ├── b.pgm
│   ├── c.pgm
│   └── bar.jpg
└── results/
```

###▶️ How to Run
Task 1 – Edge Detection
python task1.py --img_path data/bar.jpg --kernel sobel
python task1.py --img_path data/bar.jpg --kernel prewitt

Task 2 – Character Detection
python task2.py --img_path data/bar.jpg --template_path data/c.pgm



###✅ Notes
No built-in correlation or convolution functions are used.

All computations are performed using nested lists and basic Python logic.

The project emphasizes understanding fundamental image processing concepts rather than using black-box libraries.

###🏁 Conclusion
This project demonstrates classical image processing techniques for edge detection and template matching.
By separating detection from evaluation, the implementation remains modular, interpretable, and suitable for academic analysis.
