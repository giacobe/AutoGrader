# AutoGrader Alpha v1.20 — README

## 📘 Overview
AutoGrader Alpha v1.20 is a high-precision Optical Mark Recognition (OMR) engine designed to automatically grade multiple-choice exam sheets. It supports multi-form exams (A–F), email extraction, bubble analysis, alignment correction, and graded PDF generation.

Example:

Grade a PDF of dozens of exam forms. Each student submits 2 pages. The first page is the OMR Bubble Sheet form. The 2nd page is the essay to be skipped by the autograder. Sometimes there are additional extra pages in addition to the first essay page. The --auto-skip-unreadable feature addresses this.  Output desired is a csv file of all of the grades, as well as an individual PDF for each student's submission. Each output file is named with the student's score.

To achieve this, execute the following command:

```bash
python .\AutoGrader_Alpha_v1_18.py --dpi 300 --all-pages --summary-csv csv grades.csv --write-graded-pdf --outdir output --jpeg-quality 85 --auto-skip-undreadable --points-per-question 3 --key .\answer_keys.txt .\exam.pdf .\omr_form_25q_v1.16.json
```

This version introduces:
- **auto-skip-unreadable

AutoGrader is ideal for large-scale classroom assessments, high-volume grading, and workflows where accuracy of bubble detection and deskewing is critical.

## ⭐ Key Features
### 🖨 High-accuracy OMR
- Detects A/B/C/D/E/F bubbles with threshold-based scoring.
- Handles BLANK, MULTI, and ambiguous marks.

### 🧭 Automatic alignment
- Deskews scanned pages using fiducials.
- Computes homography → aligns scans to the design template.
- Optional translation-only refinement.

### 📄 Multi-form exam support (A–F)
- Detects exam form via specialized form bubbles.
- Reads multi-form answer keys.

### 📧 Email extraction via OCR
- - Psyche - no it doesn't ... not yet.  Extracts student email from a specified ROI.
- Uses only the email’s local part (before `@`) for filenames.

### 📝 Robust grading output
- Generates graded PDFs with check ✓, X marks, boxes, and status text.
- Produces per-page CSVs and a master summary CSV.

### 📁 Smart file naming
- Score-first filenames (e.g. `60pts_exam_p001.pdf`)
- Auto-prefix `"MULTI_"` when needed.

### 🧪 Debugging support
- Diagnostic overlays for alignment, thresholding, bubble fill intensity, and form detection.

## 🛠 Installation

### 📦 Required Python Packages
Install via pip:

```bash
pip install opencv-python numpy pymupdf pillow PyPDF2 pytesseract
```

### 🔍 Install Tesseract OCR
Required for email OCR:
- **Windows:** `choco install tesseract`
- **macOS:** `brew install tesseract`
- **Ubuntu:** `sudo apt install tesseract-ocr`

### 📄 Optional: using `requirements.txt`
Create:

```
opencv-python
numpy
pymupdf
pillow
PyPDF2
pytesseract
```

Install using:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Grade a single page
```bash
python autograder.py exam.pdf form25q.json --page 1 --key key.txt --write-graded-pdf
```

### Grade all pages
```bash
python autograder.py exam.pdf form25q.json --all-pages --key key.txt --write-graded-pdf
```

### Save summary CSV
```bash
python autograder.py exam.pdf form25q.json --all-pages --key key.txt --summary-csv results.csv
```

### Assign 3 points per question
```bash
python autograder.py exam.pdf form25q.json --all-pages --key key.txt --points-per-question 3
```

### Force form C
```bash
python autograder.py exam.pdf form25q.json --force-form C
```

## ⚙️ Command-Line Options

| Option | Description |
|--------|-------------|
| `-h` | Show help message and exit. |
| `--dpi DPI` | Rasterization resolution (default: 300). |
| `--page PAGE` | Grade only the specified page. |
| `--all-pages` | Grade all pages in the PDF. |
| `--bubble-inset BUBBLE_INSET` | Shrink sampling radius to avoid edges. |
| `--abs-thresh ABS_THRESH` | Absolute fill threshold (0–1) to count a bubble as filled. |
| `--margin MARGIN` | Minimum difference between best and second-best bubble to avoid MULTI classification. |
| `--key KEY` | Answer key file with A–F sections. |
| `--csv CSV` | Write detailed per-page question CSV. |
| `--summary-csv SUMMARY_CSV` | Write combined summary CSV for all pages. |
| `--write-graded-pdf` | Generate graded PDF output with overlay marks. |
| `--debug-dir DEBUG_DIR` | Write diagnostic images to this directory. |
| `--outdir OUTDIR` | Where graded PDFs/CSVs should be saved. |
| `--jpeg-quality JPEG_QUALITY` | JPEG quality (1–100) for background images. |
| `--grayscale` | Embed background in grayscale to reduce size. |
| `--skip-n SKIP_N` | After grading one page, skip N pages. |
| `--auto-skip-unreadable` | Skips all non-scannable pages
| `--points-per-question N` | Number of points per correct answer. |
| `--force-form X` | Force the exam form (A–F). |

## 📤 Output Files

### Graded PDF
Each graded PDF is named:

```
[MULTI_]##pts_originalfilename_p###
```

Examples:

- `60pts_midterm_p001.pdf`  
- `MULTI_48pts_quizA_p002.pdf`

### Per-page CSV (optional)
Contains:
- raw bubble fill scores  
- chosen answer  
- BLANK/MULTI indicators  

### Summary CSV
For all graded pages:
- form letter  
- email  
- correct / total  
- blank / multi counts  
- per-question correctness  

## 📊 Processing Flow

```
           ┌─────────────┐
           │  Input PDF  │
           └──────┬──────┘
                  ▼
          ┌───────────────┐
          │ PDF Rasterizer │
          └──────┬────────┘
                 ▼
     ┌────────────────────────┐
     │ Deskew via Fiducials   │
     └──────────┬─────────────┘
                ▼
       ┌────────────────┐
       │ Warp to Design │
       │  Homography    │
       └──────┬─────────┘
              ▼
   ┌─────────────────────┐
   │ Bubble Thresholding │
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐
   │ Score A/B/C/D Bubbles│
   └─────────┬───────────┘
             ▼
     ┌───────────────────┐
     │ Determine Status  │
     │ OK / BLANK /MULTI │
     └─────────┬─────────┘
               ▼
       ┌─────────────────┐
       │ Write Outputs   │
       │ PDF + CSV       │
       └─────────────────┘
```

## 🐞 Troubleshooting

### 🟥 “Could not detect form”
Try lowering threshold:
```
--abs-thresh 0.20 --margin 0.05
```

### 🟦 “OCR email not detected”
Increase DPI:
```
--dpi 400
```

### 🟨 Graded PDF alignment is off
Use debug mode:
```
--debug-dir debug
```
