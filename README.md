# Gaze2Code: Token-Level OCR-Aided Eye-Tracking Framework for Code Comprehension

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-yellowgreen)

## 🧠 Overview

**Gaze2Code** is a research-oriented framework that bridges **eye-tracking data** with **source code analysis** using **token-level AOIs** (Areas of Interest) extracted via OCR. It is designed to automate and optimize cognitive attention modeling in novice programmer comprehension tasks.

> This tool facilitates accurate fixation-to-token mapping using advanced image preprocessing, OCR confidence filtering, and structured AOI token bounding boxes.

## 🔍 Features

- 📦 Token-level AOI generation using OCR (Tesseract + CLAHE + Gaussian Thresholding)
- 📈 Fixation alignment based on AOI bounding boxes
- 🧪 Supports benchmark datasets for code comprehension experiments
- 🧼 Preprocessing: CLAHE, adaptive Gaussian thresholding, morphological filtering
- 📐 Upscaling with Lanczos interpolation for OCR precision
- ✅ Confidence filtering for token acceptance
- 📊 AOI performance evaluation (before/after optimization)

## Dataset
https://www.emipws.org/dataset/

## 🗂️ Repository Structure

Author: Wudao Yang
https://scholar.google.com/citations?user=k1ivP9MAAAAJ&hl=en&oi=ao
https://orcid.org/my-orcid?orcid=0000-0001-8411-1450
