# Heart Rate Estimation with Face Detection

![License](https://img.shields.io/badge/license-MIT-blue.svg)

A Python project that utilizes OpenCV for face detection and webcam input to estimate the user's heart rate based on facial color changes.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Introduction

This project aims to estimate a user's heart rate by detecting their face and analyzing facial color changes caused by blood flow. It uses computer vision techniques to detect the face and signal processing to calculate the heart rate in beats per minute (BPM).

## Prerequisites

Before running this project, ensure you have the following dependencies installed:

- Python (version 3.6 or higher)
- OpenCV (version 4.x)
- NumPy
- SciPy

You can install the required Python packages using `pip`:

```bash
pip install opencv-python numpy scipy
