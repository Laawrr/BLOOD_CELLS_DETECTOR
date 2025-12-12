# 🩸 YOLOv8 Blood Cell Detection

This project trains and tests a **YOLOv8 model** to detect blood cells in images or videos. It is designed for student-captured or microscopy images, and can be used for both academic and research purposes.

---

## **Features**
- Train YOLOv8 on custom blood cell dataset  
- Test trained model on new images or videos  
- Visualize predictions with confidence scores, bounding boxes, and annotations  
- Export trained model to multiple formats (ONNX, TorchScript, CoreML, TensorRT)

---

## **Project Structure**

```
blood_cell/
│
├─ datasets/
│   ├─ blood_cells/            # Training dataset with images and labels
│   │   └─ data.yaml           # YOLOv8 dataset config
│   └─ test/                   # Folder for testing image/videos
│       └─ images/
│
├─ runs/                       # YOLOv8 output (trained weights, results)
│
├─ BLOOD_CELLS.ipynb           # Training script
├─ TEST_MODEL.py               # Testing script for images
└─ requirements.txt            # Python dependencies
```

---

## **Clone Repository**

1. Clone the repository:
```bash
git clone https://github.com/Laawrr/BLOOD_CELLS_DETECTOR.git
cd blood_cell
```
---

## **Setup (Using Conda)**

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

2. Create a new environment:
```bash
conda create -n bloodcell python=3.12 -y
conda activate bloodcell
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "from ultralytics import YOLO; print('Ultralytics YOLO loaded successfully')"
```

---

## **Training**

1. Open BLOOD_CELLS.ipynb  in Jupyter Notebook or JupyterLab.
2. Configure training parameters in the notebook if needed (EPOCHS, BATCH_SIZE, IMG_SIZE, etc.).
3. Run the notebook cells to start training.

- Best weights are saved at:
```
runs/blood_cell_project/yolov8_blood_cells_v1/weights/best.pt
```

---

## **Testing on Images**

1. Place test images in:
```
datasets/test/images/
```

2. Run the testing script:
```bash
python TEST_MODEL.py
```

- Annotated images will be saved to:  
```
runs/blood_cell_project/test_results/
```
- Detection info (class, confidence, bounding box) will be printed.

**Note:** You can test using images or videos; one of the two is sufficient.

---

## **Exporting the Model**

Export trained model to essential formats:
```python
model.export(format="onnx")        # Cross-platform inference
model.export(format="torchscript") # PyTorch C++ or mobile
model.export(format="coreml")      # iOS/macOS
model.export(format="tensorrt")    # NVIDIA GPU optimization
```

---

## **Requirements**

`requirements.txt`:
```
ipython==8.12.3
PyYAML==6.0.3
torch==2.6.0+cu124
ultralytics==8.2.103
```
Install with:
```bash
pip install -r requirements.txt
```

---

## **Documentation Tips**

- Record confidence scores, correct detections, misdetections, and limitations.  
- This helps fulfill testing instructions and evaluates model performance accurately.

---

## **License**

This project is for **educational and research purposes**.
