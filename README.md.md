Created a Sudoku Solver AI using python and Open CV to read a Sudoku puzzle from an image and solving it. 

[SudkoAI](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629). This article is a part of the series Sudoku Solver AI with OpenCV and the written tutorial I have followed:

**Part 1:** [Image Processing](https://becominghuman.ai/image-processing-sudokuai-opencv-45380715a629) \
**Part 2:** [Sudoku and Cell Extraction](https://becominghuman.ai/sudoku-and-cell-extraction-sudokuai-opencv-38b603066066) \
**Part 3:** [Solving the Sudoku](https://becominghuman.ai/part-3-solving-the-sudoku-ai-solver-13f64a090922)


## Run
```
python3 main.py
```

## Steps
1. **Import the image**
2. **Pre Processing the Image** \
   2.1 Gaussian blur: We need to gaussian blur the image to reduce noise in thresholding algorithm \
   2.2 Thresholding: Segmenting the regions of the image \
   2.3 Dilating the image: In cases like noise removal, erosion is followed by dilation.
3. **Sudoku Extraction** \
3.1 Find Contours \
3.2 Find Corners: Using Ramer Doughlas Peucker algorithm / approxPolyDP for finding corners \
3.3 Crop and Warp Image: We remove all the elements in the image except the sudoku \
3.4 Extract Cells 
4. **Interpreting the Digits** \
4.1 Import the libraries and load the dataset \
4.2 Preprocess the data \
4.3 Creating the Model \
4.4 Predicting the digits
5. **Solving the Sudoku**