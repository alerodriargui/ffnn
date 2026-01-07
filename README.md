# FFNN Forest Cover Type Classification

This project implements a multi-class classification system using Feed-forward Neural Networks (FFNN) as part of the Computational Intelligence course at the Universitat de les Illes Balears.

## Objective
The goal is to predict the "Cover type" (7 categories) of 30x30m forest patches based on 54 cartographic features, including elevation, slope, soil type, and wilderness areas.

## Project Structure
- `handout.ipynb`: The main deliverable containing the full implementation of tasks T1 through T9.
- `ds20.csv`: The dataset assigned for this project.
- `handout_2_ffnn.pdf`: The detailed assignment instructions.

## Tasks Implemented (T1–T9)
- **T1: Baseline Model**: Single hidden layer network with ReLU activation.
- **T2–T4: Hyperparameters**: Experiments with SGD vs. Adam optimizers and Tanh vs. ReLU activations.
- **T5–T6: Architecture**: Expansion to 2 and 3 hidden layer networks.
- **T7: Batch Size**: Analysis of performance across different batch sizes (32, 128, 512).
- **T8: Learning Rate**: Implementation of a dynamic learning rate schedule using `ReduceLROnPlateau`.
- **T9: Final Evaluation**: 5-fold cross-validation on the best architecture, aiming for ~75% accuracy.

## Requirements
To run the notebook, you need the following Python libraries installed:
- `pandas`
- `numpy`
- `torch` (PyTorch)
- `scikit-learn`
- `matplotlib`

## How to Run
1. Ensure `ds20.csv` is in the same directory as the notebook.
2. Open `handout.ipynb` using Jupyter Notebook or VS Code.
3. Run all cells to execute the training, experiments, and final cross-validation.

## Results Summary
The implementation achieves the project goal by automating a robust evaluation pipeline. The 3-layer architecture with an Adam optimizer and learning rate scheduler provides a strong baseline for the forest cover classification task.
