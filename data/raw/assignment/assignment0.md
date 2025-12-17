# ELEC 576 / COMP 576 – Fall 2025  
## Assignment 0  

**Due: September 16, 2025, 11:59 p.m. via Canvas**

This assignment is to help you prepare for future assignments. You must submit your report as a **PDF file** on Rice Canvas.

---

## 1. Python Machine Learning Stack (Anaconda)

You will use Python in this course. To prepare for future assignments and the final project, install Python and its packages using **Anaconda**, a high-performance distribution of Python and R including 100+ popular packages.

Follow the instructions: **Installing Anaconda**.

Confirm installation using:

```
conda list
```

You should see the list of installed packages.

You can also check using:

```
python
```

If Anaconda is installed, the startup message will include **“Continuum Analytics, Inc.”**  
Exit with:

```
quit()
```

Read the **Conda Cheat Sheet** to learn basic `conda` commands.

### Task 1  
Run:

```
conda info
```

Paste the result into your report.

---

## 2. Interactive Terminal (IPython/Jupyter)

IPython/Jupyter provides an interactive computational environment with code execution, text, math, plots, and media.

Follow:

- IPython Tutorial  
- Jupyter Documentation  

See also: **Gallery of Jupyter Notebooks**

---

## 3. Transition from MATLAB to Python

MATLAB is powerful, but Python offers better memory efficiency and speed for data science.

Read: **NumPy for MATLAB Users**

To run Python on macOS/Linux:

```
python
```

Windows users: follow **Running Python in Windows**.

Before running the examples in the tutorial, import:

```python
import numpy as np
import scipy.linalg
```

### Task 2  
Run all Python commands in the **“Linear Algebra Equivalents”** table from the tutorial using IPython.  
Paste results in your report using any matrix of your choice.

*(Optional)*  
Complete the Stanford NumPy Tutorial.

---

## 4. Plotting (Matplotlib/PyPlot)

Matplotlib is the main Python plotting library. See the **Matplotlib Gallery**.

Pyplot provides MATLAB-style plotting functions. Read the **Pyplot Tutorial**.

### Task 3  
Run this script in IPython and paste the generated figure into your report:

```python
import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [1,2,7,14])
plt.axis([0, 6, 0, 20])
plt.show()
```

### Task 4  
Use Matplotlib to create a figure of your choice.  
Paste the code and figure into your report.

---

## 5. Version Control System (GitHub)

Git helps manage changes in collaborative projects.  
Read: **Why VCS is necessary**

Register for a **GitHub Student Account** for free private repos and complete the GitHub tutorials.

### Task 5  
Paste your GitHub account (username/profile link) into your report.

---

## 6. Integrated Development Environment (IDE)

Recommended Python IDEs: **PyCharm**, **Spyder**, **Google Colab**

For PyCharm:

- Apply for a free student license  
- Install PyCharm  
- Follow PyCharm Tutorials (including VCS setup)  
- See PyCharm debugging guide  

### Task 6  
Start a new project in your preferred IDE.  
Commit and push it to GitHub as a **public project**.  
Paste the project link into your report.

---

## Submission Instructions

Submit a **PDF** containing intermediate and final results plus any necessary code on Canvas.

---

## Collaboration Policy

Collaboration on ideas is encouraged, but **write-ups must be done individually**.

---

## Plagiarism Policy

Plagiarism is not tolerated.  
Credit all external sources explicitly.
