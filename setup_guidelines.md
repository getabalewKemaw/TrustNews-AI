# TrustNewsAI Setup Guidelines

These steps set up the machine learning notebook environment on Windows with VS Code and Jupyter.

## 1. Open the Project

Open this folder in VS Code:

```powershell
C:\Users\Hp\Downloads\Prs\TrustNewsAI
```

For a fresh clone, open the folder where the repository was cloned.

## 2. Create the Virtual Environment

Run these commands from the project root:

```powershell
cd C:\Users\Hp\Downloads\Prs\TrustNewsAI\ml_services
python -m venv venv
```

## 3. Activate the Virtual Environment

```powershell
.\venv\Scripts\activate
```

The terminal should show `(venv)` before the path.

## 4. Install Python Packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Register the Jupyter Kernel

```powershell
python -m ipykernel install --user --name trustnewsai-venv --display-name "Python (TrustNewsAI venv)"
```

Verify that the kernel exists:

```powershell
jupyter kernelspec list
```

You should see:

```text
trustnewsai-venv
```

## 6. Add the Dataset Files

Place the Kaggle Fake and Real News Dataset files here:

```text
ml_services\data\raw\True.csv
ml_services\data\raw\Fake.csv
```

The notebook expects those exact filenames.

## 7. Open the Notebook in VS Code

Open:

```text
ml_services\notebooks\01_dataset_exploration.ipynb
```

If VS Code shows JSON instead of notebook cells:

1. Right-click the `.ipynb` file.
2. Select `Reopen Editor With...`.
3. Select `Jupyter Notebook`.

## 8. Select the Correct Kernel

In the notebook, click the kernel name in the top right.

Choose:

```text
Python (TrustNewsAI venv)
```

Do not choose the WindowsApps Python kernel.

After selecting the kernel, restart it and run the first cell.

## 9. Verify the Environment

Run this in a notebook cell:

```python
import sys
import pandas as pd

print(sys.executable)
print(pd.__version__)
```

The Python path should include:

```text
ml_services\venv\Scripts\python.exe
```

## 10. Common Fixes

If `ModuleNotFoundError: No module named 'pandas'` appears, the wrong kernel is selected. Select `Python (TrustNewsAI venv)` and restart the kernel.

If the first cell returns `(False, False)`, check that `True.csv` and `Fake.csv` are inside `ml_services\data\raw`.

If VS Code asks for a Jupyter server URL, start Jupyter from the active venv:

```powershell
cd C:\Users\Hp\Downloads\Prs\TrustNewsAI\ml_services
.\venv\Scripts\activate
jupyter notebook --no-browser --ip=127.0.0.1 --port=8888
```

Copy the `http://127.0.0.1:8888/...token=...` URL from the terminal and paste it into VS Code.
