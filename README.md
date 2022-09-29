# code-smell-classifier
Assignment 1 for CMU 18668 Data Science

To run the classifier, first create an virtual environment
```bash
python3 -m venv env
```

Then enter the virtual environment
```bash
source env/bin/activate
```

Next, install the required libraries
```bash
(env) pip3 install -r requirements.txt
```

After installing the libraries, go to the *project* folder and run the main script
```bash
(env) cd projects
(env) python3 main.py
```

To train models without GUI, use the `--gui` flag and set it to `False`, then use `--smell_type` flag to select datasets, and finally use `--classifier` flag to select classifier algorithm. 
Example:
```bash
(env) python3 main.py --gui False --smell_type fe gc dc --classifier mlp
```
