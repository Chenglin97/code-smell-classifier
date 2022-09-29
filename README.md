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

After installing the libraries, go to the *project* folder and run the *main.py* script to **launch the application**

```bash
(env) cd projects
(env) python3 main.py
```

The application should look like this:

<img width="517" alt="app preview" src="https://user-images.githubusercontent.com/95840377/193090885-c20f82c3-0b95-4abc-9bf2-ba56a7ec831c.png">


To train models without GUI, use the `--gui` flag and set it to `False`, then use `--smell_type` flag to select datasets, and finally use `--classifier` flag to select classifier algorithm. 
Example:

```bash
(env) python3 main.py --gui False --smell_type fe gc dc --classifier mlp
```


The result should look like this:

<img width="765" alt="cli results preview" src="https://user-images.githubusercontent.com/95840377/193091212-a1c74345-3cc7-4c6d-8e6d-d3d27b548431.png">

