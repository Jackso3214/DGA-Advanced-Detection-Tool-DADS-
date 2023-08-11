# DGA-Advanced-Detection-Tool-DADS-


#setup for WINDOWS using POWERSHELL
- create a virtual environment:
```
#assuming you're in your repository's folder
mkdir venv
cd venv
py -m venv .
```

- activate the virtual environment:
```
#assuming you're in your repository's folder, POWERSHELL
.\venv\Scripts\Activate.ps1
```

- download the requirements txt from the repository (if you haven't pulled it yet)
- install the required python packages
```
pip install -r requirements.txt
```


#To run the program and see the help, assuming the virtual environment is enabled/activated
Note: if try python or python3 instead of py if you get an error for python 
```
py "Python Files/main.py" -h

```
