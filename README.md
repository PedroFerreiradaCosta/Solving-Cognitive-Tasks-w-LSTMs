# Elucidating Cognitive Processes Using LSTMs
An Artificial Neural Network that applies LSTM layers to solve a number of cognitive tasks.
This project was developed for my Master Thesis at C3NL, Imperial College London under the supervision of Professor Robert Leech.

For further read please see:
* https://fenix.tecnico.ulisboa.pt/cursos/mebiom/dissertacao/1409728525632094
* https://ccneuro.org/2019/Papers/ViewPapers.asp?PaperNum=1201

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip install --upgrade virtualenv
```
Create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Citation
If you find this code useful for your research, please cite:

    @article{daelucidating,
    title={Elucidating Cognitive Processes Using LSTMs},
    author={da Costa, Pedro F and Popescu, Sebastian and Leech, Robert and Lorenz, Romy}
    year={2019}}
