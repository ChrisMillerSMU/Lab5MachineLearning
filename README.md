# Lab5MachineLearning

## **Server/Python Set-Up**

### **Python Environment**

List of steps for setup with Python 3.8 (for compatibility with Turi) (used Rosetta terminal, see here: “Open Finder and go to "Applications", then "Utilities". Make a copy of Terminal or iTerm2, whichever is your preference. Call your copy something like "Rosetta Terminal". Right click on it and select "Get Info". Click the checkbox which says "Open using Rosetta". Launch your Rosetta Terminal and install TuriCreate like you normally would.”


FOR AN M1/M2 Mac!!! you might need to:

1. Install using a terminal that is running x86 (i386) using rosetta:
https://developer.apple.com/forums/thread/718722

2. Instead of running the first command below, you might need to run the following from the Rosetta terminals follows:


```{bash}
CONDA_SUBDIR=osx-64 conda create "turi-env" python=3.8
```

If you are not sure which instruction set the terminal is using, try the "arch" command. If you get back arm64, this is M1/M2 instruction set (will not work for Turi). If you get back "i386" or "x86_64" this is using the instruction set needed for Turi.

Instructions for non-M1/M2 Macs:

```{bash}
conda create -n "turi-env" python=3.8
```

Instructions for all Macs: 
Note that numpy must be an older version to be compatible with Turi ...

```
conda activate turi-env 
python3 -m pip install --upgrade pip 
pip3 install numpy==1.23.1  
pip3 install pandas 
pip3 install matplotlib  
pip3 install scikit-learn
pip3 install seaborn 
pip3 install jupyter 
pip3 install coremltools
pip3 install turicreate
pip3 install pymongo
pip3 install asyncio
pip3 install fastapi
pip3 install uvicorn
pip3 install motor
```

### **MongoDB Environment**

For installing mongodb, you can use home-brew and the following instructions:

Install brew:
```{bash}
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew tap mongodb/brew
brew update
brew install mongodb-community@6.0
```




When starting a MongoDB server, run the following command:


```{bash}
brew services start mongodb-community@6.0
```

When stopping a MongoDB server, run the following command:


```{bash}
brew services stop mongodb-community@6.0
```


### **Run the Server**

For running the Uvicorn/FastAPI server on `localhost` or IP address 127.0.0.1 on port 8000, the following command must be run:

```{bash}
uvicorn main:app --port 8000
```

To check FastAPI auto-generated Swagger docs for viewing a full summary of request handlers and Pydantic schemas, navigate to <a href="http://127.0.0.1:8000/docs">http://127.0.0.1:8000/docs</a>.

## **Dataset Ackowledgement**

The NSynth dataset for musical notes.

```
@InProceedings{pmlr-v70-engel17a,
  title =    {Neural Audio Synthesis of Musical Notes with {W}ave{N}et Autoencoders},
  author =   {Jesse Engel and Cinjon Resnick and Adam Roberts and Sander Dieleman and Mohammad Norouzi and Douglas Eck and Karen Simonyan},
  booktitle =    {Proceedings of the 34th International Conference on Machine Learning},
  pages =    {1068--1077},
  year =     {2017},
  editor =   {Doina Precup and Yee Whye Teh},
  volume =   {70},
  series =   {Proceedings of Machine Learning Research},
  address =      {International Convention Centre, Sydney, Australia},
  month =    {06--11 Aug},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v70/engel17a/engel17a.pdf},
  url =      {http://proceedings.mlr.press/v70/engel17a.html},
}
```
