# Content


- [Installation](#Installation)
- [How does this program work](#How-does-this-program-work)
    - [BluePrint](#BluePrint)
    - [How to run this program](#How-to-run-this-program)
    - [Files checklist](#Files-checklist)


# Installation

You can download this program by
```
git clone https://github.com/yhao21/Econ498II_Midterm.git
```
Also, please ensure you have downloaded the following packages.
```
sudo pip3.8 install pandas numpy sklearn matplotlib
```
For Windows user
```
python -m pip install pandas numpy sklearn matplotlib
```

# How does this program work
### BluePrint

```
step 1: Resize dataset, extract observations from 2018 to 2020.
        Save sample data to sample_data.csv.
step 2: Create subset of sample data.
        Save subset to sub_sample.csv.
step 3: Plot locations of each type of crime.
        Save plots in directory called figures.
step 4: Clean data:
        1) Extract useful variables.
        2) Convert Bool values.
        3) Form datetime columns.
        4) Form dummy variables.
        5) Merge crime count columns (optional).
step 5: Model selection (RF, Logistic, SVM).
        Plot accuracy rates for each split.
				
```


### How to run this program

In `RunMe.py`, you can either uncomment all steps, run them in one time, or
run each step separately. Do not skip any step.
I have made comments for core parts of each function in the program.



### Files checklist


| file name | description |
| -------: | :--------- |
|sample_data.csv_|Observations from 2018 to 2020|
|sub__sample.csv|Resized subset sample (10% observations)|
|CleanData_without_crime_count.csv|Dataset can be used to train machine, withoutcrime count|
|CleanData_with_crime_count.csv|Dataset can be used to train machine, with crime count|
|acc_trend_RF_without_crime_count.png|Accuracy rate without crime count|
|acc_trend_RF_with_crime_count|Accuracy rate with crime count|




