# NBA All-Star Predictor  
Author: Anowar Mahmud  

## Description  
Built a machine learning model to predict NBA All-Star selections using SQL, Python, pandas, and scikit-learn. Created engineered features like total points, efficiency metrics, and game thresholds. Achieved 95% prediction accuracy** and added interactive predictions for custom player inputs.

## Features  
- Read and processed real NBA player stat data from 2017 using SQL and pandas  
- Cleaned and prepared dataset with height conversion and total stat calculations  
- Created new features for more accurate predictions (total points, rebounds, assists, game percentage, etc.)  
- Trained a Random Forest model to predict All-Star status  
- Measured feature importance to understand key factors in selection  
- Built a logistic regression model to show how PPG affects All-Star probability  
- Visualized results using matplotlib  
- Created an interactive prediction system for entering new player stats  

## Extra Features  
- Generates graphs showing both model feature importance and All-Star probability by PPG  

## Tools & Libraries  
- SQL  
- Python  
- pandas  
- scikit-learn  
- matplotlib  
- numpy  

## Issues Encountered  
The model was originally dominated by PPG, ignoring other key determinants from consideration.

## Acknowledgements  
- NBA dataset source ---> https://www.kaggle.com/datasets/drgilermo/nba-players-stats
- NBA Reference for All-Stars (2017) ---> https://www.basketball-reference.com/allstar/NBA_2017.html

## File Notes  
- ALLSTAR.csv – Dataset used for the project which was created through SQL.

