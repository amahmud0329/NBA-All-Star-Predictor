import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Read the data set (Step 1)
data = pd.read_csv('ALLSTAR.csv')

# Create/Apply relevant functions (Step 2)
def height_toinches(h):
    feet, inches = h.split('-')
    return int(feet) * 12 + int(inches)

data['height_inches'] = data['height'].apply(height_toinches)

# Create more robust features (Step 3)
data['total_points'] = data['ppg'] * data['all_games']
data['total_assists'] = data['apg'] * data['all_games'] 
data['total_rebounds'] = data['rpg'] * data['all_games']
data['game_pct'] = data['all_games'] / 82
data['sufficient_games'] = (data['all_games'] >= 65).astype(int)

# Make them features to avoid syntax errors (Step 4)
features = ["ppg", "apg", "rpg", "FG%", "FT%", "3P%", "Age", "height_inches", "all_games", "total_points",
            "total_assists", "total_rebounds", "game_pct", "sufficient_games"]
Y = data["All_Star"]

# Drops any unused data 
data = data.dropna(subset=features + ["All_Star"])

# X = the dependencies Y = the target (Step 5)
X = data[features]
Y = data["All_Star"]

# Scale the X to make it take all dependencies fairly 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model to see patterns (Step 6)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.3, random_state=42, stratify = Y
)

# Model fitting and creation for Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# For logistic regression
subset = data.dropna(subset=['FG%', 'All_Star'])

X_ppg = subset["ppg"].values.reshape(-1, 1)
Y_allstar = subset["All_Star"].values

model_ppg = LogisticRegression()
model_ppg.fit(X_ppg, Y_allstar)

ppg_range = np.linspace(subset["ppg"].min(), subset["ppg"].max(), 300).reshape(-1, 1)
probabilities = model_ppg.predict_proba(ppg_range)[:, 1] 

# Accuracy Testing 
accuracy = accuracy_score(Y_test, Y_pred)
print("\n Accuracy score:", accuracy)

# Importance of each category
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n📊 Feature Importance:")
print(feature_importance)

# Data set
new_player = pd.DataFrame({
    "ppg": [25],
    "apg": [5],
    "rpg": [2],
    "FG%": [67],
    "FT%": [66],
    "3P%": [37],
    "Age": [26],
    "height_inches": [80],
    "all_games": [80]
})

# Converting data for the test player
new_player['total_points'] = new_player['ppg'] * new_player['all_games']
new_player['total_rebounds'] = new_player['rpg'] * new_player['all_games']
new_player['total_assists'] = new_player['apg'] * new_player['all_games']
new_player['game_pct'] = new_player['all_games'] / 82
new_player['sufficient_games'] = (new_player['all_games'] >= 30).astype(int)

# Printing all stats
print("\n New player stats")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(new_player)

# Using the new player's stats with the features 
new_player_features = new_player[features]
new_player_scaled = scaler.transform(new_player_features)
new_pred = model.predict(new_player_scaled)

# Message for inputs
if (new_player['ppg'].iloc[0] >= 32 or 
    (new_player['apg'].iloc[0] >= 12 and 
     new_player['rpg'].iloc[0] >= 12 and
     new_player['FG%'].iloc[0] >= 50 and
     new_player['sufficient_games'].iloc[0] >= 65)):
    print("\n A god among men...")
else:
     print("\nPrediction:", "All-Star" if new_pred[0] == 1 else "Didn't make the cut")

# Printing the insights 
print(f"\nKey insights:")
print(f"- Total Points in the Season: {new_player['total_points'].iloc[0]}")
print(f"- Total Assists in the Season: {new_player['total_assists'].iloc[0]}")
print(f"- Total Rebounds in the Season: {new_player['total_rebounds'].iloc[0]}")
print(f"- Game Percantage: {new_player['game_pct'].iloc[0]}")
print(f"- Sufficient games threshold met: {bool(new_player['sufficient_games'].iloc[0])}")

# Graph that ranks each stat based on relevance
plt.figure(figsize= (12, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color = 'skyblue')
plt.xlabel("Feature Importance Scores")
plt.title("Feature Importance from 2017 All Stars")
plt.tight_layout()

# Probability graph to see how PPG affects someone's chances for being an all-star
plt.figure(figsize=(8, 5))
plt.plot(ppg_range, probabilities, color='blue', label="Probability of All-Star")
plt.xlabel("Points Per Game (PPG)")
plt.ylabel("Probability of Being an All-Star")
plt.title("All-Star Probability vs PPG")
plt.grid(True)
plt.legend()
plt.show()
