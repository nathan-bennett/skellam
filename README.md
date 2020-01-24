# skellam

Skellam is a personal python project I have been working on which allows you to predict variables that belong to the Skellam distribution.


The Skellam distribution is a frequency distribution of the difference between two independent variates following the 
same Poisson distribution (Irwin, 1937). One notable example that follows a Skellam distribution is the margin in a soccer game as this is the difference
between two Poisson distributed variables, the home team's score and the away team's score.


## Usage
```python
# Create margin variable which follows the Skellam distribution
df['margin'] = df['Home Score'] - df['Away Score'] 

# Define model and provide independent variables, the intercept and the odds
# of the home team, and also the dependent variable, the margin.
model = SkellamRegression(df[['intercept', 'Home Odds']], df['margin'])

# Initiate the training of the model
model.train()

# To produce metrics such as R2
results = model.model_performance()
results.r2()
```

## References
1. Irwin, J. (1937). The Frequency Distribution of the Difference between Two Independent Variates following the same Poisson Distribution. Journal of the Royal Statistical Society, 100(3), 415-416. doi:10.2307/2980526
 
