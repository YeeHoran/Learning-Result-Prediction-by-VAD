import pandas as pd
import statsmodels.api as sm
import pandas as pd

# Creating a sample DataFrame
data = {'column1': [1, 2, 3], 'column2': ['a', 'b', 'c'], 'source':[1,2,3]}
df = pd.DataFrame(data)

# Incorrect usage leading to AttributeError
# The 'source' attribute does not exist in a DataFrame
try:
    print(df.source)
except AttributeError as e:
    print(f"AttributeError: {e}")

# Define the observation parameters
in_class_condition = [1, 2, 3, 4, 5]
interactions_with_teacher = [1, 2, 3, 4, 5]
quiz_results = [6, 7, 8, 9, 10]

# Create a dataframe with the observation parameters
df = pd.DataFrame({'in_class_condition': in_class_condition,
                   'interactions_with_teacher': interactions_with_teacher,
                   'quiz_results': quiz_results})

# Add a constant term to the dataframe
df = sm.add_constant(df)

# Define the dependent variable
y = [0, 0, 1, 1, 1]

# Fit the regression model
model = sm.OLS(y, df).fit()

# Print the model summary
print(model.summary())
