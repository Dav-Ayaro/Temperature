# Importing Required Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

# Step 1: Upload and Load Data
def upload_and_load_data():
    print("Please upload your Raw_birthrate_data.xlsx file...")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df = pd.read_excel(file_name, sheet_name='Sheet1')
    return df

# Step 2: Data Cleaning
def clean_data(df):
    # Clean column names
    df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '') for col in df.columns]

    # Correct column naming issues
    df.rename(columns={'state_': 'state'}, inplace=True)

    # Standardize state names
    df['state'] = df['state'].str.replace(' ', '_')

    return df

# Step 3: Feature Engineering
def create_features(df):
    # New variables
    df['higher_ed'] = df["bachelor'sdegree"] + df['graduateorprofessionaldegree']
    df['poverty_dummy'] = (df['below100percenofpovertylevel'] > df['100to199_percentofpovertylevel']).astype(int)
    df['labor_force_rate'] = df['women16to50yearsinlaborforce'] / df['birthratewomen15to50_years']

    # Convert 'year' to string
    df['year'] = df['year'].astype(str)

    return df

# Step 4: Run Regressions
def run_regressions(df):
    print("\nRunning regression with year and state fixed effects...")
    model_fe = smf.ols(
        'birthratewomen15to50_years ~ higher_ed + women16to50yearsinlaborforce + '
        'below100percenofpovertylevel + C(year) + C(state)',
        data=df
    ).fit()

    print(model_fe.summary())

    print("\nRunning regression with interaction terms...")
    # Create interaction terms
    df['poor_highered'] = df['below100percenofpovertylevel'] * df['higher_ed']
    df['poor_laborforce'] = df['below100percenofpovertylevel'] * df['labor_force_rate']

    model_interaction = smf.ols(
        'birthratewomen15to50_years ~ higher_ed + women16to50yearsinlaborforce + '
        'below100percenofpovertylevel + poor_highered + poor_laborforce + C(year) + C(state)',
        data=df
    ).fit()

    print(model_interaction.summary())

    return model_fe, model_interaction, df

# Step 5: High Birth States and Religious States
def create_state_characteristics(df):
    # Average birth rate by state
    state_avg = df.groupby('state')['birthratewomen15to50_years'].mean().sort_values(ascending=False)
    print("\nStates by average birth rate:\n", state_avg)

    # Define high birth rate states
    high_birth_cutoff = state_avg.quantile(0.75)
    df['high_birth_state'] = df['state'].apply(lambda x: 1 if state_avg[x] >= high_birth_cutoff else 0)

    # Define religious states
    religious_states_list = ['Utah', 'Alabama', 'Mississippi', 'Louisiana', 'Arkansas']
    religious_states_list = [state.replace(' ', '_') for state in religious_states_list]
    df['religious_state'] = df['state'].isin(religious_states_list).astype(int)

    return df

def run_state_characteristic_model(df):
    print("\nRunning regression with religious state interaction...")
    model_state_char = smf.ols(
        'birthratewomen15to50_years ~ higher_ed + women16to50yearsinlaborforce + '
        'below100percenofpovertylevel + below100percenofpovertylevel*religious_state + '
        'C(year) + C(state)',
        data=df
    ).fit()

    print(model_state_char.summary())

    return model_state_char

# Step 6: Visualization
def visualize_data(df, state_avg):
    plt.figure(figsize=(15, 10))

    # Plot 1: Average Birth Rate by State
    plt.subplot(2, 2, 1)
    sns.barplot(x='birthratewomen15to50_years', y='state', data=df,
                estimator=np.mean, ci=None, order=state_avg.index)
    plt.title('Average Birth Rate by State')

    # Plot 2: Birth Rate Trends Over Time
    plt.subplot(2, 2, 2)
    sns.lineplot(x='year', y='birthratewomen15to50_years', hue='high_birth_state', data=df,
                 estimator=np.mean, ci=None)
    plt.title('Birth Rate Trends Over Time')

    # Plot 3: Poverty vs Birth Rate
    plt.subplot(2, 2, 3)
    sns.scatterplot(x='below100percenofpovertylevel', y='birthratewomen15to50_years',
                    hue='religious_state', data=df)
    plt.title('Poverty and Birth Rate')

    # Plot 4: Higher Education vs Birth Rate
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='higher_ed', y='birthratewomen15to50_years',
                    hue='high_birth_state', data=df)
    plt.title('Higher Education and Birth Rate')

    plt.tight_layout()
    plt.show()

# Step 7: Model Comparison
def compare_models(model_fe, model_interaction, model_state_char):
    print("\nModel Comparison:")
    print(f"Fixed Effects Model AIC: {model_fe.aic:.2f}")
    print(f"Interaction Model AIC: {model_interaction.aic:.2f}")
    print(f"State Characteristic Model AIC: {model_state_char.aic:.2f}")

# Main Workflow
def main():
    df = upload_and_load_data()
    print(df.head())

    df = clean_data(df)
    print("\nCleaned columns:", df.columns.tolist())

    df = create_features(df)
    model_fe, model_interaction, df = run_regressions(df)

    df = create_state_characteristics(df)
    model_state_char = run_state_characteristic_model(df)

    state_avg = df.groupby('state')['birthratewomen15to50_years'].mean().sort_values(ascending=False)
    visualize_data(df, state_avg)

    compare_models(model_fe, model_interaction, model_state_char)

# Run the Program
if __name__ == "__main__":
    main()