import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Set the page configuration
st.set_page_config(
    page_title="Web Test Performance Comparison",
    layout="centered",
    initial_sidebar_state="auto",
)

# Title of the app
st.title("Web Test Performance Comparison")

# Instructions
st.markdown("""
This application allows you to compare the performance of two content tests (**Test A** and **Test B**) . 
Upload a CSV file containing individual transaction-level data, and the app will display summary stats and perform a statistical analysis to determine if there's a significant difference between the two tests.

Required columns are: 

- 'ContentTest': e.g. 1/2 or a/b

- 'TransactionFaceValueUsd': The face value of the transaction in usd

- 'LeadTime': Lead time in days 

- 'IsSold': Whether the transaction was sold or not. 0 = not sold, 1 = sold

""")

st.markdown("---")

# Sidebar for file upload
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    # Check for required columns
    required_columns = ['ContentTest', 'TransactionFaceValueUsd', 'LeadTime', 'IsSold']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # Display the first few rows of the data
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Convert ContentTest to string if it's not
    if data['ContentTest'].dtype != 'object' and data['ContentTest'].dtype != 'category':
        data['ContentTest'] = data['ContentTest'].astype(str)

    # Check unique values in ContentTest
    unique_tests = data['ContentTest'].unique()
    if len(unique_tests) != 2:
        st.error(f"'ContentTest' should have exactly two unique values (e.g., '1' and '2'). Found: {unique_tests}")
        st.stop()

    # Map ContentTest to binary numeric variable
    test_mapping = {unique_tests[0]: 0, unique_tests[1]: 1}
    data['ContentTest_binary'] = data['ContentTest'].map(test_mapping)

    # Convert other columns to numeric
    numerical_columns = ['TransactionFaceValueUsd', 'LeadTime', 'IsSold']
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, set errors to NaN

    # Check for any NaN values after conversion
    if data[['ContentTest_binary'] + numerical_columns].isnull().values.any():
        st.warning("Some numerical columns contain non-numeric values or missing data. These have been set to NaN.")
        st.subheader("Data with Missing or Non-Numeric Values")
        st.write(data[data[['ContentTest_binary'] + numerical_columns].isnull().any(axis=1)])
        st.markdown("""
        **Action Required:** Please ensure that the columns **ContentTest**, **TransactionFaceValueUsd**, **LeadTime**, and **IsSold** contain valid values. 
        Remove or correct the rows with missing or invalid data before proceeding.
        """)
        st.stop()

    # Summary Statistics
    st.subheader("Summary Statistics")

    test_a = data[data['ContentTest_binary'] == 0]
    test_b = data[data['ContentTest_binary'] == 1]

    # Function to calculate summary statistics
    def summarize(group, group_name):
        summary = {
            'Group': group_name,
            'Sample Size': group.shape[0],
            'Average Lead Time': f"{group['LeadTime'].mean():.2f}",
            'Average Transaction Value (USD)': f"${group['TransactionFaceValueUsd'].mean():.2f}",
            'Conversion Rate (%)': f"{group['IsSold'].mean() * 100:.2f}%"
        }
        return summary

    summary_a = summarize(test_a, f"Test {unique_tests[0]}")
    summary_b = summarize(test_b, f"Test {unique_tests[1]}")

    summary_df = pd.DataFrame([summary_a, summary_b])
    st.table(summary_df.set_index('Group'))

    # Statistical Analysis
    st.subheader("Statistical Analysis")
    st.markdown("""
    We perform a **Logistic Regression** to determine if there's a significant difference in conversion rates between **Test A** and **Test B** while controlling for **Lead Time** and **Transaction Face Value (USD)**.
    """)

    # Prepare data for logistic regression
    X = data[['ContentTest_binary', 'LeadTime', 'TransactionFaceValueUsd']]
    y = data['IsSold']

    # Add constant term for intercept
    X = sm.add_constant(X)


    # Check if all features are numeric
    if not np.all([np.issubdtype(dtype, np.number) for dtype in X.dtypes]):
        st.error("Not all features are numeric. Please check your data and ensure all features are correctly encoded as numeric types.")
        st.stop()

    # Fit the logistic regression model
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=False)
    except Exception as e:
        st.error(f"Error fitting the logistic regression model: {e}")
        st.stop()

    # Extract the coefficient for ContentTest_binary
    coef = result.params['ContentTest_binary']
    p_value = result.pvalues['ContentTest_binary']
    odds_ratio = np.exp(coef)

    # Display the results
    st.markdown("### Results for ContentTest")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="P-Value", value=f"{p_value:.4f} ({p_value*100:.2f}%)")

    with col2:
        st.metric(label="Odds Ratio", value=f"{odds_ratio:.3f}")

    # Interpretation in layman's terms
    st.markdown("### Interpretation")

    if p_value < 0.05:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"

    if odds_ratio > 1:
        direction = "increase"
        effect_size = f"{(odds_ratio - 1) * 100:.1f}%"
    elif odds_ratio < 1:
        direction = "decrease"
        effect_size = f"{(1 - odds_ratio) * 100:.1f}%"
    else:
        direction = "no change"
        effect_size = "0%"

    st.markdown(f"""
- **P-Value**: {p_value:.4f} ({p_value*100:.2f}%)  
  This indicates that the difference between **Test {unique_tests[0]}** and **Test {unique_tests[1]}** is **{significance}**.
  
- **Odds Ratio**: {odds_ratio:.3f}  
  Being in **Test {unique_tests[1]}** is associated with a **{direction}** in the odds of conversion compared to **Test {unique_tests[0]}** by approximately **{effect_size}**.

**Note:** 
- An **Odds Ratio > 1** means higher odds of conversion for **Test {unique_tests[1]}**.
- An **Odds Ratio < 1** means lower odds of conversion for **Test {unique_tests[1]}**.
- An **Odds Ratio = 1** means no difference in odds of conversion between the tests.
""")

    # Footer notes
    st.markdown("""
---
**Note:** This analysis controls for **Lead Time** and **Transaction Face Value (USD)** to isolate the effect of **ContentTest** on the conversion rate.
    """)
else:
    st.info("Please upload a CSV file containing the following columns: **ContentTest**, **TransactionFaceValueUsd**, **LeadTime**, **IsSold**.")