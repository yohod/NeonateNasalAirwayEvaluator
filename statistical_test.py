import pandas as pd
from pingouin import mwu, ttest

def p_val(title, df_norm, df_obstruct, df_surgery, alternative='less', test='mwu'):
    """
    Calculate p-values, averages, and standard deviations for statistical comparison between different groups.

    :param title: Title or region name.
    :param df_norm: DataFrame for the 'normal' group.
    :param df_obstruct: DataFrame for the 'obstruct' group.
    :param df_surgery: DataFrame for the 'surgery' group.
    :param alternative: Alternative hypothesis for the statistical test ('less', 'two-sided', or 'greater').
    :param test: Type of statistical test to perform ('mwu' for Mann-Whitney U or 'ttest' for t-test).
    :return: DataFrame containing statistical data.
    """
    # Lists to store average and standard deviation for each group
    avg_list = []
    std_list = []

    # Combine obstruct and surgery groups for CNPAS
    df_cnpas = pd.concat([df_obstruct, df_surgery], ignore_index=True)

    # Calculate average and standard deviation for each group
    for df in [df_norm, df_obstruct, df_surgery, df_cnpas]:
        avg = round(df.mean(),1)
        avg_list.append(avg)
        std = round(df.std(ddof=0),1)  # ddof=0, to normalize STD by N instead of N-1
        std_list.append(std)


    # Perform statistical tests based on the chosen test type
    if test == 'mwu':

        st1 = mwu(df_obstruct, df_norm, alternative=alternative)['p-val'].iloc[0]  # obstruct to norm
        st2 = mwu(df_surgery, df_norm, alternative=alternative)['p-val'].iloc[0]   # surgery to norm
        st3 = mwu(df_surgery, df_obstruct, alternative=alternative)['p-val'].iloc[0]   # surgery to obstruct
        st4 = mwu(df_cnpas, df_norm, alternative=alternative)['p-val'].iloc[0]   # all CNPAS to norm
    elif test == 'ttest':
        st1 = ttest(df_obstruct, df_norm, alternative=alternative)['p-val'].iloc[0]  # obstruct to norm
        st2 = ttest(df_surgery, df_norm, alternative=alternative)['p-val'].iloc[0]   # surgery to norm
        st3 = ttest(df_surgery, df_obstruct, alternative=alternative)['p-val'].iloc[0]   # surgery to obstruct
        st4 = ttest(df_cnpas, df_norm, alternative=alternative)['p-val'].iloc[0]  # all CNPAS to norm

    # Create a DataFrame containing the statistical data
    statistical_data = pd.DataFrame({
        'region': title,
        'test': test,
        'avg(1)': [avg_list[0]], 'std(1)': [std_list[0]], 'p1(mwt)': [st1],
        'avg(2)': [avg_list[1]], 'std(2)': [std_list[1]], 'p2(mwt)': [st2],
        'avg(3)': [avg_list[2]], 'std(3)': [std_list[2]], 'p3(mwt)': [st3],
        'avg(4)': [avg_list[3]], 'std(4)': [std_list[3]], 'p4(mwt)': [st4]
    })

    return statistical_data
