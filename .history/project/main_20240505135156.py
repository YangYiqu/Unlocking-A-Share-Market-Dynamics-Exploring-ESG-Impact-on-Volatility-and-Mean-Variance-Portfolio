import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import gradio as gr
import tempfile



def main(file):
    df=pd.read_excel(file)
    N=20 # 分割的份数
    years = df['year'].unique()[-9:]
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(8, 6), sharex=True,sharey=True)
    axes = axes.flatten()
    # 遍历每个年份并绘制子图
    for i, year in enumerate(years):
        data_year = df[df['year'] == year]
        ax = axes[i]
        ax.scatter(data_year['esg_score'], data_year['return'],alpha=0.3)
        ax.set_title(f'Year {year}')
        ax.set_xlabel('ESG Score')
        ax.set_ylabel('Return')

        mean_esg_score = data_year['esg_score'].mean()
        mean_return = data_year['return'].mean()
        ax.axhline(y=mean_return, color='red', linestyle='--', linewidth=1, alpha=0.7)  # 画出 y 轴均值的直线
        ax.axvline(x=mean_esg_score, color='red', linestyle='--', linewidth=1, alpha=0.7)  # 画出 x 轴均值的直线

    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file1:
        plt.savefig(temp_file1.name)
        temp_file1.close()
        temp_file_path1 = temp_file1.name

    
    df_pivot = df.pivot_table(index='year', columns='id', values='return')
    df_pivot = df_pivot.dropna(axis=1)

    # 按照 'id' 分组并计算 'esg_score' 的平均值
    average_esg_score_2017_2022 = df[df["year"] != max(df["year"])].groupby('id')['esg_score'].mean()
    average_esg_score_2023 = df[df['year'] == max(df["year"])].groupby('id')['esg_score'].mean()

    # 对 esg_2017_2022 进行排序，以此为标准划分数据为10份
    valid_ids = df_pivot.columns.tolist()
    filtered_average_esg_score = average_esg_score_2017_2022.loc[valid_ids]
    sorted_esg_score=filtered_average_esg_score.sort_values()

    bins = pd.qcut(sorted_esg_score, q=N)

    result = pd.DataFrame({
        'esg_score': sorted_esg_score,
        'quantile': bins
    })

    # 根据分割后的区间进行拆分
    dfs = []
    for _, group in result.groupby(bins):
        # 删除 'quantile' 列
        group = group.drop(columns=['quantile'])
        dfs.append(group)

    # 创建一个空的 DataFrame 用于存放抽样结果
    sampled_df = pd.DataFrame(columns=dfs[0].columns)

    # 从每个分组中随机选择一行并添加到 sampled_df 中
    for group_df in dfs:
        sampled_row = group_df.sample(n=1, random_state=40)  # 从每个分组中随机选择一行
        sampled_df = pd.concat([sampled_df, sampled_row])  # 将抽样结果添加到 sampled_df 中

    esg_score_df=sampled_df
    esg_score_array=np.array(esg_score_df.values)

    selected_columns=esg_score_df.index.to_list()
    returns = df_pivot.loc[:, selected_columns]

    risk_aversion=10
    esg_coefficient=5/1000#2/1000
    risk_free=0.015
    # 分割数据集
    returns_2017_to_2022 = returns[:-1]
    returns_2023 = returns.tail(1)

    expected_returns = returns_2017_to_2022.mean()
    minimum_expected_return = min(expected_returns)
    maximum_expected_return = max(expected_returns)
    print("expected return min 20 stocks",min(expected_returns))
    print("expected return max 20 stocks",max(expected_returns))
    covariance_matrix = returns_2017_to_2022.cov(ddof=0)
    
    def generate_ptfs(returns, N):
        ptf_rs = []
        ptf_stds = []
        ptf_esgs=[]
        for i in range(N):
            weights = np.random.random(len(returns.columns))
            weights /= np.sum(weights)
            ptf_rs.append(np.sum(returns.mean() * weights))
            ptf_stds.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))))
            ptf_esgs.append(np.dot(esg_score_array.T,weights)/np.sum(weights))
        ptf_rs = np.array(ptf_rs)
        ptf_stds = np.array(ptf_stds)
        ptf_sharpes = (ptf_rs-risk_free) / ptf_stds
    
        return ptf_rs, ptf_stds,ptf_sharpes,ptf_esgs
    
    def ptf_stats(weights):
        weights = np.array(weights)
        ptf_r = np.dot(expected_returns, weights)
        ptf_std = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        ptf_esg = (np.dot(esg_score_array.T, weights)/np.sum(weights))[0]
        return np.array([ptf_r, ptf_std, (ptf_r - risk_free) / ptf_std, ptf_esg])


    #最大似然估计（MLE）
    def objective_function(weights):
        return -np.dot(expected_returns, weights) + 1/2*risk_aversion* np.dot(np.dot(weights, covariance_matrix), weights)- esg_coefficient*np.dot(esg_score_array.T,weights)/np.sum(weights)

    def min_var(weights):
        return np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))

    def sharpe_function(weights):
        return -np.dot(expected_returns, weights)/np.sqrt(np.dot(np.dot(weights, covariance_matrix), weights))

    def efficient_frontier(start_r, end_r, steps):
        target_rs = np.linspace(start_r, end_r, steps)
        target_stds = []
        for r in target_rs:
            cons= ({'type': 'eq', 'fun': lambda weights: np.dot(expected_returns, weights) - r},
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
            bnds = [(0, 1)] * len(expected_returns)
            res = minimize(min_var, x0=np.ones(len(expected_returns)) / len(expected_returns), bounds = bnds, constraints=cons)
            target_stds.append(res.fun)
        target_stds = np.array(target_stds)
        return target_rs, target_stds

    def efficient_frontier_with_esg(start_r, end_r, steps, esg_target_score):
        target_rs = np.linspace(start_r, end_r, steps)
        target_stds = []
        for r in target_rs:
            cons= ({'type': 'eq', 'fun': lambda weights: np.dot(expected_returns, weights) - r},
                    {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1},
                    {'type': 'eq', 'fun': lambda weights: np.dot(esg_score_array.T,weights)/np.sum(weights)-esg_target_score})
            bnds = [(0, 1)] * len(expected_returns)
            res = minimize(min_var, x0=np.ones(len(expected_returns)) / len(expected_returns), bounds = bnds, constraints=cons)
            target_stds.append(res.fun)
        target_stds = np.array(target_stds)
        return target_rs, target_stds
    



    optimizer = minimize(objective_function, x0=np.ones(len(expected_returns)) / len(expected_returns),
                        bounds=[(0, 1)] * len(expected_returns),constraints={'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})

    mle_weights=optimizer.x
    print("The top five weights",sorted(mle_weights)[-5:])
    print("objective function value: ",optimizer.fun)
    print("Status:",optimizer.success)

    def portfolio_metrics_test(weights, returns):
        portfolio_return = np.dot(weights, returns.mean())
        return portfolio_return

    print("Test sample by 2023")
    test_returns=portfolio_metrics_test(mle_weights, returns_2023)
    print("MLE Portfolio by all stocks - Expected Return: {:.4f}".format(test_returns))

    def objective_function_group_return(weights,return_df,esg_score_array):
        return -np.dot(return_df.mean(), weights) + 1/2*risk_aversion* np.dot(np.dot(weights, return_df.cov(ddof=0)), weights)- esg_coefficient*np.dot(esg_score_array.T,weights)/np.sum(weights)


    test_returns_group=[]
    for index,df in enumerate(dfs):
        esg_score_array_group=np.array(df.values) #128
        selected_columns_group=df.index.to_list()
        returns_group = df_pivot.loc[:, selected_columns_group]
        returns_group_2017_to_2022=returns_group[:-1]
        returns_group_2023=returns_group.tail(1)
        expected_returns_group_2017_to_2022=returns_group_2017_to_2022.mean()
        
        optimizer_group = minimize(objective_function_group_return, x0=np.ones(len(expected_returns_group_2017_to_2022)) / len(expected_returns_group_2017_to_2022),
                                args=(returns_group_2017_to_2022,esg_score_array_group),
                            bounds=[(0, 1)] * len(expected_returns_group_2017_to_2022),constraints={'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
        
        mle_weights_group=optimizer_group.x
        
        test_returns_group.append(portfolio_metrics_test(mle_weights_group, returns_group_2023))

        print("Test sample by 2023 (group {})".format(index))
        print("MLE Portfolio by all stocks - Expected Return: {:.4f}".format(portfolio_metrics_test(mle_weights_group, returns_group_2023)))




    bin_upper_limits = np.array(sorted([interval.right for interval in set(bins.values)]))
    bin_lower_limits = np.array(sorted([interval.left for interval in set(bins.values)]))
    print(bin_lower_limits)
    print(bin_upper_limits)
    plt.figure(figsize=(15,8))
    plt.grid(True)
    plt.axhline(y=test_returns, color='black', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(56, test_returns-0.01, 'The estimated returns of all the stocks', verticalalignment='top', horizontalalignment='left')
    plt.bar(bin_lower_limits, test_returns_group, width=(bin_upper_limits-bin_lower_limits), align='edge', edgecolor='black')
    plt.xlabel('ESG Bins')
    plt.ylabel('2023 Estimated Returns')
    plt.title('2023 Estimated Returns by different ESG bins')
    

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file2:
        plt.savefig(temp_file2.name)
        temp_file2.close()
        temp_file_path2 = temp_file2.name
    

    plt.figure(figsize=(15,8))
    plt.grid(True)
    plt.axhline(y=test_returns, color='black', linestyle='--', linewidth=2, alpha=0.7)
    plt.text(0, test_returns-0.01, 'The estimated returns of all the stocks', verticalalignment='top', horizontalalignment='left')
    # plt.bar(bin_lower_limits, test_returns_group, width=(bin_upper_limits-bin_lower_limits), align='edge', edgecolor='black')
    x_values = range(len(test_returns_group))

    y_values = test_returns_group

    plt.bar(x_values, y_values, width=1)  # 设置柱状图的宽度为0.5
    plt.xlabel('ESG Groups')
    plt.ylabel('2023 Estimated Returns')
    plt.title('2023 Estimated Returns by different ESG Groups')
    plt.xticks(x_values, x_values)


    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file3:
        plt.savefig(temp_file3.name)
        temp_file3.close()
        temp_file_path3 = temp_file3.name


    ptf_rs, ptf_stds, ptf_sharpes, ptf_esgs= generate_ptfs(returns_2017_to_2022, 5000)

    plt.figure(figsize=(15, 8))
    plt.scatter(ptf_stds, ptf_rs, c=ptf_sharpes, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('5000 Randomly Generated Portfolios In The Risk-Return Space')


    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file4:
        plt.savefig(temp_file4.name)
        temp_file4.close()
        temp_file_path4 = temp_file4.name

    

    opts = minimize(sharpe_function, x0=np.ones(len(expected_returns)) / len(expected_returns),
                        bounds=[(0, 1)] * len(expected_returns),constraints=[
                            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},       
                        ])

    opts_72 = minimize(sharpe_function, x0=np.ones(len(expected_returns)) / len(expected_returns),
                            bounds=[(0, 1)] * len(expected_returns),constraints=[
                                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  
                                {'type': 'eq', 'fun': lambda weights: np.dot(esg_score_array.T,weights)-72},      
                            ])

    opts_75 = minimize(sharpe_function, x0=np.ones(len(expected_returns)) / len(expected_returns),
                            bounds=[(0, 1)] * len(expected_returns),constraints=[
                                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  
                                {'type': 'eq', 'fun': lambda weights: np.dot(esg_score_array.T,weights)-75},      
                            ])


    opt_var = minimize(min_var, x0=np.ones(len(expected_returns)) / len(expected_returns),
                            bounds=[(0, 1)] * len(expected_returns),constraints=[
                                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},       
                            ])


    opt_var_72 = minimize(min_var, x0=np.ones(len(expected_returns)) / len(expected_returns),
                            bounds=[(0, 1)] * len(expected_returns),constraints=[
                                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  
                                {'type': 'eq', 'fun': lambda weights: np.dot(esg_score_array.T,weights)-72},])

    opt_var_75 = minimize(min_var, x0=np.ones(len(expected_returns)) / len(expected_returns),
                            bounds=[(0, 1)] * len(expected_returns),constraints=[
                                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  
                                {'type': 'eq', 'fun': lambda weights: np.dot(esg_score_array.T,weights)-75}] )


    target_rs, target_stds = efficient_frontier(minimum_expected_return, maximum_expected_return, 50)
    target_rs_esg_72,target_stds_esg_72 = efficient_frontier_with_esg(minimum_expected_return, maximum_expected_return, 50, 72)
    target_rs_esg_75,target_stds_esg_75 = efficient_frontier_with_esg(minimum_expected_return, maximum_expected_return, 50, 75)

    plt.figure(figsize=(15, 8))
    plt.scatter(ptf_stds, ptf_rs, c=(ptf_rs - risk_free)/ptf_stds, marker='o')
    plt.scatter(target_stds, target_rs, c=(target_rs - risk_free)/target_stds, marker='x',label='No target')
    plt.scatter(target_stds_esg_72, target_rs_esg_72, c=(target_rs_esg_72 - risk_free)/target_stds_esg_72, marker=',',label="ESG target 72")
    plt.scatter(target_stds_esg_75, target_rs_esg_75, c=(target_rs_esg_75 - risk_free)/target_stds_esg_75, marker='*',label="ESG target 75")

    plt.plot(ptf_stats(opts['x'])[1], ptf_stats(opts['x'])[0], 'r*', markersize=20.0)
    plt.plot(ptf_stats(opts_72['x'])[1], ptf_stats(opts_72['x'])[0], 'r*', markersize=20.0)
    plt.plot(ptf_stats(opts_75['x'])[1], ptf_stats(opts_75['x'])[0], 'r*', markersize=20.0)

    plt.plot(ptf_stats(opt_var['x'])[1], ptf_stats(opt_var['x'])[0], 'b*', markersize=20.0)
    plt.plot(ptf_stats(opt_var_72['x'])[1], ptf_stats(opt_var_72['x'])[0], 'b*', markersize=20.0)
    plt.plot(ptf_stats(opt_var_75['x'])[1], ptf_stats(opt_var_75['x'])[0], 'b*', markersize=20.0)

    plt.plot([0, ptf_stats(opts["x"])[1]], [risk_free, ptf_stats(opts["x"])[0]],  linestyle='--', color='grey',marker='.')
    plt.plot([0, ptf_stats(opts_72["x"])[1]], [risk_free, ptf_stats(opts_72["x"])[0]],  linestyle='--', color='grey',marker='.')
    plt.plot([0, ptf_stats(opts_75["x"])[1]], [risk_free, ptf_stats(opts_75["x"])[0]],  linestyle='--', color='grey',marker='.')


    print("sharpe values of tangent portfolios",ptf_stats(opts["x"])[2],ptf_stats(opts_72["x"])[2],ptf_stats(opts_75["x"])[2])
    plt.text(0+0.005, risk_free, 'risk free rate', verticalalignment='top', horizontalalignment='left')

    plt.grid(True)
    plt.legend()
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.xlim(0, max(target_stds_esg_75))
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier Using {} Stocks'.format(N))

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file5:
        plt.savefig(temp_file5.name)
        temp_file5.close()
        temp_file_path5 = temp_file5.name



    x_lower = min(esg_score_array)[0]
    x_upper = max(esg_score_array)[0]
    x_range = np.linspace(x_lower, x_upper, 200)
    print(x_lower)
    print(x_upper)
    # esg_target_val=60

    sharpe_list=[]
    for x in x_range:
        optimizer = minimize(sharpe_function, x0=np.ones(len(expected_returns)) / len(expected_returns),
                            bounds=[(0, 1)] * len(expected_returns),constraints=[
                                {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  
                                {'type': 'eq', 'fun': lambda weights: np.dot(esg_score_array.T,weights)-x},      
                            ])
        sharpe_list.append(-optimizer.fun)
    max_sharpe_index = sharpe_list.index(max(sharpe_list))
    max_sharpe = max(sharpe_list)



    plt.figure(figsize=(15, 8))

    plt.plot(x_range,sharpe_list,label="ESG-SR frontier")
    plt.plot(x_range[:max_sharpe_index+1], sharpe_list[:max_sharpe_index+1], linewidth=10, alpha=0.6,color='lightblue', label="ESG-nonefficient frontier")
    plt.plot(x_range[max_sharpe_index+1:], sharpe_list[max_sharpe_index+1:], linewidth=10, alpha=0.6,color='lightgreen', label="ESG-efficient frontier ")

    plt.plot(x_range[max_sharpe_index], max_sharpe, 'y*', markersize=20.0,label="Tangency Portfolio using ESG Information")

    plt.plot(ptf_stats(opts['x'])[3], ptf_stats(opts['x'])[2], 'r*', markersize=20.0,label="Tangency Portfolio ignoring ESG Information")
    plt.grid(True)
    plt.xlabel('ESG Score')
    plt.ylabel('Sharpe Ratio')
    plt.xlim(min(x_range),max(x_range))
    plt.scatter(ptf_esgs, ptf_sharpes, s=80,alpha=0.3,marker='o',label="Random portfolios")
    plt.legend()
    plt.title('ESG-efficient Frontier Using {} Stocks'.format(N))
    print("with ERG information",x_range[max_sharpe_index],max_sharpe)
    print("without ERG information",ptf_stats(opts['x'])[3], ptf_stats(opts['x'])[2])

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file6:
        plt.savefig(temp_file6.name)
        temp_file6.close()
        temp_file_path6 = temp_file6.name

    return temp_file_path1,temp_file_path4,temp_file_path5,temp_file_path6,temp_file_path2,temp_file_path3

    
gr.Interface(
    main, 
    inputs="file",  # 使用文件上传作为输入
    # outputs=[
    #     "image", "image", "image",
    #     "image", "image", "image"
    # ],  
    outputs=gr.Gallery(),
    title="Mean-Variance Optimization including ESG Information",
    description="Upload an Excel or CSV file and get the DataFrame. The file should have columns ['id','year','esg_score','return']",
).launch()