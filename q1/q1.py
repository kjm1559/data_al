import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

def make_merge_data(users, claim, detail):
    # merge data
    claim_merge = detail.merge(claim, on='claim_id')
    # remove self response
    claim_merge = claim_merge[claim_merge.claim_user_id != claim_merge.recv_user_id]
    
    # claim 요청 횟수
    req_user_count = claim.groupby(by='claim_user_id').count().reset_index()[['claim_user_id', 'claim_id']].rename(columns={'claim_user_id':'user_id', 'claim_id':'req_count'})

    # 사용 시작, 마지막 시간 계산
    req_user_count['first_claim_at'] = claim.groupby(by='claim_user_id').claim_at.min().reset_index()['claim_at']#.rename(columns={'claim_user_id':'user_id', 'claim_at':'first_claim_at'})
    req_user_count['last_claim_at'] = claim.groupby(by='claim_user_id').claim_at.max().reset_index()['claim_at']#.rename(columns={'claim_user_id':'user_id', 'claim_at':'last_claim_at'})
    
    merged_data = users.merge(req_user_count, on='user_id', how='left')

    # 전체 요청 받은 수
    recv_total = claim_merge.groupby(by='recv_user_id').count().reset_index().rename(columns={'claim_id':'recv_total'})[['recv_user_id', 'recv_total']]
    # CLAIM 상태가 아닌 경우만 reponse로 인정
    response_rev = claim_merge[claim_merge.status != 'CLAIM'].groupby(by='recv_user_id').count().reset_index().rename(columns={'claim_id':'response'})[['recv_user_id', 'response']]
    response_rate = recv_total.merge(response_rev, on='recv_user_id', how='left')
    # 반응률 계산
    response_rate['rate'] = response_rate['response'] / response_rate['recv_total']
    response_rate = response_rate.rename(columns={'recv_user_id': 'user_id'})

    merged_data = merged_data.merge(response_rate, on='user_id', how='left')
    # 통계를 위해 Nan을 0으로 채움
    merged_data.fillna(0, inplace=True)
    
    merged_data['sum_count'] = merged_data['response'] + merged_data['req_count']
        
    # string을 datetime으로 변환
    merged_data['last_claim_at'] = merged_data['last_claim_at'].astype('datetime64')
    merged_data['first_claim_at'] = merged_data['first_claim_at'].astype('datetime64')
    
    # 유저별 사용기간 계산
    merged_data['usage_time'] = merged_data.last_claim_at - merged_data.first_claim_at
    merged_data['usage_days'] = merged_data.usage_time.dt.days + 1
    merged_data['sum_count_per_date'] = merged_data['sum_count'] / merged_data['usage_days']

    plt.clf()
    plt.hist(merged_data[merged_data.rate == 1.0].sum_count, bins=20)
    plt.title('100% response user histogram')
    plt.xlabel('usage count')
    plt.ylabel('count')
    plt.savefig('all_response_user_histogram.png')

    print('total merged data length :', len(merged_data))
    merged_data = merged_data[merged_data.req_count > 0]
    print('use more than a day length :', len(merged_data))

    plt.clf()
    plt.hist(merged_data[merged_data.rate == 1.0].sum_count, bins=20)
    plt.title('100% response user remove no request histogram')
    plt.xlabel('usage count')
    plt.ylabel('count')
    plt.savefig('all_response_user_remove_no_req_histogram.png')

    return merged_data

def least_square_linear_regression(x, y):
    A = np.array([x, np.ones(len(x))])
    w = np.linalg.lstsq(A.T, y)[0]
    return w
 
if __name__ == '__main__':
    detail = pd.read_csv("../csv_data/dutchpay_claim_detail.csv")
    claim = pd.read_csv("../csv_data/dutchpay_claim.csv")
    users = pd.read_csv("../csv_data/users.csv")

    merged_data = make_merge_data(users, claim, detail)

    # 상관분석 약한 양의 상관관계를 가지고 있음
    print('cross correlation ------------')
    corr = merged_data[['rate', 'sum_count', 'sum_count_per_date']].corr()
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns_plot = sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues', vmin=-1, vmax=1, cbar_kws={"shrink": .8})
    plt.title('Feature cross correlation matrix')
    sns_plot.get_figure().savefig('corr.png')
    print(corr['rate'])

    import scipy.stats as stats
    corr = stats.pearsonr(merged_data['rate'], merged_data['sum_count'])
    print('rate, sum_count T test p-value :', corr[1])
    corr = stats.pearsonr(merged_data['rate'], merged_data['sum_count_per_date'])
    print('rate, sum_count_per_date T test p-value :', corr[1])

    plt.clf()
    sns_plot = sns.pairplot(merged_data[['rate', 'sum_count', 'sum_count_per_date']], kind='reg')
    plt.title('pair polt')
    plt.savefig('pair_plot.png')

    sort_x = merged_data['rate'].values.copy()
    sort_x.sort()
    
    w = least_square_linear_regression(merged_data['rate'].values, merged_data['sum_count'])
    y = w[0] * sort_x + w[1]

    plt.clf()
    plt.scatter(merged_data['rate'].values, merged_data['sum_count'].values, s=1)
    plt.plot(sort_x, y, c='red')
    plt.xlabel('response rate')
    plt.ylabel('usage')
    plt.savefig('usage_rate_scatter.png')
    print('linear regression for usage')
    print('A :', w[0], 'b :', w[1])
    print(sort_x, y)

    w = least_square_linear_regression(merged_data['rate'].values, merged_data['sum_count_per_date'])
    y = w[0] * sort_x + w[1]

    plt.clf()
    plt.scatter(merged_data['rate'].values, merged_data['sum_count_per_date'].values, s=1)
    plt.plot(sort_x, y, c='red')
    plt.xlabel('response rate')
    plt.ylabel('usage per date')
    plt.savefig('usage_per_date_rate_scatter.png')
    print('linear regression for usage per date')
    print('A :', w[0], 'b :', w[1])
    print(sort_x, y)