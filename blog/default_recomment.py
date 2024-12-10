
from blog.models import  Average, MyDataAsset, SProduct, DProduct
import pandas as pd
from datetime import datetime
from django.http import JsonResponse# type: ignore
from blog.bank_logo import *
from django.core.serializers.json import DjangoJSONEncoder
import json
def asset_check(customer_id, user):
    # Average 테이블에서 고객 소득분위 기준 데이터 조회
    average_data = Average.objects.filter(
        stageclass=user.Stageclass,
        inlevel=user.Inlevel
    ).first()

    user_asset_data = MyDataAsset.objects.filter(CustomerID=customer_id).first()

    average_values = {
        '총자산': (average_data.asset + average_data.finance),
        '현금자산': average_data.finance,
        '수입': average_data.income,
    '   지출': average_data.spend
    }
    user_data = {
        '총자산': user_asset_data.total,
        '현금자산': user_asset_data.financial,
        '수입': user_asset_data.monthly_income,
        '지출': user_asset_data.expenses
    }

    return average_values, user_data



# 맵핑 : 적금 상품 클러스터 - 가중치 부여
def assign_cluster(stage_class, sex, age):
    if stage_class == 0:
        if sex == 'M' and age in [19, 20, 21]:
            return [5, 2, 1, 4]
        else:
            return [0, 1, 4]
    else:
        return [1, 4]
    

# 적금 상품 추천
def default_SProduct(request, user, birth_year, current_year, age, cluster):
    # Django ORM을 사용하여 데이터 가져오기
    cluster_savings = SProduct.objects.all()
    data = list(cluster_savings.values())  # ORM QuerySet을 리스트로 변환
    cluster_savings = pd.DataFrame(data) # DataFrame으로 변환
    # 결과를 저장할 빈 데이터프레임 생성 (모든 열 포함)
    final_result = pd.DataFrame(columns=cluster_savings.columns)
    

    for i in cluster:
        filtered_df = cluster_savings[cluster_savings['cluster1'] == i]
        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by=['max_preferential_rate', 'base_rate'], ascending=[False, False])
            if not sorted_df.empty:
                top_result = sorted_df.head(5)
                final_result = pd.concat([final_result, top_result], ignore_index=True)
    
    # 적금 최종 추천 
    final_recommend_json = final_result.head(5)[["DSID","product_name", "bank_name", "max_preferential_rate", "base_rate", "signup_method"]].to_dict(orient='records')
    request.session['final_recommend'] = json.dumps(final_recommend_json, cls=DjangoJSONEncoder)
    return final_result, final_recommend_json


# 사용자 예금 top cluster
def DProduct_top(user):
    # 예금 추천 처리
    cluster_scores = {i: 0 for i in range(7)}
    if user.Stageclass in [0, 1, 2, 3]:
        for cluster in [2, 4, 5, 6]:
            cluster_scores[cluster] += 1
    elif user.Stageclass in [4, 5, 6, 7]:
        for cluster in [0, 1, 2, 3, 4, 5, 6]:
            cluster_scores[cluster] += 1
    if user.Inlevel in [0, 1]:
        for cluster in [0, 1, 2, 6]:
            cluster_scores[cluster] += 1
    elif user.Inlevel in [2, 3, 4]:
        for cluster in [3, 4, 5]:
            cluster_scores[cluster] += 1

    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    top_clusters = [cluster[0] for cluster in sorted_clusters]

    return top_clusters


# 예금 상품 디폴트 추천
def default_DProduct(top_clusters, final_recommend_json):
    filtered_results = []

    for cluster in top_clusters:
        filtered_deposits_query = DProduct.objects.filter(cluster=cluster).values('dsid', 'name', 'bank', 'baser', 'maxir','method')
        filtered_results.append(pd.DataFrame(filtered_deposits_query))
    
    final_recommendations = pd.concat(filtered_results, ignore_index=True)
    # 중복 제거
    final_recommendations_drop_duplicates = final_recommendations.drop_duplicates(subset=["name", "bank", "baser", "maxir", "method"])
    top2 = final_recommendations_drop_duplicates.sort_values(by='maxir', ascending=False).head(5)
    deposit_recommend_dict = top2.to_dict(orient='records')
    
    final_recommend_display = final_recommend_json[:2]  # 적금 2개
    deposit_recommend_display = deposit_recommend_dict[:3]  # 예금 3개
    return deposit_recommend_dict, final_recommend_display, deposit_recommend_display



def get_top_data_by_customer_class(es, stageclass, inlevel):

    stageclass = stageclass
    inlevel = inlevel

    # Elasticsearch 쿼리
    query = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"customer_class.Stageclass.keyword": stageclass}},  # keyword로 정확 매칭
                    {"term": {"customer_class.Inlevel": inlevel}}
                ]
            }
        },
        "aggs": {
            "group_by_data": {
                "terms": {
                    "script": {
                    "source": """
                        def data = doc['data.product_name.keyword'].value + '|' +
                                doc['data.bank.keyword'].value + '|' +
                                doc['data.baser.keyword'].value + '|' +
                                doc['data.maxir.keyword'].value + '|' +
                                doc['data.method.keyword'].value;
                        return data;
                    """,
                    "lang": "painless"
                    },
                    "size": 3,
                    "order": {"_count": "desc"}
                },
                "aggs": {
                    "top_hits": {
                        "top_hits": {
                            "size": 1,
                            "_source": {
                            "includes": ["data", "customer_class", "timestamp"]
                            }
                        }
                    }
                }
            }
        },
        "size": 0
    }

    try:
        # Elasticsearch에서 데이터 가져오기
        response = es.search(index="ps_product_click_logs", body=query)
        # 집계 데이터 추출
        aggs_results = response.get("aggregations", {}).get("group_by_data", {}).get("buckets", [])

        # 상위 3개 데이터만 추출
        top_data = []
        for bucket in aggs_results:
            top_hit = bucket.get("top_hits", {}).get("hits", {}).get("hits", [])
            if top_hit:
                top_data.append({
                    "data": top_hit[0]["_source"]["data"],
                    "count": bucket["doc_count"]  # 해당 데이터의 카운트
                })

        return top_data

    except Exception as e:
        # 오류 처리
        return JsonResponse({"error": str(e)}, status=500)
