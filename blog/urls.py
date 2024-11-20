from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.main,name='main'), # 없는 경로를 호출하고 있음
    path('report_ex', views.report_ex,name='report_ex'),
    path('main/', views.main, name='main'),
    # path('report', views.report,name='report'),
    path('loginmain', views.summary_view,name='loginmain'),
    path('info1/', views.info1,name='info1'),
    path('info2/', views.info2,name='info2'),
    path('info3/', views.info3,name='info3'),
    path('info4/', views.info4,name='info4'),
    path('mypage', views.mypage,name='mypage'),
    path('recommend_savings_top5', views.top5,name='top5'),
    path('log_click_event', views.log_click_event, name='log_click_event'),
    path('spending_mbti', views.spending_mbti, name='spending_mbti'),
    path('favorite', views.favorite, name='favorite'),
    path('add_favorite/', views.add_favorite, name='add_favorite'),
    path('remove_favorite/', views.remove_favorite, name='remove_favorite'),
]