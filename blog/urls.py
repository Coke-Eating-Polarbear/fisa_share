from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.main,name='main'), # 없는 경로를 호출하고 있음
    path('terms2', views.terms_content2, name='terms2'),
    path('terms3', views.terms_content3, name='terms3'),
    path('terms4', views.terms_content4, name='terms4'),
    path('terms5', views.terms_content5, name='terms5'),
    path('login', views.login_view,name='login'),
    path('report_ex', views.report_ex,name='report_ex'),
    path('join_agree', views.agree,name='agree'),
    path('signup', views.signup,name='signup'),
    path('main', views.summary_view,name='main'),
    path('savings_info1', views.info1,name='info1'),
    path('savings_info2', views.info2,name='info2'),
    path('savings_info3', views.info3,name='info3'),
    path('savings_info4', views.info4,name='info4'),
    path('mypage', views.mypage,name='mypage'),
    path('recommend_savings_top5', views.top5,name='top5'),
    path('log_click_event', views.log_click_event, name='log_click_event'),
    path('spending_mbti', views.spending_mbti, name='spending_mbti'),
    path('temp', views.temp, name='temp'),
]