from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.main,name='main'), # 없는 경로를 호출하고 있음
    path('terms2', views.terms_content2, name='terms2'),
    path('terms3', views.terms_content3, name='terms3'),
    path('login.html', views.login_view,name='login'),
    path('report.html', views.report,name='report'),
    path('join_agree.html', views.agree,name='agree'),
    path('signup.html', views.signup,name='signup'),
    path('maintwo.html', views.summary_view,name='signup'),
    path('savings_info1.html', views.info1,name='info1'),
    path('savings_info2.html', views.info2,name='info2'),
    path('savings_info3.html', views.info3,name='info3'),
    path('savings_info4.html', views.info4,name='info4'),
    path('recommend_savings_top5.html', views.top5,name='top5'),
    path('log_click_event', views.log_click_event, name='log_click_event'),
]