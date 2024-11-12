from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.main,name='main'), # 없는 경로를 호출하고 있음
    path('login.html', views.login_view,name='login'),
    path('report.html', views.report,name='report'),
    path('join_agree.html', views.agree,name='agree'),
    path('signup.html', views.signup,name='signup'),
    path('maintwo.html', views.maintwo,name='signup'),
]