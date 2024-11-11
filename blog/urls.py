from . import views
from django.urls import path, include
urlpatterns = [
    path('', views.main,name='main'), # 없는 경로를 호출하고 있음
    path('login.html', views.user_login,name='login'),
    path('report.html', views.report,name='report'),
]