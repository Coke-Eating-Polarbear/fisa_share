from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('agree', views.agree, name='agree'),
    path('signup/', views.signup, name='signup'),
    path('terms2', views.terms_content2, name='terms2'),
    path('terms3', views.terms_content3, name='terms3'),
    path('terms4', views.terms_content4, name='terms4'),
    path('terms5', views.terms_content5, name='terms5'),
    path('findid/', views.findid, name='findid'),
]
