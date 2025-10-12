from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),  # updated from 'index' to 'dashboard'
]
