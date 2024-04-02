from django.urls import path
from .views import generate_audio

urlpatterns = [
    path('',generate_audio)
]