from django.urls import path
from .views import audio_list,delete_audio

urlpatterns = [
    path('', audio_list, name='audio_list'),
    path('delete/<int:audio_id>/', delete_audio, name='delete_audio'),
]