from django.shortcuts import render,get_object_or_404,redirect
from django.urls import reverse
from .models import AudioFile

def audio_list(request):
    audios = AudioFile.objects.all().order_by('-created_at')
    return render(request, 'gallery.html', {'audios': audios})

def delete_audio(request, audio_id):
    audio = get_object_or_404(AudioFile, id=audio_id)
    audio.file.delete()
    audio.delete()
    return redirect(reverse('gallery:audio_list'))