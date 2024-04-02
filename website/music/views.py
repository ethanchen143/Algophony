from django.shortcuts import render, redirect
from .forms import CommandForm
from django.conf import settings
from django.apps import apps
import os
import shutil

AudioFile = apps.get_model('gallery', 'AudioFile')

# TODO: add a waiting page
def generate_audio(request):
    form = CommandForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        command = form.cleaned_data['command']
        audiocraft(command)
        return redirect('gallery:audio_list')
    return render(request, 'index.html', {'form': form})
def audiocraft(text_prompt):
    from gradio_client import Client
    client = Client("https://facebook-musicgen.hf.space/")
    result = client.predict(text_prompt, "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/audio_sample.wav", fn_index=0)
    if result:
        temp_audio_path = result[1]
        audio_filename = os.path.basename(temp_audio_path)
        new_audio_path = os.path.join(settings.MEDIA_ROOT, 'audio', audio_filename)
        os.makedirs(os.path.dirname(new_audio_path), exist_ok=True)
        # Move the file from the temp location to MEDIA_ROOT/audio
        shutil.move(temp_audio_path, new_audio_path)
        AudioFile.objects.create(
            title=f"Audio - {text_prompt}",
            file=os.path.join('audio', audio_filename)
        )