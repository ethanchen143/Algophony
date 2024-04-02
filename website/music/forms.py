from django import forms

class CommandForm(forms.Form):
    command = forms.CharField(widget=forms.Textarea
    (attrs={'placeholder': 'pop rock song', 'class': 'form-control', 'rows': 10, 'cols': 50}), label='')