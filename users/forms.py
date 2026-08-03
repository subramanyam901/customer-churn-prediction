from django import forms
from .models import UserRegistrationModel


class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(label='Full Name', widget=forms.TextInput(attrs={'placeholder': 'Full Name', 'class': 'form-control'}), required=True)
    loginid = forms.CharField(label='Username', widget=forms.TextInput(attrs={'placeholder': 'Choose Username', 'class': 'form-control'}), required=True)
    email = forms.CharField(label='Email', widget=forms.EmailInput(attrs={'placeholder': 'Email Address', 'class': 'form-control'}), required=True)
    
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Create Password', 'class': 'form-control'}),
        required=True)
    
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Confirm Password', 'class': 'form-control'}),
        required=True)

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get("password")
        confirm_password = cleaned_data.get("confirm_password")
        if password != confirm_password:
            raise forms.ValidationError("Passwords do not match!")

    class Meta:
        model = UserRegistrationModel
        fields = ['name', 'loginid', 'email', 'password']