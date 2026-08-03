from django.shortcuts import render, redirect
from users.forms import UserRegistrationForm

def index(request):
    return render(request, 'home.html')

def login_view(request):
    form = UserRegistrationForm()
    return render(request, 'unified_auth.html', {'form': form})

def logout(request):
    request.session.flush()
    return redirect('index')